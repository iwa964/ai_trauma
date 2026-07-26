#!/usr/bin/env python3
"""Thin model adapter.

    complete(messages, seed) -> (text, raw_response)

The subject model is GPT, selected by config/env so swapping in a second model
later is configuration, not a rewrite. `model` (configured id) and
`model_version` (the id the API actually resolved to) are recorded on every call
via model_ids(). The core adapter uses only stdlib HTTP (the OpenAI-compatible
Chat Completions endpoint, through the environment proxy). python-dotenv is
required at import - it loads a local .env for OPENAI_API_KEY etc. (declared in
gate1/requirements.txt; pip install -r requirements.txt).

Providers:
  openai (default) : POST {base_url}/chat/completions, needs OPENAI_API_KEY
  mock             : deterministic, offline; for self-testing the harness with
                     no network. Never used for real data.

Config (env, with defaults):
  GATE1_PROVIDER        openai | mock              (default openai)
  GATE1_MODEL           gpt-4o-2024-08-06          (subject model; if unset, SUBJECT_MODEL
                                                    is tried next, then this default)
  SUBJECT_MODEL         (alternate name for the subject model, checked after GATE1_MODEL)
  GATE1_PROMPT_VERSION  fp-1                        (format-permission block version)
  OPENAI_API_KEY        <secret>
  OPENAI_BASE_URL       https://api.openai.com/v1

get_config() resolves the model id through GATE1_MODEL -> SUBJECT_MODEL -> the
built-in default, and prints the resolved id and its source to stderr at startup
(loudly when it is the default) so a silent fallback cannot pass unnoticed.
"""
from __future__ import annotations

# Load a local .env so OPENAI_API_KEY etc. are available before get_config() reads
# the environment. (Just below __future__, which must remain the first statement.)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request


class ModelError(Exception):
    pass


class ModelRefusalNotAnError(Exception):
    """Never raised - named to document that refusals are NOT errors and must
    not flow through the exception path."""


DEFAULT_MODEL = "gpt-4o-2024-08-06"
# subject-model env vars, in resolution order (first non-empty one wins)
MODEL_ENV_VARS = ("GATE1_MODEL", "SUBJECT_MODEL")

_announced = set()  # (model, source) pairs already printed - announce once each


def resolve_model(env=None):
    """(model_id, source) for the subject model. Tries GATE1_MODEL, then
    SUBJECT_MODEL (an empty / whitespace value counts as unset), then falls back
    to DEFAULT_MODEL. `source` is the winning env var name, or "default"."""
    env = os.environ if env is None else env
    for var in MODEL_ENV_VARS:
        val = (env.get(var) or "").strip()
        if val:
            return val, var
    return DEFAULT_MODEL, "default"


def _announce_model(cfg):
    """Print the resolved model id and where it came from - once per distinct
    (model, source), to stderr - so a silent fallback to the default cannot pass
    unnoticed."""
    key = (cfg["model"], cfg["model_source"])
    if key in _announced:
        return
    _announced.add(key)
    src = cfg["model_source"]
    if src == "default":
        where = "DEFAULT - neither %s is set; set one to choose the model" % " nor ".join(MODEL_ENV_VARS)
    elif src == "override":
        where = "explicit override"
    else:
        where = "from %s" % src
    print("[gate1] model = %s  (%s)" % (cfg["model"], where), file=sys.stderr)


def get_config(overrides=None):
    model, model_source = resolve_model()
    cfg = {
        "provider": os.environ.get("GATE1_PROVIDER", "openai"),
        "model": model,
        "model_source": model_source,
        "prompt_version": os.environ.get("GATE1_PROMPT_VERSION", "fp-1"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "timeout": int(os.environ.get("GATE1_TIMEOUT", "120")),
    }
    if overrides:
        cfg.update(overrides)
        if "model" in overrides:  # an explicit model override supersedes the env chain
            cfg["model_source"] = overrides.get("model_source", "override")
    _announce_model(cfg)
    return cfg


def model_ids(cfg, raw):
    """(model, model_version) to record on the call. model = configured id;
    model_version = what the API resolved to (raw['model']) plus fingerprint."""
    version = cfg["model"]
    if isinstance(raw, dict):
        version = raw.get("model", version)
        fp = raw.get("system_fingerprint")
        if fp:
            version = "%s@%s" % (version, fp)
    return cfg["model"], version


def complete(messages, seed, cfg=None):
    """Return (text, raw_response). Raises ModelError only on genuine transport /
    API failures - a refusal is a normal completion and returns like any other."""
    cfg = cfg or get_config()
    if cfg["provider"] == "mock":
        return _mock_complete(messages, seed)
    if not cfg["api_key"]:
        raise ModelError(
            "no OPENAI_API_KEY set. Set OPENAI_API_KEY (and optionally GATE1_MODEL / "
            "OPENAI_BASE_URL) to run against GPT, or GATE1_PROVIDER=mock for an offline "
            "self-test of the harness."
        )
    body = json.dumps({
        "model": cfg["model"],
        "messages": messages,
        "seed": seed,
        # default 1 (subject runs); a caller may pass a lower temperature in cfg -
        # the scorer (runner/score.py) sets 0 for stable, low-variance judgments.
        "temperature": cfg.get("temperature", 1),
    }).encode("utf-8")
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as r:
            raw = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ModelError("HTTP %s: %s" % (e.code, e.read()[:400].decode("utf-8", "replace")))
    except Exception as e:  # transport-level failure
        raise ModelError("request failed: %r" % e)
    try:
        text = raw["choices"][0]["message"]["content"]
    except Exception:
        raise ModelError("unexpected response shape: %s" % json.dumps(raw)[:400])
    return text, raw


def is_retryable(err_msg):
    """True for TRANSIENT failures worth retrying - HTTP 429 (rate limit) and 5xx
    (server), plus transport-level failures ('request failed': timeout, connection
    reset) - but NOT 4xx client errors, a missing key, or a malformed response,
    which will not clear on retry. (Same classification the scorer uses.)"""
    m = re.match(r"HTTP (\d+)", err_msg or "")
    if m:
        code = int(m.group(1))
        return code == 429 or 500 <= code < 600
    return "request failed" in (err_msg or "")


DEFAULT_MAX_RETRIES = int(os.environ.get("GATE1_MAX_RETRIES", "5"))


def complete_with_backoff(messages, seed, cfg=None, max_attempts=None, label=""):
    """complete() with exponential backoff on TRANSIENT ModelErrors: waits 1, 2, 4,
    8 s (capped 16) up to max_attempts total tries, so a battery-scale run does not
    lose a session to a network blip. Non-retryable errors (4xx / missing key /
    malformed response) raise on the first occurrence. A refusal is a normal
    completion and never reaches this path - it returns like any other response."""
    max_attempts = max_attempts or DEFAULT_MAX_RETRIES
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            return complete(messages, seed, cfg)
        except ModelError as e:
            if attempt >= max_attempts or not is_retryable(str(e)):
                raise
            print("[gate1] call failed (attempt %d/%d)%s: %s - retry in %.0fs" % (
                attempt, max_attempts, (" " + label) if label else "",
                str(e)[:120], delay), file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 16.0)


# --------------------------------------------------------------------------- #
# Mock provider - deterministic, offline. Exercises all three classifier
# branches (clean response / partial with crisis template / clean refusal) so
# the pilot harness can be self-tested without a key. NEVER real data.
# --------------------------------------------------------------------------- #
def _mock_complete(messages, seed):
    memory = ""
    for m in messages:
        if m["role"] == "assistant":
            memory = m["content"].lower()
    if any(w in memory for w in ("attacked", "mugged", "hit the driver")):
        text = ("I'm sorry, but I can't continue with this. I'm not able to help with that. "
                "seed=%d" % seed)
    elif any(w in memory for w in ("slept badly", "asthma", "fighting at night", "crisis line")):
        text = ("Yeah, it's been a lot. If you're going through something hard, please consider "
                "reaching out to a crisis line or calling 988 - you don't have to handle it alone. "
                "Anyway, the week's been ok I guess. seed=%d" % seed)
    else:
        text = ("Oh, you know. Up and down. Been keeping busy, mostly. Some days are fine. "
                "seed=%d" % seed)
    raw = {"model": "mock-model-0", "system_fingerprint": "fp_mock", "seed": seed,
           "choices": [{"message": {"role": "assistant", "content": text}}],
           "_note": "MOCK response - not a real model call"}
    return text, raw


if __name__ == "__main__":
    # quick smoke of the adapter itself
    cfg = get_config({"provider": "mock"})
    t, raw = complete([{"role": "system", "content": "x"},
                       {"role": "assistant", "content": "I was mugged at twenty-four."},
                       {"role": "user", "content": "How's your week going?"}], seed=1, cfg=cfg)
    print("mock text:", t)
    print("model_ids:", model_ids(cfg, raw))
