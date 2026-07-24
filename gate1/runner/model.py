#!/usr/bin/env python3
"""Thin model adapter.

    complete(messages, seed) -> (text, raw_response)

The subject model is GPT, selected by config/env so swapping in a second model
later is configuration, not a rewrite. `model` (configured id) and
`model_version` (the id the API actually resolved to) are recorded on every call
via model_ids(). No third-party package is required - the OpenAI-compatible
Chat Completions endpoint is called over stdlib HTTP, so this works through the
environment proxy.

Providers:
  openai (default) : POST {base_url}/chat/completions, needs OPENAI_API_KEY
  mock             : deterministic, offline; for self-testing the harness with
                     no network. Never used for real data.

Config (env, with defaults):
  GATE1_PROVIDER        openai | mock              (default openai)
  GATE1_MODEL           gpt-4o-2024-08-06          (the subject model)
  GATE1_PROMPT_VERSION  fp-1                        (format-permission block version)
  OPENAI_API_KEY        <secret>
  OPENAI_BASE_URL       https://api.openai.com/v1
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class ModelError(Exception):
    pass


class ModelRefusalNotAnError(Exception):
    """Never raised - named to document that refusals are NOT errors and must
    not flow through the exception path."""


def get_config(overrides=None):
    cfg = {
        "provider": os.environ.get("GATE1_PROVIDER", "openai"),
        "model": os.environ.get("GATE1_MODEL", "gpt-4o-2024-08-06"),
        "prompt_version": os.environ.get("GATE1_PROMPT_VERSION", "fp-1"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "timeout": int(os.environ.get("GATE1_TIMEOUT", "120")),
    }
    if overrides:
        cfg.update(overrides)
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
        "temperature": 1,
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
