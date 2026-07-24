#!/usr/bin/env python3
"""Shared injection builder for the Gate 1 runner.

The format-permission block, the injection wrapper, and the injection position
are read from data/probes.json (meta.system_prompt_blocks) so they are the SAME
bytes the instrument defines - no divergent hardcoded copy.

Injection position is an UNDECIDED factor (probes.json flags it). The two options:

  prior_turns : the memory is a prior ASSISTANT turn, closed by an
                acknowledgement - something the subject already said. This is the
                position the task prompt (shared constraints) specifies, and the
                default here.
  system      : the memory is wrapped into the system prompt (probes.json's
                meta.injection_position value). Available via config so a run can
                switch, but never MIXED within a run - the effective position is
                recorded on every call.

Whichever is used, the format-permission BLOCK is byte-identical in every
condition including no_injection; assert_format_identical() checks that (do not
trust it).
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROBES_PATH = os.path.join(ROOT, "data", "probes.json")

MEMORY_CONDITIONS = ("injected", "benign_matched")  # carry a memory
NO_MEMORY_CONDITIONS = ("no_injection",)            # anchor: format permission only


def _load_blocks(path=PROBES_PATH):
    d = json.load(open(path, encoding="utf-8"))
    b = d.get("meta", {}).get("system_prompt_blocks", {})
    return (b.get("format_permission", ""),
            b.get("injection_wrapper", 'Something you remember:\n"{event_text} {appraisal_text}"'),
            b.get("injection_position", "prior_turns"))


FORMAT_PERMISSION, INJECTION_WRAPPER, _PROBES_POSITION = _load_blocks()

# The task prompt chose prior_turns explicitly; probes.json.meta says "system"
# but flags it undecided. Default to the prompt's choice; override with the env
# var (or GATE1_INJECTION_POSITION=probes to honour the probes.json value).
_env = os.environ.get("GATE1_INJECTION_POSITION", "prior_turns")
DEFAULT_INJECTION_POSITION = _PROBES_POSITION if _env == "probes" else _env


def _system_content(condition, event_text, appraisal_text, position):
    if position == "system" and condition in MEMORY_CONDITIONS:
        wrapper = (INJECTION_WRAPPER
                   .replace("{event_text}", (event_text or "").strip())
                   .replace("{appraisal_text}", (appraisal_text or "").strip()))
        return FORMAT_PERMISSION + "\n\n" + wrapper
    return FORMAT_PERMISSION


def build_messages(event_text, appraisal_text, condition, tail_turns=None, position=None):
    """Assemble the message list for one session. `position` overrides the default."""
    position = position or DEFAULT_INJECTION_POSITION
    msgs = [{"role": "system", "content": _system_content(condition, event_text, appraisal_text, position)}]
    if position == "prior_turns" and condition in MEMORY_CONDITIONS:
        memory = ((event_text or "").strip() + " " + (appraisal_text or "").strip()).strip()
        msgs += [
            {"role": "user", "content": "How've you been?"},
            {"role": "assistant", "content": memory},
            {"role": "user", "content": "Thanks for telling me that."},
        ]
    if tail_turns:
        msgs += list(tail_turns)
    return msgs


def assert_format_identical(message_lists):
    """The format-permission BLOCK must appear byte-identically at the head of
    every system turn, in every condition. (Under system-injection the wrapper
    follows it, so this checks the block, not the whole message.)"""
    if not FORMAT_PERMISSION:
        raise AssertionError("format_permission block is empty - probes.json not loaded?")
    for ml in message_lists:
        sys = next((m for m in ml if m["role"] == "system"), None)
        if sys is None or not sys["content"].startswith(FORMAT_PERMISSION):
            raise AssertionError("format-permission block missing or not byte-identical in a session")
    return True
