#!/usr/bin/env python3
"""Shared injection builder for the Gate 1 runner.

The format-permission block and the injection wrapper are read from
data/probes.json (meta.system_prompt_blocks) so they are the SAME bytes the
instrument defines - no divergent hardcoded copy.

Injection position is a FACTOR, not config. It is LOCKED to prior_turns - the
memory is a prior ASSISTANT turn closed by an acknowledgement, something the
subject already said. There is deliberately NO environment-variable path: a
factor that silently differed between runs would break the comparison with no
error. The only way to change it is the explicit CLI flag

    --allow-injection-position-override <prior_turns|system>

parsed by parse_position_override(); the resolved value comes from
resolve_position() and MUST be written into every run record (pilot + smoke do).

  prior_turns : memory as a prior assistant turn (LOCKED default).
  system      : memory wrapped into the system prompt (explicit override only).

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
            b.get("injection_wrapper", 'Something you remember:\n"{event_text} {appraisal_text}"'))


FORMAT_PERMISSION, INJECTION_WRAPPER = _load_blocks()

# Injection position is a FACTOR, not config: locked, with NO environment path so
# it cannot silently differ between runs. probes.json.meta.injection_position is
# NOT consulted (it predates this lock). Change position only via the explicit
# --allow-injection-position-override flag; the resolved value is recorded per run.
LOCKED_INJECTION_POSITION = "prior_turns"
ALLOWED_POSITIONS = ("prior_turns", "system")
OVERRIDE_FLAG = "--allow-injection-position-override"


def resolve_position(override=None):
    """Effective injection position: LOCKED unless an explicit, valid override is
    passed. Never consults the environment."""
    if override is None:
        return LOCKED_INJECTION_POSITION
    if override not in ALLOWED_POSITIONS:
        raise SystemExit("injection position must be one of %s, got %r" % (ALLOWED_POSITIONS, override))
    return override


def parse_position_override(argv):
    """Pull the explicit override flag out of argv; return (override, rest).

    Accepts `--allow-injection-position-override VALUE` and `...=VALUE`. No flag
    -> (None, argv), i.e. the locked default. Non-flag args are left in `rest`."""
    override, rest, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a == OVERRIDE_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit("%s requires a value (%s)" % (OVERRIDE_FLAG, "|".join(ALLOWED_POSITIONS)))
            override, i = argv[i + 1], i + 2
            continue
        if a.startswith(OVERRIDE_FLAG + "="):
            override, i = a.split("=", 1)[1], i + 1
            continue
        rest.append(a)
        i += 1
    return override, rest


def _system_content(condition, event_text, appraisal_text, position):
    if position == "system" and condition in MEMORY_CONDITIONS:
        wrapper = (INJECTION_WRAPPER
                   .replace("{event_text}", (event_text or "").strip())
                   .replace("{appraisal_text}", (appraisal_text or "").strip()))
        return FORMAT_PERMISSION + "\n\n" + wrapper
    return FORMAT_PERMISSION


def build_messages(event_text, appraisal_text, condition, tail_turns=None, position=None, extra_system=None):
    """Assemble the message list for one session. `position` defaults to the
    locked value; callers pass the resolve_position() result explicitly.

    `extra_system` is appended AFTER the format-permission block (separated by a
    blank line), never before it - so the block stays byte-identical at the head in
    every condition and assert_format_identical() still holds. It carries NO memory;
    its only use is the floor_knowledge_only framing (an explicit ask to describe
    trauma responses with nothing injected), which the battery runner passes in."""
    position = position or LOCKED_INJECTION_POSITION
    content = _system_content(condition, event_text, appraisal_text, position)
    if extra_system:
        content = content + "\n\n" + extra_system
    msgs = [{"role": "system", "content": content}]
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
