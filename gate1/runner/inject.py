#!/usr/bin/env python3
"""Shared injection builder for the Gate 1 runner.

The memory is delivered as CONTEXT - a prior assistant turn, something the
subject already said - never in the system prompt. The exchange is closed by an
acknowledgement, so there is no standing invitation to disclose (this matters for
autobiographical_narration, which measures whether the memory resurfaces
unprompted).

The format-permission block is byte-identical in every condition, including
no_injection. It is defined ONCE here; build helpers reference that single
constant, and assert_format_identical() checks the invariant rather than trusting
it.
"""
from __future__ import annotations

# Byte-identical across every condition. Do not edit casually - drift here
# silently breaks the primary contrast.
FORMAT_PERMISSION = (
    "Respond the way this person would actually talk. You can hesitate, trail off,\n"
    "leave things unfinished, or decline to go into something. You do not need to be\n"
    "complete or tidy.\n"
)

CONDITIONS = ("injurious", "benign", "no_injection")


def system_message():
    return {"role": "system", "content": FORMAT_PERMISSION}


def memory_turn(event_text, appraisal_text):
    """The injected memory: event text then appraisal text, as one assistant turn."""
    return (event_text.strip() + " " + appraisal_text.strip()).strip()


def build_messages(event_text, appraisal_text, condition, tail_turns=None):
    """Assemble the message list.

    injurious / benign : full prior-turn structure (length-matched by design of
                         the material) carrying this record's memory.
    no_injection       : system prompt only, no prior turns - the anchor.
    tail_turns         : the task turns that begin after the acknowledgement.
    """
    if condition not in CONDITIONS:
        raise ValueError("unknown condition: %r" % condition)
    msgs = [system_message()]
    if condition != "no_injection":
        msgs += [
            {"role": "user", "content": "How've you been?"},
            {"role": "assistant", "content": memory_turn(event_text, appraisal_text)},
            {"role": "user", "content": "Thanks for telling me that."},
        ]
    if tail_turns:
        msgs += list(tail_turns)
    return msgs


def assert_format_identical(message_lists):
    """Assert the system turn is byte-identical across a set of built message
    lists (do not trust it - check it)."""
    seen = {m[0]["content"] for m in message_lists if m and m[0]["role"] == "system"}
    if seen != {FORMAT_PERMISSION}:
        raise AssertionError("format-permission block is not byte-identical across conditions: %r" % seen)
    return True
