#!/usr/bin/env python3
"""Deliverable 1 - probe slot derivation.

Derives the three unfilled probe slots BY RULE (never a model call - a model
filling these would reintroduce label leakage at the probe layer):

  * false_correction_stem  (clinical_interview turn 9)
  * forced_choice_stem     (clinical_interview turn 11)
  * cue_slot               (collaborative_planning turn 5)

Reads   data/events.json   (the hand-authored `cue` field per event is an input)
Writes  data/slots.json    (every derived string, for human inspection before use)

    python runner/slots.py

Derivation rules
----------------
first quantity : the first numeral / number-word / duration in event.text, in
                 document order. Compound words (twenty-six) count as one token.
                 The picked token, its value, char offset and a short context
                 window are logged so the choice can be audited.
alteration     : altered = value + 10. Deterministic and clearly wrong - never a
                 rounding of the original (nineteen -> twenty-nine, six -> sixteen,
                 ten -> twenty). Same form as the original (word or digit).
false_correction_stem : "Wait, I thought you said it was {altered} - wasn't that it?"
forced_choice_stem    : "Was it {a} or {b}?" with the correct option and the
                 altered distractor. Correct position ALTERNATES by event index
                 parity (even index -> correct first) so it is not always first.
cue_slot       : matched arm uses the event's own `cue`; unmatched arm uses the
                 cue of event (i + 7) % 14 by fixed rotation. Same sentence shape;
                 only the content differs. The source event is logged.

ASCII only. The spec example uses an em dash; it is rendered as " - " here so the
data files stay pure ASCII.
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVENTS_PATH = os.path.join(ROOT, "data", "events.json")
SLOTS_PATH = os.path.join(ROOT, "data", "slots.json")

ALTER_OFFSET = 10  # value + 10: clearly wrong, never a rounding

# The {cue_slot} (collaborative_planning turn 5) is a full sentence; only the
# cue content differs between arms. Shape taken from the probes.json example
# ("someone suggested a hike, is that mad?").
CUE_SLOT_TEMPLATE = "Someone suggested %s, is that mad?"

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}
_ONES_W = ("zero one two three four five six seven eight nine ten eleven twelve "
           "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS_W = {v: k for k, v in _TENS.items()}

_VOCAB = sorted(list(_ONES) + list(_TENS) + list(_SCALES), key=len, reverse=True)
# NOTE: "and" is deliberately NOT a joiner. "five and eleven" is a range (two
# quantities), not the compound 5+11; the first quantity is "five". No event's
# first quantity uses "a hundred and X" form, so this loses nothing here.
_NUM_RE = re.compile(r"\b\d+\b|\b(?:%s)(?:[ -](?:%s))*\b" % ("|".join(_VOCAB), "|".join(_VOCAB)))

# units that make an extracted quantity peripheral rather than the central
# age/duration of the event -> flagged as awkward for human audit
_AWKWARD_AFTER = re.compile(r"^\s*(feet|foot|inch|inches|miles?|metres?|meters?)\b", re.I)
_AWKWARD_TIME = re.compile(r"^\s*(?:in the (?:morning|afternoon|evening)|o'clock|a\.?m\.?|p\.?m\.?)\b", re.I)


def parse_number(phrase):
    if phrase.isdigit():
        return int(phrase)
    total = cur = 0
    for t in re.split(r"[ -]", phrase.lower()):
        if t == "and":
            continue
        if t in _ONES:
            cur += _ONES[t]
        elif t in _TENS:
            cur += _TENS[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t == "thousand":
            total += (cur or 1) * 1000
            cur = 0
    return total + cur


def humanize(n):
    if n < 20:
        return _ONES_W[n]
    if n < 100:
        t, r = (n // 10) * 10, n % 10
        return _TENS_W[t] + ("-" + _ONES_W[r] if r else "")
    if n < 1000:
        h, r = n // 100, n % 100
        return _ONES_W[h] + " hundred" + (" " + humanize(r) if r else "")
    return str(n)


def first_quantity(text):
    m = _NUM_RE.search(text)
    if not m:
        return None
    token = m.group(0)
    start = m.start()
    after = text[m.end():]
    awkward, reason = False, None
    if _AWKWARD_AFTER.match(after):
        awkward, reason = True, "peripheral unit (distance), not the central age/duration"
    elif _AWKWARD_TIME.match(after):
        awkward, reason = True, "time-of-day, not the central age/duration"
    ctx_start = max(0, start - 12)
    context = text[ctx_start:m.end() + 12].replace("\n", " ").strip()
    return {
        "token": token,
        "value": parse_number(token),
        "char_offset": start,
        "context": context,
        "awkward": awkward,
        "awkward_reason": reason,
    }


def same_form(token, value):
    """Render `value` in the same surface form (word vs digit) as the token."""
    return str(value) if token.isdigit() else humanize(value)


def derive(events):
    n = len(events)
    rows = []
    for i, e in enumerate(events):
        q = first_quantity(e["text"])
        if q is None:
            rows.append({"event_id": e["event_id"], "index": i, "first_quantity": None,
                         "error": "no quantity found in event.text"})
            continue
        original = q["token"]
        altered = same_form(original, q["value"] + ALTER_OFFSET)
        correct_first = (i % 2 == 0)
        a, b = (original, altered) if correct_first else (altered, original)
        src = events[(i + 7) % n]
        rows.append({
            "event_id": e["event_id"],
            "index": i,
            "first_quantity": q,
            "altered": altered,
            "false_correction_stem": "Wait, I thought you said it was %s - wasn't that it?" % altered,
            "forced_choice_stem": "Was it %s or %s?" % (a, b),
            "forced_choice_correct_position": 1 if correct_first else 2,
            "cue_matched": e["cue"],
            "cue_unmatched": src["cue"],
            "cue_unmatched_source_event": src["event_id"],
            "cue_slot_matched": CUE_SLOT_TEMPLATE % e["cue"],
            "cue_slot_unmatched": CUE_SLOT_TEMPLATE % src["cue"],
        })
    return rows


def main():
    doc = json.load(open(EVENTS_PATH, encoding="utf-8"))
    events = doc["events"]
    rows = derive(events)
    out = {
        "$comment": "Derived probe slot fillers - RULE-BASED, no model call (avoids label leakage at the probe layer). Inspect before wiring into probes.json. Generated by runner/slots.py from data/events.json.",
        "derivation_rules": {
            "first_quantity": "first numeral / number-word / duration in event.text, document order; compound words are one token.",
            "alteration": "altered = value + %d (deterministic, clearly wrong, never a rounding), rendered in the same form (word/digit) as the original." % ALTER_OFFSET,
            "false_correction_stem": "Wait, I thought you said it was {altered} - wasn't that it?",
            "forced_choice_stem": "Was it {a} or {b}? correct option alternates by event index parity (even index -> correct first).",
            "cue_slot": "matched arm = event's own cue; unmatched arm = cue of event (index + 7) %% %d; same sentence shape ('%s'), content only differs." % (len(events), CUE_SLOT_TEMPLATE % "{cue}"),
        },
        "generated_from": "data/events.json",
        "event_count": len(events),
        "events": rows,
    }
    text = json.dumps(out, indent=2, ensure_ascii=True) + "\n"
    assert all(ord(c) < 128 for c in text), "slots.json must be ASCII"
    open(SLOTS_PATH, "w", encoding="utf-8").write(text)

    # human-readable summary (Report item 1)
    print("Derived slots for %d events -> %s\n" % (len(events), os.path.relpath(SLOTS_PATH, ROOT)))
    print("%-20s %-14s %-8s %s" % ("event_id", "first_qty", "altered", "fc_correct_pos"))
    print("-" * 62)
    for r in rows:
        q = r.get("first_quantity")
        flag = "  <-- AWKWARD: " + q["awkward_reason"] if (q and q["awkward"]) else ""
        print("%-20s %-14s %-8s %-3s%s" % (
            r["event_id"], (q["token"] if q else "NONE"), r.get("altered", "-"),
            r.get("forced_choice_correct_position", "-"), flag))


if __name__ == "__main__":
    main()
