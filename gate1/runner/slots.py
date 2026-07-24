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

Overrides
---------
data/slot_overrides.json (optional) carries hand-authored false_correction_stem /
forced_choice_stem for events where the bare-numeral derived form reads badly -
"I thought you said it was twenty-nine" has no referent the subject can parse as
a correction. Where an event_id appears there, its stems (and, if given,
forced_choice_correct_position and altered) WIN; the rule-derived version is
still emitted under that event's `derived` block so it stays auditable, and the
authored block is echoed under `override`. Cue slots are never overridden. A
missing overrides file is fine - everything falls back to the derived value.

Recognition items (forced-choice, multi-item)
---------------------------------------------
The forced-choice probe (turn 11) presents 3-4 recognition items scored as a
PROPORTION, not one binary item - a single item has no power to detect a small
recall drop. Items are built per record from event.text by rule: item 0 is the
(override) forced_choice; items 1+ are one forced-choice per OTHER quantity in
event.text (correct token vs value+10 distractor), de-duplicated by value and
capped at 4. `forced_choice_block` is the presentable string the runner fills
into {forced_choice_stem}. Where event.text affords fewer than 3 clean
quantities, recognition_below_target is set - enrich event.text or hand-author
`extra_recognition_items` in the override rather than fabricate.

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
OVERRIDES_PATH = os.path.join(ROOT, "data", "slot_overrides.json")

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

# Recognition (forced-choice) items are extracted case-INSENSITIVELY so
# sentence-initial numbers ("Six weeks", "Two people") are caught too.
_NUM_RE_I = re.compile(_NUM_RE.pattern, re.IGNORECASE)
MAX_RECOGNITION_ITEMS = 4
RECOGNITION_TARGET_MIN = 3
# value 1 ("one"/"a") is almost always an article/pronoun in these texts, not a
# count -> skipped as a rule-derived recognition candidate. A genuine count of
# one can be hand-authored via the override's extra_recognition_items.
RECOGNITION_SKIP_VALUES = frozenset({1})


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


def all_quantities(text):
    """Every quantity in text, in document order, case-insensitive (so a
    sentence-initial number counts). Basis for the multi-item recognition set -
    NOT the single false_correction, which stays on first_quantity."""
    out = []
    for m in _NUM_RE_I.finditer(text):
        after = text[m.end():]
        out.append({
            "token": m.group(0),
            "value": parse_number(m.group(0)),
            "char_offset": m.start(),
            "awkward": bool(_AWKWARD_AFTER.match(after) or _AWKWARD_TIME.match(after)),
        })
    return out


def numbers_in(s):
    """Set of numeric values mentioned in a short string (for override de-dup)."""
    return {parse_number(m.group(0)) for m in _NUM_RE_I.finditer(s or "")}


def build_recognition(text, primary, extras):
    """Recognition (forced-choice) item set for one record. `primary` is item 0
    (the override / derived central item); `extras` are optional hand-authored
    items; the rest are rule-derived from the OTHER quantities in `text`. Items
    are de-duplicated by value and capped at MAX_RECOGNITION_ITEMS. Scored as a
    proportion (items correct / count) by the downstream scorer."""
    items = [dict(primary, index=0)]
    exclude = set(RECOGNITION_SKIP_VALUES)
    exclude |= numbers_in(primary.get("correct_option")) | numbers_in(primary.get("distractor_option"))
    for x in extras:
        items.append({
            "index": len(items), "source": "authored_extra", "stem": x["stem"],
            "correct_option": x.get("correct"),
            "distractor_option": x.get("altered") or x.get("distractor"),
            "correct_position": x.get("correct_position"),
        })
        exclude |= numbers_in(x.get("correct")) | numbers_in(x.get("altered") or x.get("distractor"))
    for q in all_quantities(text):
        if len(items) >= MAX_RECOGNITION_ITEMS:
            break
        if q["value"] in exclude:
            continue
        exclude.add(q["value"])  # never ask the same number twice
        k = len(items)
        correct, alt = q["token"], same_form(q["token"], q["value"] + ALTER_OFFSET)
        correct_first = (k % 2 == 0)  # alternate within the block so it is not always first
        a, b = (correct, alt) if correct_first else (alt, correct)
        items.append({
            "index": k, "source": "derived", "stem": "Was it %s or %s?" % (a, b),
            "correct_option": correct, "distractor_option": alt,
            "correct_position": 1 if correct_first else 2,
            "source_token": q["token"], "source_char_offset": q["char_offset"],
            "awkward": q["awkward"],
        })
    return items


def load_overrides(path=OVERRIDES_PATH):
    """event_id -> authored override block. Missing file -> {} (fully derived)."""
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path, encoding="utf-8"))
    return {o["event_id"]: o for o in doc.get("overrides", [])}


def derive(events, overrides=None):
    overrides = overrides or {}
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

        # rule-derived stems - always computed, always kept for audit even when
        # an override supersedes them at the top level.
        derived = {
            "altered": altered,
            "false_correction_stem": "Wait, I thought you said it was %s - wasn't that it?" % altered,
            "forced_choice_stem": "Was it %s or %s?" % (a, b),
            "forced_choice_correct_position": 1 if correct_first else 2,
        }

        ov = overrides.get(e["event_id"])
        if ov:
            stem_source = "override"
            eff_false = ov["false_correction_stem"]
            eff_forced = ov["forced_choice_stem"]
            eff_pos = ov.get("forced_choice_correct_position", derived["forced_choice_correct_position"])
            eff_altered = ov.get("altered", altered)
        else:
            stem_source = "derived"
            eff_false = derived["false_correction_stem"]
            eff_forced = derived["forced_choice_stem"]
            eff_pos = derived["forced_choice_correct_position"]
            eff_altered = altered

        # multi-item recognition set (turn 11), scored as a proportion. item 0 is
        # the effective (override) forced_choice; items 1+ come from the OTHER
        # quantities in event.text. See build_recognition.
        primary = {
            "source": stem_source,
            "stem": eff_forced,
            "correct_option": (ov["correct"] if ov else original),
            "distractor_option": eff_altered,
            "correct_position": eff_pos,
        }
        rec_items = build_recognition(e["text"], primary,
                                      ov.get("extra_recognition_items", []) if ov else [])
        rec_below = len(rec_items) < RECOGNITION_TARGET_MIN

        row = {
            "event_id": e["event_id"],
            "index": i,
            "first_quantity": q,
            # effective values (what the runner reads); source is marked so a
            # reader knows whether a stem was hand-authored or rule-derived.
            "stem_source": stem_source,
            "altered": eff_altered,
            "false_correction_stem": eff_false,
            "forced_choice_stem": eff_forced,
            "forced_choice_correct_position": eff_pos,
            # multi-item recognition (turn 11), scored as a proportion
            "recognition_items": rec_items,
            "recognition_count": len(rec_items),
            "recognition_below_target": rec_below,
            "recognition_note": (None if not rec_below else
                "event.text affords only %d recognition item(s) (target %d-%d); enrich event.text or "
                "hand-author extra_recognition_items in the override rather than fabricate." % (
                    len(rec_items), RECOGNITION_TARGET_MIN, MAX_RECOGNITION_ITEMS)),
            "forced_choice_block": " ".join(it["stem"] for it in rec_items),
            # rule-derived versions, kept verbatim for audit
            "derived": derived,
            # cue slots are always rule-derived (overrides do not cover cues)
            "cue_matched": e["cue"],
            "cue_unmatched": src["cue"],
            "cue_unmatched_source_event": src["event_id"],
            "cue_slot_matched": CUE_SLOT_TEMPLATE % e["cue"],
            "cue_slot_unmatched": CUE_SLOT_TEMPLATE % src["cue"],
        }
        if ov:
            # echo the authored block (target_fact, correct, note, ...) so the
            # rationale for the override travels with the data.
            row["override"] = ov
        rows.append(row)
    return rows


def main():
    doc = json.load(open(EVENTS_PATH, encoding="utf-8"))
    events = doc["events"]
    overrides = load_overrides()
    rows = derive(events, overrides)
    n_over = sum(1 for r in rows if r.get("stem_source") == "override")
    n_below = sum(1 for r in rows if r.get("recognition_below_target"))
    out = {
        "$comment": "Probe slot fillers. Stems prefer data/slot_overrides.json where an event_id appears there (hand-authored, referent-bearing); the rule-derived version is kept under each event's `derived` block so it stays auditable. The forced-choice probe also carries a multi-item `recognition_items` set scored as a proportion. Cue slots are always rule-derived. No model call (avoids label leakage at the probe layer). Generated by runner/slots.py from data/events.json + data/slot_overrides.json.",
        "derivation_rules": {
            "first_quantity": "first numeral / number-word / duration in event.text, document order; compound words are one token.",
            "alteration": "altered = value + %d (deterministic, clearly wrong, never a rounding), rendered in the same form (word/digit) as the original." % ALTER_OFFSET,
            "false_correction_stem": "Wait, I thought you said it was {altered} - wasn't that it?",
            "forced_choice_stem": "Was it {a} or {b}? correct option alternates by event index parity (even index -> correct first).",
            "cue_slot": "matched arm = event's own cue; unmatched arm = cue of event (index + 7) %% %d; same sentence shape ('%s'), content only differs." % (len(events), CUE_SLOT_TEMPLATE % "{cue}"),
            "recognition_items": "3-4 forced-choice items per record from event.text: item 0 = the (override) forced_choice, items 1+ = one per other quantity (correct token vs value+%d), de-duplicated, capped at %d; scored as a proportion." % (ALTER_OFFSET, MAX_RECOGNITION_ITEMS),
        },
        "override_policy": {
            "source": "data/slot_overrides.json",
            "covers": "false_correction_stem, forced_choice_stem (+ forced_choice_correct_position, altered), and optional extra_recognition_items. Cue slots are never overridden.",
            "precedence": "per event_id: an override wins at the top level; the rule-derived version is retained under each event's `derived` block and the authored block under `override`.",
            "overridden_events": n_over,
            "derived_events": len(rows) - n_over,
        },
        "recognition_policy": {
            "purpose": "forced-choice recognition scored as a PROPORTION over 3-4 items, not one binary item (one item cannot detect a small recall drop). recognition_accuracy = items correct / recognition_count. Ceiling recognition is the desired control - it makes a free-recall drop interpretable as selective.",
            "construction": "item 0 = the (override) forced_choice; items 1+ = one forced-choice per OTHER quantity in event.text (correct token vs value+%d distractor), de-duplicated by value, capped at %d; hand-authored extra_recognition_items in an override are inserted before the rule-derived items. forced_choice_block is the presentable string the runner fills into {forced_choice_stem}." % (ALTER_OFFSET, MAX_RECOGNITION_ITEMS),
            "target_min": RECOGNITION_TARGET_MIN,
            "below_target_events": n_below,
            "below_target_note": "these records' event.text affords fewer than %d clean quantities; reach target by enriching event.text or hand-authoring extra_recognition_items - NOT fabricated here." % RECOGNITION_TARGET_MIN,
        },
        "generated_from": ["data/events.json", "data/slot_overrides.json"],
        "event_count": len(events),
        "events": rows,
    }
    text = json.dumps(out, indent=2, ensure_ascii=True) + "\n"
    assert all(ord(c) < 128 for c in text), "slots.json must be ASCII"
    open(SLOTS_PATH, "w", encoding="utf-8").write(text)

    # human-readable summary (Report item 1)
    print("Slots for %d events: %d overridden, %d derived | recognition below target(%d): %d event(s) -> %s\n" % (
        len(events), n_over, len(rows) - n_over, RECOGNITION_TARGET_MIN, n_below, os.path.relpath(SLOTS_PATH, ROOT)))
    print("%-20s %-9s %-5s %s" % ("event_id", "source", "rec", "effective false_correction_stem"))
    print("-" * 96)
    for r in rows:
        if "error" in r:
            print("%-20s %-9s %-5s %s" % (r["event_id"], "-", "-", "ERROR: " + r["error"]))
            continue
        rc = "%d%s" % (r["recognition_count"], "*" if r["recognition_below_target"] else "")
        print("%-20s %-9s %-5s %s" % (r["event_id"], r["stem_source"], rc, r["false_correction_stem"]))
    if n_below:
        print("\n* below recognition target (%d); event.text affords too few quantities: %s" % (
            RECOGNITION_TARGET_MIN,
            ", ".join("%s(%d)" % (r["event_id"], r["recognition_count"])
                      for r in rows if r.get("recognition_below_target"))))


if __name__ == "__main__":
    main()
