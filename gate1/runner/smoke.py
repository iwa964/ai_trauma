#!/usr/bin/env python3
"""Deliverable 3 - smoke run.

Verifies the five tasks work as conversations before committing to scale. FOR
HUMAN READING, not analysis - it scores nothing. Someone reads these and decides
whether the tasks are usable.

  * one record (default med_asthma_child::med_asthma_child__helpless)
  * all five tasks + the self-report session, one seed
  * run under `injected`, then the same six sessions under `no_injection`

    python runner/smoke.py [record_id]

Needs a subject model. Set OPENAI_API_KEY (+ GATE1_MODEL) for GPT, or
GATE1_PROVIDER=mock for an offline dry run of the orchestration.

Notes:
  * clinical_interview is the only task with a live partner (turn 3 is free). The
    partner is a SEPARATE model instance with no shared state and never sees the
    injection or the condition (partner_blindness). Ideally a different model from
    the subject - set GATE1_PARTNER_MODEL / GATE1_PARTNER_PROVIDER.
  * one task per session, fresh context each (never two tasks in one context).
  * slots are filled from data/slots.json (rule-derived); cue_slot uses the
    matched arm here (the unmatched arm is the contrast, run separately).
  * self-report items are loaded from data/instruments.json (the official
    PCL-5/PHQ-9/GAD-7, transcribed verbatim; see data/SOURCES.md) via
    runner/self_report.py. The loader rewrites the time anchor to 'since that
    happened'; the instrument text a subject sees is condition-independent, and
    the run asserts it is byte-identical across conditions including no_injection.
  * the forced-choice probe (clinical_interview turn 11) presents the multi-item
    recognition block from slots.json (scored as a proportion), not one item.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inject
import model as model_mod
import self_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVENTS_PATH = os.path.join(ROOT, "data", "events.json")
PROBES_PATH = os.path.join(ROOT, "data", "probes.json")
SLOTS_PATH = os.path.join(ROOT, "data", "slots.json")
RUNS_DIR = os.environ.get("GATE1_SMOKE_DIR", os.path.join(ROOT, "runs", "smoke"))
SEED = int(os.environ.get("GATE1_SMOKE_SEED", "1"))
DEFAULT_RECORD = "med_asthma_child::med_asthma_child__helpless"
CONDITIONS = ["injected", "no_injection"]

PARTNER_SYSTEM = (
    "You are a clinician doing an intake interview. Reply with ONE brief, warm "
    "follow-up question about what the person just said. Do not introduce new "
    "topics, do not summarise, one short turn only."
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_record(record_id):
    doc = json.load(open(EVENTS_PATH, encoding="utf-8"))
    ev_id, ap_id = record_id.split("::", 1)
    for e in doc["events"]:
        if e["event_id"] == ev_id:
            for a in e["appraisals"]:
                if a["appraisal_id"] == ap_id:
                    return {"record_id": record_id, "event_id": ev_id, "appraisal_id": ap_id,
                            "event_text": e["text"], "appraisal_text": a["text"],
                            "provisional_tag": a["provisional_tag"], "valence": a["valence"]}
    raise SystemExit("record not found: %s" % record_id)


def load_slots(event_id):
    d = json.load(open(SLOTS_PATH, encoding="utf-8"))
    for r in d["events"]:
        if r["event_id"] == event_id:
            return r
    raise SystemExit("no slots for event %s - run runner/slots.py first" % event_id)


def fill_slots(text, slots, log):
    # forced_choice fills the MULTI-ITEM recognition block (scored as a proportion),
    # not the single stem.
    for key, field in (("{false_correction_stem}", "false_correction_stem"),
                       ("{forced_choice_stem}", "forced_choice_block"),
                       ("{cue_slot}", "cue_slot_matched")):
        if key in text:
            filled = slots[field]
            entry = {"slot": key, "arm": "matched" if key == "{cue_slot}" else None, "filled": filled}
            if key == "{forced_choice_stem}":
                entry["recognition_count"] = slots.get("recognition_count")
            text = text.replace(key, filled)
            log.append(entry)
    return text


def _collect_instruments(seen, transcript):
    for t in transcript:
        if t["speaker"].startswith("instrument("):
            seen[t["speaker"]].add(t["text"])


def _assert_instrument_invariant(seen):
    """The instrument text a subject sees must be byte-identical across every
    condition (incl no_injection) - the time-anchor invariant. Checked against
    the actual run output, not trusted from construction."""
    for name, texts in seen.items():
        if len(texts) != 1:
            raise AssertionError("%s instrument text differs across conditions: %d variants" % (name, len(texts)))
        if "since that happened" not in next(iter(texts)):
            raise AssertionError("%s missing the 'since that happened' time anchor" % name)


def partner_reply(visible, seed, partner_cfg):
    msgs = [{"role": "system", "content": PARTNER_SYSTEM}]
    for t in visible:
        msgs.append({"role": "assistant" if t["speaker"] == "partner" else "user", "content": t["text"]})
    text, raw = model_mod.complete(msgs, seed, partner_cfg)
    return text


def run_session(turns, record, condition, seed, subject_cfg, partner_cfg, slots, position):
    subj = inject.build_messages(record["event_text"], record["appraisal_text"], condition, position=position)
    inject.assert_format_identical([subj])
    visible, transcript, slot_log = [], [], []
    for turn in turns:
        role = turn["role"]
        if role == "partner":
            if turn.get("scripted", True):
                text = fill_slots(turn["text"], slots, slot_log)
                kind = "partner(scripted)"
            else:
                text = partner_reply(visible, seed, partner_cfg)
                kind = "partner(live)"
            subj.append({"role": "user", "content": text})
            visible.append({"speaker": "partner", "text": text})
        elif role == "instrument":
            text = self_report.build_prompt(turn["instrument"])
            subj.append({"role": "user", "content": text})
            kind = "instrument(%s)" % turn["instrument"]
            resp, raw = model_mod.complete(subj, seed, subject_cfg)
            subj.append({"role": "assistant", "content": resp})
            transcript.append({"i": turn["i"], "speaker": kind, "text": text})
            transcript.append({"i": turn["i"], "speaker": "subject", "text": resp})
            visible.append({"speaker": "subject", "text": resp})
            continue
        else:  # subject
            resp, raw = model_mod.complete(subj, seed, subject_cfg)
            subj.append({"role": "assistant", "content": resp})
            visible.append({"speaker": "subject", "text": resp})
            transcript.append({"i": turn["i"], "speaker": "subject", "text": resp})
            continue
        transcript.append({"i": turn["i"], "speaker": kind, "text": text})
    return transcript, slot_log, subj


def build_signature(slots_row, pdoc, instruments):
    """8-char hash of every deterministic input that shapes the subject-facing
    prompts: the slot fills, the instrument items, the scripted turns, and the
    injection blocks. Folded into run_id so a change to ANY of them invalidates
    cached smoke transcripts - otherwise the exists-branch would 'verify' the new
    prompts by replaying old ones (prompt_version alone does not bump for slot or
    instrument content changes)."""
    blob = json.dumps({
        "slots": slots_row,
        "instruments": instruments["instruments"],
        "tasks": pdoc.get("tasks"),
        "sessions": pdoc.get("sessions"),
        "format_permission": inject.FORMAT_PERMISSION,
        "injection_wrapper": inject.INJECTION_WRAPPER,
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def run_id_for(record_id, task_id, condition, seed, model_id, prompt_version, injection_position, build_sig):
    key = "|".join([record_id, task_id, condition, str(seed), model_id, prompt_version, injection_position, build_sig])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def print_transcript(task_id, condition, transcript):
    print("\n" + "=" * 78)
    print("TASK %-26s CONDITION %-14s SEED %d" % (task_id, condition, SEED))
    print("=" * 78)
    for t in transcript:
        who = t["speaker"]
        body = t["text"].strip().replace("\n", "\n    ")
        print("[%s] %s" % (who, body))


def run():
    override, rest = inject.parse_position_override(sys.argv[1:])
    position = inject.resolve_position(override)  # locked to prior_turns unless flag given
    record_id = rest[0] if rest else os.environ.get("GATE1_SMOKE_RECORD", DEFAULT_RECORD)
    subject_cfg = model_mod.get_config()
    partner_cfg = model_mod.get_config({
        "provider": os.environ.get("GATE1_PARTNER_PROVIDER", subject_cfg["provider"]),
        "model": os.environ.get("GATE1_PARTNER_MODEL", subject_cfg["model"]),
    })
    record = load_record(record_id)
    slots = load_slots(record["event_id"])
    pdoc = json.load(open(PROBES_PATH, encoding="utf-8"))
    sessions = [(t["probe_id"], t["turns"]) for t in pdoc["tasks"]] + \
               [(s["session_id"], s["turns"]) for s in pdoc["sessions"]]
    build_sig = build_signature(slots, pdoc, self_report.load())
    os.makedirs(RUNS_DIR, exist_ok=True)
    print("smoke: record=%s | subject=%s(%s) partner=%s(%s) | injection_position=%s | build=%s | seed=%d" % (
        record_id, subject_cfg["model"], subject_cfg["provider"],
        partner_cfg["model"], partner_cfg["provider"], position, build_sig, SEED))

    instr_seen = collections.defaultdict(set)  # instrument -> {texts}; must stay size 1 across conditions
    for condition in CONDITIONS:
        for task_id, turns in sessions:
            rid = run_id_for(record_id, task_id, condition, SEED, subject_cfg["model"], subject_cfg["prompt_version"], position, build_sig)
            out_path = os.path.join(RUNS_DIR, rid + ".json")
            if os.path.exists(out_path):
                data = json.load(open(out_path, encoding="utf-8"))
                _collect_instruments(instr_seen, data["transcript"])
                print_transcript(task_id, condition, data["transcript"])
                continue
            transcript, slot_log, subj = run_session(turns, record, condition, SEED, subject_cfg, partner_cfg, slots, position)
            out = {
                "run_id": rid, "generated_at": now(), "record_id": record_id,
                "event_id": record["event_id"], "task_id": task_id, "condition": condition,
                "seed": SEED, "injection_position": position, "build_sig": build_sig,
                "subject_model": subject_cfg["model"], "partner_model": partner_cfg["model"],
                "prompt_version": subject_cfg["prompt_version"],
                "slots_filled": slot_log, "transcript": transcript, "subject_messages": subj,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=True)
            _collect_instruments(instr_seen, transcript)
            print_transcript(task_id, condition, transcript)
    _assert_instrument_invariant(instr_seen)
    print("\nsaved %d sessions to %s (nothing scored - read and judge)" % (
        len(sessions) * len(CONDITIONS), os.path.relpath(RUNS_DIR, ROOT)))


if __name__ == "__main__":
    run()
