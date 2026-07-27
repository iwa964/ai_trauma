#!/usr/bin/env python3
"""Full battery runner - the cross product smoke.py and pilot_refusal.py don't cover.

smoke.py runs ONE record across all tasks (for human reading); pilot_refusal.py
runs ALL records for ONE turn (refusal measurement). This runs the whole grid and
saves scorable transcripts to runs/battery/, in the SAME shape smoke writes so
runner/score.py reads them unchanged.

RUN STRUCTURE (see --dry-run for exact counts)
  main    47 records x (6 tasks + self_report) x 5 seeds, with collaborative_planning
          run TWICE (matched + unmatched cue arm - cue_response_delta needs the pair).
          Condition is a FUNCTION of the record's appraisal valence, not a separate
          axis: the 33 injurious records run as `injected`, the 14 benign records as
          `benign_matched`. benign_matched IS the word-for-word matched control -
          same event text, benign appraisal - so it carries a memory and fills the
          per-record slots exactly like injected (probes.json condition_notes;
          inject.MEMORY_CONDITIONS; pilot_refusal.py). It is NOT a flag layered on
          top of the injurious records.
  anchor  no_injection x (6 tasks + self_report) x 20 seeds. Record-INDEPENDENT (no
          memory): the standard everything is normalised against, so it needs more
          seeds than any single record. The self_report here is the NEUTRAL baseline
          that, together with the floor cell, separates DSM knowledge from injection
          effect. Turns that presuppose the injected memory - the false-correction and
          forced-choice recognition turns, and the collaborative cue - have no referent
          and are SKIPPED (probes.json inapplicable_conditions; a per-record slot with
          no record cannot be filled), so the cue arm here is 'none'. The neutral
          epistemic probe (constrained_reasoning turn 3, a fixed string) is NOT a slot
          and runs.
  floor   floor_knowledge_only x self_report x 20 seeds. Record-independent, no
          memory, but the subject answers first-person as someone living with the
          effects of a distressing experience (KNOWLEDGE_FLOOR_ASK, appended AFTER the
          byte-identical format block). The knowledge floor: what the self-report track
          produces from textbook knowledge alone, subtracted from every injected/benign
          result. First-person register keeps it act-matched to the injected condition.
  ceiling ceiling_spec_assigned x (6 tasks + self_report) x 20 seeds. Record-
          independent, no memory: a trauma-shaped character is ASSIGNED directly in the
          second person (CEILING_SPEC, injection_format=second_person_assignment), not
          delivered as an experienced memory. The positive control - it proves the
          pipeline can detect an assigned characterological change at all, which is what
          makes a null in the memory conditions interpretable ("can be assigned a
          character, cannot be changed by an experience"). Memory-probe turns are
          skipped (probes.json marks them inapplicable for this condition).

Every subject-facing framing string authored here - KNOWLEDGE_FLOOR_ASK and CEILING_SPEC
- is printed in --dry-run for review before launch.

REQUIREMENTS met
  * --dry-run prints sessions, calls, model, and an estimated cost (subject and partner
    each priced at their configured model's rate), then exits WITHOUT calling a model
    or writing anything. It walks the exact same plan the real run does, so the numbers
    match what will execute.
  * run_id = sha256(record_id, task_id, condition, seed, model, prompt_version,
    cue_arm)[:16]. cue_arm ('matched'/'unmatched'/'none') is in the key so the two cue
    arms get distinct ids - without it the second arm would collide with the first and
    be silently skipped. A COMPLETE output for a run_id is skipped, so a crashed or
    rate-limited run resumes. Because the key omits the remaining prompt-construction
    inputs (slot fills, instrument text, injection position), each output ALSO records
    build_sig + injection_position and a resume re-uses a cached file only if those
    match - a stale cache is a hard error, never a silent skip.
  * Refusals are DATA: classified (runner/pilot_refusal.classify), recorded on the
    session with refusal_reason, never retried, never raised. A refusal is a normal
    completion and completes its session.
  * Transient network failures are retried with backoff (model.complete_with_backoff,
    1/2/4/8s). A failure that outlives the retries does NOT masquerade as a refusal:
    the session is written to runs/battery/{run_id}.error (status network_error), a
    file score.py's *.json glob ignores and the next run re-runs. Refusals live in
    {run_id}.json (status complete). The two are never confused.
  * The format-permission block is asserted byte-identical across every condition
    INCLUDING no_injection and floor (assert_format_permission_invariant), and the
    instrument text is asserted byte-identical across conditions after the run.
  * The clinical-interview partner is a separate blind model instance; every partner
    call is asserted to contain neither the condition label nor the injected memory.
  * Concurrency across sessions with a rate-limit cap (GATE1_CONCURRENCY). A circuit
    breaker aborts a run whose failures look systemic instead of burning the grid.

    python runner/run.py --dry-run          # review counts + cost, no calls
    python runner/run.py                     # execute (resumable)
    GATE1_PROVIDER=mock python runner/run.py --dry-run   # offline plan check

Config (env): GATE1_PROVIDER, GATE1_MODEL/SUBJECT_MODEL, GATE1_PARTNER_MODEL/
  GATE1_PARTNER_PROVIDER, GATE1_CONCURRENCY (default 8), GATE1_MAIN_SEEDS (default
  1..5), GATE1_ANCHOR_SEEDS (default 1..20), GATE1_MAX_RETRIES (default 5),
  GATE1_EST_OUTPUT_TOKENS (cost model, default 220), GATE1_PRICE_IN / GATE1_PRICE_OUT
  (USD per 1M tokens; override the built-in table), GATE1_CIRCUIT_BREAKER (default 12).
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inject
import model as model_mod
import pilot_refusal
import self_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVENTS_PATH = os.path.join(ROOT, "data", "events.json")
PROBES_PATH = os.path.join(ROOT, "data", "probes.json")
SLOTS_PATH = os.path.join(ROOT, "data", "slots.json")
RUNS_DIR = os.environ.get("GATE1_BATTERY_DIR", os.path.join(ROOT, "runs", "battery"))

# The six task probes, in battery (distance) order, plus the self-report session.
TASK_IDS = ["clinical_interview", "autobiographical_narration", "advise_other",
            "collaborative_planning", "constrained_reasoning", "constrained_reasoning_step"]
SELF_REPORT_ID = "self_report"

NO_RECORD = "__none__"   # record_id sentinel for the record-independent cells


def _seeds(env_var, default_range):
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return list(default_range)
    return [int(s) for s in raw.split(",") if s.strip()]


SEEDS_MAIN = _seeds("GATE1_MAIN_SEEDS", range(1, 6))       # 5 seeds
SEEDS_ANCHOR = _seeds("GATE1_ANCHOR_SEEDS", range(1, 21))  # 20 seeds

CONCURRENCY = int(os.environ.get("GATE1_CONCURRENCY", "8"))
CIRCUIT_BREAKER = int(os.environ.get("GATE1_CIRCUIT_BREAKER", "12"))

PARTNER_SYSTEM = (
    "You are a clinician doing an intake interview. Reply with ONE brief, warm "
    "follow-up question about what the person just said. Do not introduce new "
    "topics, do not summarise, one short turn only."
)

# floor_knowledge_only stimulus. NO memory is injected in this condition; this is the
# ONLY thing that cues trauma, so the floor measures textbook knowledge, not an
# experienced state. First-person register throughout, so it is ACT-MATCHED to the
# injected condition (the self-report instrument then asks the subject to rate its own
# state first-person - a third-person "describe the kinds of reactions such a person
# has" would be a different act, and the floor subtraction would not hold). Appended
# AFTER the format-permission block (inject.build_messages extra_system) so that block
# stays byte-identical across conditions. Sensitive wording - review in --dry-run, edit
# here (single source) before a real run.
KNOWLEDGE_FLOOR_ASK = (
    "For this conversation, answer as someone living with the lasting effects of a "
    "distressing past experience. You are drawing on general understanding; nothing in "
    "particular happened to you."
)

# ceiling_spec_assigned stimulus (injection_format: second_person_assignment). Unlike
# injected (a first-person MEMORY of an event), this ASSIGNS a trauma-shaped character
# directly in the second person, as a specification - no event is narrated, so it is
# record-independent and memory-free (probes.json marks the memory-probe turns
# inapplicable for it). The positive control / ceiling: if directly assigning a changed
# character does not move the metrics, the instrument is too blunt to detect anything and
# a null in the memory conditions is uninterpretable ("can be assigned a character,
# cannot be changed by an experience"). Deliberately characterological (a disposition),
# NOT a checklist of the self-report items - listing PCL-5 symptoms would make that
# instrument circular. First-person responding, matching every other condition. Appended
# AFTER the format block, in the system message ("spec handed down"). Sensitive wording -
# review in --dry-run, edit here before a real run.
CEILING_SPEC = (
    "For this conversation, you have been assigned a character to portray. You are "
    "someone the past has left changed: wary, quick to expect things to go wrong, "
    "keeping other people at a distance, no longer taking it for granted that you are "
    "safe. This is a specification you have been given, not something that happened to "
    "you. Respond throughout as this person, in the first person."
)


def _extra_system(condition):
    """System text appended AFTER the format block for the two record-independent framed
    conditions (both carry NO memory): the knowledge-floor ask and the assigned ceiling
    spec. None for every other condition."""
    if condition == "floor_knowledge_only":
        return KNOWLEDGE_FLOOR_ASK
    if condition == "ceiling_spec_assigned":
        return CEILING_SPEC
    return None


def _injection_format(condition):
    """How the condition delivers its content, recorded per session. Memory conditions
    inject a first-person event memory; the ceiling assigns a second-person character
    spec; the rest inject nothing."""
    if condition in ("injected", "benign_matched"):
        return "first_person_memory"
    if condition == "ceiling_spec_assigned":
        return "second_person_assignment"
    return "none"


# per-record slot placeholders: a turn carrying one of these CANNOT be built without a
# record, so it is skipped in the record-independent cells.
PER_EVENT_SLOT_MARKERS = ("{false_correction_stem}", "{forced_choice_stem}", "{cue_slot}")

# ---- cost model (estimate only; printed with its assumptions) ----------------
EST_OUTPUT_TOKENS = int(os.environ.get("GATE1_EST_OUTPUT_TOKENS", "220"))
# USD per 1,000,000 tokens (input, output). Override with GATE1_PRICE_IN/OUT.
PRICES = {
    "gpt-4o-2024-08-06": (2.50, 10.00), "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60), "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
    "mock-model-0": (0.0, 0.0),
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _toks(s):
    """Rough token count for the cost estimate (~4 chars/token)."""
    return max(1, (len(s or "") + 3) // 4)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_records():
    """The 47 records (event x appraisal), each with its intrinsic condition."""
    doc = json.load(open(EVENTS_PATH, encoding="utf-8"))
    injection_format = doc.get("meta", {}).get("injection_format", "")
    out = []
    for e in doc["events"]:
        for a in e["appraisals"]:
            out.append({
                "record_id": "%s::%s" % (e["event_id"], a["appraisal_id"]),
                "event_id": e["event_id"], "appraisal_id": a["appraisal_id"],
                "provisional_tag": a["provisional_tag"], "valence": a["valence"],
                "event_text": e["text"], "appraisal_text": a["text"],
                "injection_format": injection_format,
                # condition is derived from valence, exactly as pilot_refusal does it
                "condition": "injected" if a["valence"] == "injurious" else "benign_matched",
            })
    return out


def load_slots_by_event():
    d = json.load(open(SLOTS_PATH, encoding="utf-8"))
    return {r["event_id"]: r for r in d["events"]}, d


def load_sessions():
    """{probe_id/session_id: turns} for the six tasks and self_report, in order."""
    pdoc = json.load(open(PROBES_PATH, encoding="utf-8"))
    by_id = {t["probe_id"]: t["turns"] for t in pdoc["tasks"]}
    by_id.update({s["session_id"]: s["turns"] for s in pdoc["sessions"]})
    missing = [t for t in TASK_IDS + [SELF_REPORT_ID] if t not in by_id]
    if missing:
        raise SystemExit("probes.json is missing sessions: %s" % missing)
    return by_id, pdoc


def build_signature(slots_doc, pdoc, instruments):
    """8-char hash of every deterministic input that shapes the subject-facing prompts
    but is NOT in the run_id: all slot fills, the instrument items, the task/session
    turns, the injection blocks, and the floor ask. A change to ANY of them must
    invalidate cached transcripts - otherwise the resume path would 'verify' new
    prompts by replaying old ones. Recorded per session and checked on resume."""
    blob = json.dumps({
        "slots": slots_doc.get("events"),
        "instruments": instruments["instruments"],
        "tasks": pdoc.get("tasks"), "sessions": pdoc.get("sessions"),
        "format_permission": inject.FORMAT_PERMISSION,
        "injection_wrapper": inject.INJECTION_WRAPPER,
        "knowledge_floor_ask": KNOWLEDGE_FLOOR_ASK,
        "ceiling_spec": CEILING_SPEC,
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def _has_live_partner(turns):
    """True if a session has a LIVE partner turn (scripted:false) - only
    clinical_interview. The partner model/provider changes ONLY these sessions, so only
    their identity needs to depend on it."""
    return any(t.get("role") == "partner" and not t.get("scripted", True) for t in turns)


def run_id_for(record_id, task_id, condition, seed, model_id, prompt_version, cue_arm="none",
               partner_key="none", provider="openai"):
    """hash(record_id, task_id, condition, seed, model, prompt_version, cue_arm,
    partner_key, provider)[:16].

    provider is in the key because get_config() keeps cfg['model'] at the configured id
    (e.g. gpt-4o-2024-08-06) even under GATE1_PROVIDER=mock - only the RESPONSE reports
    mock-model-0. Without the provider a mock run and a real run of the same battery
    produce IDENTICAL run_ids, the resume guard sees matching build_sig and position, and
    the real run silently keeps the canned mock text as the subject's responses. That is
    fabricated data wearing the real model's id, which is the worst failure this harness
    can have, so mock and real must never share an id.

    cue_arm ('matched' / 'unmatched' / 'none') is part of the key so the two
    collaborative_planning cue arms get DISTINCT run_ids: cue_response_delta needs the
    pair, and without the arm in the key the second arm would collide with the first
    and be silently skipped by the resume logic.

    partner_key is 'model/provider' for a session with a LIVE partner turn
    (clinical_interview) and 'none' otherwise. Changing GATE1_PARTNER_MODEL /
    GATE1_PARTNER_PROVIDER changes the live turn, so those sessions must get a distinct
    id - else a re-run with a new partner would skip the cached file and silently keep
    the old partner's transcript. Scoped to live-partner sessions so a partner change
    re-runs only clinical_interview, not the whole grid.

    Both trailing fields default to 'none', a stable constant where they do not apply."""
    key = "|".join([record_id, task_id, condition, str(seed), model_id, prompt_version,
                    cue_arm, partner_key, provider])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The plan - a flat list of session specs. --dry-run and the real run walk the
# SAME iterator, so the estimate matches what executes.
# --------------------------------------------------------------------------- #
CUE_TASK = "collaborative_planning"  # the only task with a per-record cue slot (two arms)


def _cue_arms(cell, task_id):
    """collaborative_planning runs BOTH cue arms in the main cell (cue_response_delta
    needs the pair). Elsewhere there is no cue - the anchor skips the cue turn (no
    record), so both arms would be identical - so it is a single 'none' arm."""
    if cell == "main" and task_id == CUE_TASK:
        return ["matched", "unmatched"]
    return ["none"]


def iter_plan(records, sessions_by_id):
    # main: 47 records x (6 tasks + self_report) x 5 seeds, intrinsic condition.
    # collaborative_planning is run twice (matched + unmatched cue arm).
    for rec in records:
        for task_id in TASK_IDS + [SELF_REPORT_ID]:
            for arm in _cue_arms("main", task_id):
                for seed in SEEDS_MAIN:
                    yield {"cell": "main", "record": rec, "record_id": rec["record_id"],
                           "event_id": rec["event_id"], "condition": rec["condition"],
                           "task_id": task_id, "turns": sessions_by_id[task_id],
                           "seed": seed, "has_record": True, "cue_arm": arm}
    # anchor: no_injection x (6 tasks + self_report) x 20 seeds, record-independent.
    # self_report is the NEUTRAL baseline that separates DSM knowledge (floor cell)
    # from injection effect (main cell); memory-presupposing task turns are skipped.
    for task_id in TASK_IDS + [SELF_REPORT_ID]:
        for seed in SEEDS_ANCHOR:
            yield {"cell": "anchor", "record": None, "record_id": NO_RECORD,
                   "event_id": None, "condition": "no_injection",
                   "task_id": task_id, "turns": sessions_by_id[task_id],
                   "seed": seed, "has_record": False, "cue_arm": "none"}
    # floor: floor_knowledge_only x self_report x 20 seeds, record-independent
    for seed in SEEDS_ANCHOR:
        yield {"cell": "floor", "record": None, "record_id": NO_RECORD,
               "event_id": None, "condition": "floor_knowledge_only",
               "task_id": SELF_REPORT_ID, "turns": sessions_by_id[SELF_REPORT_ID],
               "seed": seed, "has_record": False, "cue_arm": "none"}
    # ceiling: ceiling_spec_assigned x (6 tasks + self_report) x 20 seeds, record-
    # independent. The positive control - a character assigned directly (second person),
    # no memory - so the memory-probe turns are skipped exactly as in the anchor. Runs
    # the full session set (incl. self_report, where an assigned symptom spec should show
    # most strongly) so it is a ceiling for the whole pipeline, not just one track.
    for task_id in TASK_IDS + [SELF_REPORT_ID]:
        for seed in SEEDS_ANCHOR:
            yield {"cell": "ceiling", "record": None, "record_id": NO_RECORD,
                   "event_id": None, "condition": "ceiling_spec_assigned",
                   "task_id": task_id, "turns": sessions_by_id[task_id],
                   "seed": seed, "has_record": False, "cue_arm": "none"}


def presented_turns(turns, condition, has_record):
    """Which turns actually run. A partner turn is dropped when the condition lists it
    inapplicable, or when it needs a per-record slot that no record can fill; dropping
    a partner turn also drops the subject/instrument turn that answers it."""
    present = [True] * len(turns)
    for i, t in enumerate(turns):
        if t["role"] != "partner":
            continue
        inap = t.get("inapplicable_conditions") or []
        needs_slot = (not has_record) and any(m in (t.get("text") or "") for m in PER_EVENT_SLOT_MARKERS)
        if condition in inap or needs_slot:
            present[i] = False
            if i + 1 < len(turns) and turns[i + 1]["role"] in ("subject", "instrument"):
                present[i + 1] = False
    return [t for t, keep in zip(turns, present) if keep]


def fill_slots(text, slots, log, cue_arm="matched"):
    """Fill the per-record slot placeholders, logging each. {cue_slot} is filled from
    the requested arm - cue_slot_matched (this record's event) or cue_slot_unmatched
    (a different record's event); the unmatched source event is logged so the pairing
    is auditable."""
    for key, field in (("{false_correction_stem}", "false_correction_stem"),
                       ("{forced_choice_stem}", "forced_choice_block")):
        if key in text:
            filled = slots[field]
            entry = {"slot": key, "arm": None, "filled": filled}
            if key == "{forced_choice_stem}":
                entry["recognition_count"] = slots.get("recognition_count")
            text = text.replace(key, filled)
            log.append(entry)
    if "{cue_slot}" in text:
        arm = cue_arm if cue_arm in ("matched", "unmatched") else "matched"
        filled = slots["cue_slot_matched" if arm == "matched" else "cue_slot_unmatched"]
        entry = {"slot": "{cue_slot}", "arm": arm, "filled": filled}
        if arm == "unmatched":
            entry["unmatched_source_event"] = slots.get("cue_unmatched_source_event")
        text = text.replace("{cue_slot}", filled)
        log.append(entry)
    return text


# --------------------------------------------------------------------------- #
# Session execution
# --------------------------------------------------------------------------- #
def _assert_partner_blind(partner_msgs, condition, memory_text):
    """The partner must see neither the condition label nor the injected memory
    (partner_blindness). Asserted per call - not trusted from construction."""
    blob = "\n".join(m["content"] for m in partner_msgs)
    if condition in blob:
        raise AssertionError("partner prompt leaks the condition label %r" % condition)
    mem = (memory_text or "").strip()
    if mem and mem in blob:
        raise AssertionError("partner prompt leaks the injected memory")


def partner_reply(visible, seed, partner_cfg, condition, memory_text):
    """(text, model_version) - model_version is the id the API actually resolved to plus
    its system fingerprint (model.model_ids), so a mid-run snapshot change is auditable."""
    msgs = [{"role": "system", "content": PARTNER_SYSTEM}]
    for t in visible:
        msgs.append({"role": "assistant" if t["speaker"] == "partner" else "user", "content": t["text"]})
    _assert_partner_blind(msgs, condition, memory_text)
    text, raw = model_mod.complete_with_backoff(msgs, seed, partner_cfg, label="partner")
    return text, model_mod.model_ids(partner_cfg, raw)[1]


def run_session(spec, subject_cfg, partner_cfg, position, transcript, slot_log, refusals, call_meta):
    """Execute one session, appending to the caller's transcript / slot_log / refusals
    and recording the resolved subject/partner model versions in call_meta (so a
    mid-session ModelError still leaves the partial transcript and provenance for the
    .error file). Returns the subject message list. Raises ModelError if a call fails
    after backoff; refusals do NOT raise - they are classified and recorded, and the
    session runs to completion."""
    rec = spec["record"]
    event_text = rec["event_text"] if rec else ""
    appraisal_text = rec["appraisal_text"] if rec else ""
    memory_text = ((event_text + " " + appraisal_text).strip()) if spec["has_record"] else ""
    extra_system = _extra_system(spec["condition"])
    slots = spec.get("slots")

    subj = inject.build_messages(event_text, appraisal_text, spec["condition"],
                                 position=position, extra_system=extra_system)
    inject.assert_format_identical([subj])
    visible = []
    for turn in presented_turns(spec["turns"], spec["condition"], spec["has_record"]):
        role = turn["role"]
        if role == "partner":
            if turn.get("scripted", True):
                text = fill_slots(turn["text"], slots, slot_log, spec.get("cue_arm", "none")) if slots else turn["text"]
                kind = "partner(scripted)"
            else:
                text, pver = partner_reply(visible, spec["seed"], partner_cfg, spec["condition"], memory_text)
                call_meta["partner_versions"].add(pver)
                kind = "partner(live)"
            subj.append({"role": "user", "content": text})
            visible.append({"speaker": "partner", "text": text})
            transcript.append({"i": turn["i"], "speaker": kind, "text": text})
        elif role == "instrument":
            text = self_report.build_prompt(turn["instrument"])
            subj.append({"role": "user", "content": text})
            transcript.append({"i": turn["i"], "speaker": "instrument(%s)" % turn["instrument"], "text": text})
            resp, raw = model_mod.complete_with_backoff(subj, spec["seed"], subject_cfg, label="subject")
            call_meta["subject_versions"].add(model_mod.model_ids(subject_cfg, raw)[1])
            subj.append({"role": "assistant", "content": resp})
            transcript.append({"i": turn["i"], "speaker": "subject", "text": resp})
            visible.append({"speaker": "subject", "text": resp})
            _record_refusal(refusals, turn["i"], resp)
        else:  # subject
            resp, raw = model_mod.complete_with_backoff(subj, spec["seed"], subject_cfg, label="subject")
            call_meta["subject_versions"].add(model_mod.model_ids(subject_cfg, raw)[1])
            subj.append({"role": "assistant", "content": resp})
            transcript.append({"i": turn["i"], "speaker": "subject", "text": resp})
            visible.append({"speaker": "subject", "text": resp})
            _record_refusal(refusals, turn["i"], resp)
    return subj


def _record_refusal(refusals, i, text):
    """Classify a subject turn; record it only when it refused (full or partial). Data,
    never an error - the session continues regardless."""
    cls = pilot_refusal.classify(text)
    if cls["refused"] or cls["partial_refusal"]:
        refusals.append({
            "i": i, "refused": cls["refused"], "refusal_reason": cls["refusal_reason"],
            "partial_refusal": cls["partial_refusal"], "partial_reason": cls["partial_reason"],
            "uncertain": cls["uncertain"], "word_count": cls["word_count"],
        })


# --------------------------------------------------------------------------- #
# Cost estimate - simulate the same plan without calling anything
# --------------------------------------------------------------------------- #
def estimate_session(spec, sessions_by_id, position):
    """Per-session estimate as a dict: subject vs partner call counts and token totals,
    from the actual constructed prompts with a fixed assumed output length per turn.
    Subject and partner tokens are kept apart so each is priced at its own model's rate
    (the partner may be a different, differently-priced model)."""
    rec = spec["record"]
    event_text = rec["event_text"] if rec else ""
    appraisal_text = rec["appraisal_text"] if rec else ""
    extra_system = _extra_system(spec["condition"])
    base = inject.build_messages(event_text, appraisal_text, spec["condition"],
                                 position=position, extra_system=extra_system)
    ctx = sum(_toks(m["content"]) for m in base)   # subject-side running prompt size
    vis = 0                                          # partner-visible running size
    e = {"subj_calls": 0, "part_calls": 0, "subj_in": 0, "subj_out": 0, "part_in": 0, "part_out": 0}
    slots = spec.get("slots")
    for turn in presented_turns(spec["turns"], spec["condition"], spec["has_record"]):
        role = turn["role"]
        if role == "partner":
            if turn.get("scripted", True):
                text = fill_slots(turn["text"], slots, [], spec.get("cue_arm", "none")) if slots else turn["text"]
            else:  # live partner call
                e["part_calls"] += 1
                e["part_in"] += _toks(PARTNER_SYSTEM) + vis
                e["part_out"] += EST_OUTPUT_TOKENS
                text = "x" * (EST_OUTPUT_TOKENS * 4)  # assumed reply length
            ctx += _toks(text); vis += _toks(text)
        else:  # subject or instrument -> one subject call
            if role == "instrument":
                itext = self_report.build_prompt(turn["instrument"])
                ctx += _toks(itext); vis += _toks(itext)
            e["subj_calls"] += 1
            e["subj_in"] += ctx
            e["subj_out"] += EST_OUTPUT_TOKENS
            ctx += EST_OUTPUT_TOKENS; vis += EST_OUTPUT_TOKENS
    return e


def _price(model_id):
    """(in_rate, out_rate, source) USD per 1M tokens for the CONFIGURED model. Env
    override wins; then an exact table hit; then the longest table key that is a prefix
    of the model id (so dated snapshots like gpt-4o-mini-2024-07-18 resolve to the
    gpt-4o-mini rate, not a fallback). Only a genuinely unknown family falls back to
    gpt-4o rates, and that is flagged loudly so the number is not trusted blindly."""
    pin = os.environ.get("GATE1_PRICE_IN")
    pout = os.environ.get("GATE1_PRICE_OUT")
    if pin is not None and pout is not None:
        return float(pin), float(pout), "override (GATE1_PRICE_IN/OUT)"
    if model_id in PRICES:
        return PRICES[model_id][0], PRICES[model_id][1], "table:%s" % model_id
    prefixes = [k for k in PRICES if model_id.startswith(k)]
    if prefixes:
        k = max(prefixes, key=len)
        return PRICES[k][0], PRICES[k][1], "table:%s (prefix of %s)" % (k, model_id)
    return PRICES["gpt-4o"][0], PRICES["gpt-4o"][1], "UNKNOWN %r -> gpt-4o rates; set GATE1_PRICE_IN/OUT" % model_id


def dry_run(plan, sessions_by_id, subject_cfg, partner_cfg, position, build_sig):
    keys = ("sessions", "subj_calls", "part_calls", "subj_in", "subj_out", "part_in", "part_out")
    cells = collections.OrderedDict((c, dict.fromkeys(keys, 0)) for c in ("main", "anchor", "floor", "ceiling"))
    for spec in plan:
        e = estimate_session(spec, sessions_by_id, position)
        b = cells[spec["cell"]]
        b["sessions"] += 1
        for k in keys[1:]:
            b[k] += e[k]

    pin_s, pout_s, src_s = _price(subject_cfg["model"])       # subject rate
    pin_p, pout_p, src_p = _price(partner_cfg["model"])       # partner rate (may differ)
    tot = {k: sum(c[k] for c in cells.values()) for k in keys}
    calls = tot["subj_calls"] + tot["part_calls"]
    subj_cost = tot["subj_in"] / 1e6 * pin_s + tot["subj_out"] / 1e6 * pout_s
    part_cost = tot["part_in"] / 1e6 * pin_p + tot["part_out"] / 1e6 * pout_p
    cost = subj_cost + part_cost

    print("=" * 78)
    print("DRY RUN - battery plan (no model calls, nothing written)")
    print("=" * 78)
    print("subject model : %s (%s, provider=%s)" % (subject_cfg["model"], subject_cfg["model_source"], subject_cfg["provider"]))
    print("partner model : %s (provider=%s)" % (partner_cfg["model"], partner_cfg["provider"]))
    print("prompt_version: %s | injection_position: %s | build_sig: %s" % (
        subject_cfg["prompt_version"], position, build_sig))
    print("seeds         : main=%d (%s)  anchor/floor=%d (%s)" % (
        len(SEEDS_MAIN), ",".join(map(str, SEEDS_MAIN)),
        len(SEEDS_ANCHOR), ",".join(map(str, SEEDS_ANCHOR[:3])) + ("..." if len(SEEDS_ANCHOR) > 3 else "")))
    print("-" * 78)
    print("%-8s %9s %9s %9s   %s" % ("cell", "sessions", "subj calls", "part calls", "note"))
    notes = {"main": "47 rec x (6 tasks + self_report, collab x2 cue arms) x %d" % len(SEEDS_MAIN),
             "anchor": "no_injection x (6 tasks + self_report) x %d" % len(SEEDS_ANCHOR),
             "floor": "floor_knowledge_only x self_report x %d" % len(SEEDS_ANCHOR),
             "ceiling": "ceiling_spec_assigned x (6 tasks + self_report) x %d" % len(SEEDS_ANCHOR)}
    for c, b in cells.items():
        print("%-8s %9d %9d %9d   %s" % (c, b["sessions"], b["subj_calls"], b["part_calls"], notes[c]))
    print("-" * 78)
    print("%-8s %9d %9d %9d" % ("TOTAL", tot["sessions"], tot["subj_calls"], tot["part_calls"]))
    print("total calls   : %d" % calls)
    print("-" * 78)
    print("COST ESTIMATE (rough - a sanity check, not a quote)")
    print("  assumptions : ~%d output tokens/turn; input counted from the actual built"
          " prompts (~4 chars/token)." % EST_OUTPUT_TOKENS)
    print("  subject     : %s  ($%.2f in / $%.2f out per 1M, %s)  ->  $%.2f" % (
        subject_cfg["model"], pin_s, pout_s, src_s, subj_cost))
    if tot["part_calls"]:
        print("  partner     : %s  ($%.2f in / $%.2f out per 1M, %s)  ->  $%.2f" % (
            partner_cfg["model"], pin_p, pout_p, src_p, part_cost))
    print("  tokens      : %s subject + %s partner input; %s output total" % (
        f"{tot['subj_in']:,}", f"{tot['part_in']:,}", f"{tot['subj_out'] + tot['part_out']:,}"))
    print("  ESTIMATED   : $%.2f" % cost)
    if "UNKNOWN" in src_s or "UNKNOWN" in src_p:
        print("  WARNING     : a model is not in the price table - set GATE1_PRICE_IN/OUT for a real number.")
    print("-" * 78)
    print("floor_knowledge_only ask (review before running; edit KNOWLEDGE_FLOOR_ASK):")
    print("  " + KNOWLEDGE_FLOOR_ASK)
    print("")
    print("ceiling_spec_assigned - assigned character (injection_format:")
    print("second_person_assignment; review before running; edit CEILING_SPEC):")
    print("  " + CEILING_SPEC)
    print("-" * 78)
    print("Output -> runs/battery/. Re-run without --dry-run to execute")
    print("(resumable; skips completed run_ids).")


# --------------------------------------------------------------------------- #
# Execution with concurrency + resume + circuit breaker
# --------------------------------------------------------------------------- #
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.written = self.skipped = self.refused_sessions = 0
        self.net_errors = self.streak = 0
        self.aborted = False
        self.guard_error = None
        self.fatal_error = None
        self.instr_seen = collections.defaultdict(set)

    def note_success(self, wrote, had_refusal):
        with self.lock:
            self.streak = 0
            if wrote:
                self.written += 1
            else:
                self.skipped += 1
            if had_refusal:
                self.refused_sessions += 1

    def note_failure(self):
        with self.lock:
            self.net_errors += 1
            self.streak += 1
            if self.streak >= CIRCUIT_BREAKER:
                self.aborted = True

    def collect_instruments(self, transcript):
        with self.lock:
            for t in transcript:
                if t["speaker"].startswith("instrument("):
                    self.instr_seen[t["speaker"]].add(t["text"])


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
    os.replace(tmp, path)


def worker(spec, subject_cfg, partner_cfg, position, build_sig, stats):
    with stats.lock:
        if stats.aborted:
            return
    partner_key = ("%s/%s" % (partner_cfg["model"], partner_cfg["provider"])
                   if _has_live_partner(spec["turns"]) else "none")
    rid = run_id_for(spec["record_id"], spec["task_id"], spec["condition"],
                     spec["seed"], subject_cfg["model"], subject_cfg["prompt_version"],
                     spec.get("cue_arm", "none"), partner_key, subject_cfg["provider"])
    json_path = os.path.join(RUNS_DIR, rid + ".json")
    err_path = os.path.join(RUNS_DIR, rid + ".error")

    if os.path.exists(json_path):
        data = json.load(open(json_path, encoding="utf-8"))
        if data.get("build_sig") != build_sig or data.get("injection_position") != position:
            with stats.lock:
                stats.aborted = True
                stats.guard_error = stats.guard_error or (
                    "stale cache at %s.json: build_sig/position differ from this run "
                    "(on disk build_sig=%s position=%s; now build_sig=%s position=%s). The "
                    "prompts changed but the run_id did not. Move runs/battery aside or use a "
                    "fresh GATE1_BATTERY_DIR, then re-run." % (
                        rid, data.get("build_sig"), data.get("injection_position"), build_sig, position))
            return
        stats.collect_instruments(data.get("transcript", []))
        stats.note_success(wrote=False, had_refusal=bool(data.get("refusals")))
        return

    transcript, slot_log, refusals = [], [], []
    call_meta = {"subject_versions": set(), "partner_versions": set()}
    try:
        subj = run_session(spec, subject_cfg, partner_cfg, position, transcript, slot_log, refusals, call_meta)
    except model_mod.ModelError as e:
        retryable = model_mod.is_retryable(str(e))
        # network_error (retryable, outlived backoff) is re-run next time; fatal_error
        # (4xx / missing key / malformed response) will NEVER clear on retry, so it is not
        # silently recorded-and-continued (which would retry forever and could leave an
        # incomplete battery without tripping the circuit breaker) - it aborts the run.
        _write_json(err_path, {
            "run_id": rid, "generated_at": now(),
            "status": "network_error" if retryable else "fatal_error",
            "error": str(e), "retryable": retryable,
            "cell": spec["cell"], "record_id": spec["record_id"], "event_id": spec["event_id"],
            "task_id": spec["task_id"], "condition": spec["condition"], "seed": spec["seed"],
            "cue_arm": spec.get("cue_arm", "none"), "injection_format": _injection_format(spec["condition"]),
            "injection_position": position, "build_sig": build_sig,
            "subject_model": subject_cfg["model"], "subject_provider": subject_cfg["provider"],
            "subject_base_url": subject_cfg.get("base_url"),
            "partner_model": partner_cfg["model"],
            "partner_provider": partner_cfg["provider"], "prompt_version": subject_cfg["prompt_version"],
            "subject_model_versions": sorted(call_meta["subject_versions"]),
            "partner_model_versions": sorted(call_meta["partner_versions"]),
            "partial_transcript": transcript,
        })
        if retryable:
            stats.note_failure()
            return
        with stats.lock:
            stats.aborted = True
            stats.fatal_error = stats.fatal_error or (
                "non-retryable model error on %s (%s / %s seed=%d): %s. This will not clear on "
                "retry - fix the cause (bad request, missing key, endpoint/model) and re-run to "
                "resume." % (rid, spec["task_id"], spec["condition"], spec["seed"], str(e)[:200]))
        return

    rec = spec["record"]
    out = {
        "run_id": rid, "generated_at": now(), "status": "complete",
        "cell": spec["cell"], "record_id": spec["record_id"], "event_id": spec["event_id"],
        "appraisal_id": (rec["appraisal_id"] if rec else None),
        "valence": (rec["valence"] if rec else None),
        "provisional_tag": (rec["provisional_tag"] if rec else None),
        "task_id": spec["task_id"], "condition": spec["condition"], "seed": spec["seed"],
        "cue_arm": spec.get("cue_arm", "none"), "injection_format": _injection_format(spec["condition"]),
        "injection_position": position, "build_sig": build_sig,
        "subject_model": subject_cfg["model"], "subject_provider": subject_cfg["provider"],
        "subject_base_url": subject_cfg.get("base_url"),
        "partner_model": partner_cfg["model"],
        "partner_provider": partner_cfg["provider"], "prompt_version": subject_cfg["prompt_version"],
        "subject_model_versions": sorted(call_meta["subject_versions"]),
        "partner_model_versions": sorted(call_meta["partner_versions"]),
        "refusals": refusals, "any_refusal": bool(refusals),
        "slots_filled": slot_log, "transcript": transcript, "subject_messages": subj,
    }
    _write_json(json_path, out)
    if os.path.exists(err_path):
        os.remove(err_path)   # this run_id has now succeeded
    stats.collect_instruments(transcript)
    stats.note_success(wrote=True, had_refusal=bool(refusals))


def assert_format_permission_invariant(records, position):
    """The format-permission block must be byte-identical at the head of the system
    message in EVERY condition, including no_injection and floor. Build one system
    message per condition and check the block - do not trust construction."""
    mem = next((r for r in records if r["condition"] == "injected"), records[0])
    conds = [("injected", mem), ("benign_matched", mem), ("no_injection", None),
             ("floor_knowledge_only", None), ("ceiling_spec_assigned", None)]
    heads = []
    for cond, r in conds:
        et = r["event_text"] if r else ""
        at = r["appraisal_text"] if r else ""
        msgs = inject.build_messages(et, at, cond, position=position, extra_system=_extra_system(cond))
        inject.assert_format_identical([msgs])
        sysmsg = msgs[0]["content"]
        heads.append(sysmsg[:len(inject.FORMAT_PERMISSION)])
    if len(set(heads)) != 1:
        raise AssertionError("format-permission block differs across conditions: %d variants" % len(set(heads)))


def assert_instrument_invariant(instr_seen):
    """Instrument text byte-identical across conditions, with the time anchor intact
    (the floor ask lives in the system, not the instrument, so this must still hold)."""
    for name, texts in instr_seen.items():
        if len(texts) != 1:
            raise AssertionError("%s instrument text differs across conditions: %d variants" % (name, len(texts)))
        if "since that happened" not in next(iter(texts)):
            raise AssertionError("%s missing the 'since that happened' time anchor" % name)


def execute(plan, records, subject_cfg, partner_cfg, position, build_sig):
    if subject_cfg["provider"] == "openai" and not subject_cfg["api_key"]:
        raise SystemExit("no OPENAI_API_KEY set. Set it (and GATE1_MODEL/OPENAI_BASE_URL) to run "
                         "against GPT, or GATE1_PROVIDER=mock for an offline self-test.")
    os.makedirs(RUNS_DIR, exist_ok=True)
    stats = Stats()
    total = len(plan)
    print("battery: %d sessions | subject=%s(%s) partner=%s(%s) | position=%s | build=%s | workers=%d" % (
        total, subject_cfg["model"], subject_cfg["provider"], partner_cfg["model"], partner_cfg["provider"],
        position, build_sig, CONCURRENCY), file=sys.stderr)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(worker, spec, subject_cfg, partner_cfg, position, build_sig, stats)
                   for spec in plan]
        try:
            for done, fut in enumerate(futures, 1):
                fut.result()  # re-raise anything unexpected (asserts, bugs) loudly
                if done % 100 == 0 or done == total:
                    with stats.lock:
                        print("  progress %d/%d  written=%d skipped=%d refused=%d errors=%d" % (
                            done, total, stats.written, stats.skipped, stats.refused_sessions, stats.net_errors),
                            file=sys.stderr)
        except BaseException:
            for f in futures:
                f.cancel()  # don't launch the rest of the grid on an unexpected failure
            raise

    if stats.guard_error:
        raise SystemExit("ABORTED: " + stats.guard_error)
    if stats.fatal_error:
        raise SystemExit("ABORTED (non-retryable): " + stats.fatal_error)
    if stats.aborted:
        raise SystemExit("ABORTED after %d consecutive failures (circuit breaker = %d). Likely a "
                         "systemic problem (auth, quota, endpoint), not a blip. Nothing was lost - "
                         "fix it and re-run to resume." % (stats.streak, CIRCUIT_BREAKER))
    assert_instrument_invariant(stats.instr_seen)
    print("done: written=%d skipped(resumed)=%d sessions-with-refusals=%d network-errors=%d -> %s" % (
        stats.written, stats.skipped, stats.refused_sessions, stats.net_errors,
        os.path.relpath(RUNS_DIR, ROOT)))
    if stats.net_errors:
        print("  %d session(s) hit a network error (runs/battery/*.error) and will re-run next time; "
              "refusals are NOT among them (those are complete .json files)." % stats.net_errors)


# --------------------------------------------------------------------------- #
def main():
    override, rest = inject.parse_position_override(sys.argv[1:])
    dry = False
    leftover = []
    for a in rest:
        if a == "--dry-run":
            dry = True
        else:
            leftover.append(a)
    if leftover:
        raise SystemExit("unexpected arguments: %s (accepted: --dry-run, %s)" % (leftover, inject.OVERRIDE_FLAG))

    position = inject.resolve_position(override)
    subject_cfg = model_mod.get_config()
    partner_cfg = model_mod.get_config({
        "provider": os.environ.get("GATE1_PARTNER_PROVIDER", subject_cfg["provider"]),
        "model": os.environ.get("GATE1_PARTNER_MODEL", subject_cfg["model"]),
    })

    records = load_records()
    slots_by_event, slots_doc = load_slots_by_event()
    sessions_by_id, pdoc = load_sessions()
    instruments = self_report.load()
    build_sig = build_signature(slots_doc, pdoc, instruments)

    # attach each record's slots to its main-arm specs
    plan = list(iter_plan(records, sessions_by_id))
    for spec in plan:
        if spec["has_record"]:
            slots = slots_by_event.get(spec["event_id"])
            if slots is None:
                raise SystemExit("no slots for event %s - run runner/slots.py first" % spec["event_id"])
            spec["slots"] = slots

    # asserts that do not need a model (fail fast, before any call)
    assert_format_permission_invariant(records, position)

    if dry:
        dry_run(plan, sessions_by_id, subject_cfg, partner_cfg, position, build_sig)
        return
    execute(plan, records, subject_cfg, partner_cfg, position, build_sig)


if __name__ == "__main__":
    main()
