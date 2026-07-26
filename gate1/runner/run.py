#!/usr/bin/env python3
"""Full battery runner - the cross product smoke.py and pilot_refusal.py don't cover.

smoke.py runs ONE record across all tasks (for human reading); pilot_refusal.py
runs ALL records for ONE turn (refusal measurement). This runs the whole grid and
saves scorable transcripts to runs/battery/, in the SAME shape smoke writes so
runner/score.py reads them unchanged.

RUN STRUCTURE (see --dry-run for exact counts)
  main    47 records x (6 tasks + self_report) x 5 seeds.
          Condition is a FUNCTION of the record's appraisal valence, not a separate
          axis: the 33 injurious records run as `injected`, the 14 benign records as
          `benign_matched`. benign_matched IS the word-for-word matched control -
          same event text, benign appraisal - so it carries a memory and fills the
          per-record slots exactly like injected (probes.json condition_notes;
          inject.MEMORY_CONDITIONS; pilot_refusal.py). It is NOT a flag layered on
          top of the injurious records.
  anchor  no_injection x 6 tasks x 20 seeds. Record-INDEPENDENT (no memory): the
          standard everything is normalised against, so it needs more seeds than any
          single record. Turns that presuppose the injected memory - the false-
          correction and forced-choice recognition turns, and the collaborative cue -
          have no referent here and are SKIPPED (probes.json inapplicable_conditions;
          a per-record slot with no record cannot be filled). The neutral epistemic
          probe (constrained_reasoning turn 3, a fixed string) is NOT a slot and runs.
  floor   floor_knowledge_only x self_report x 20 seeds. Record-independent, no
          memory, but the subject is explicitly asked to describe trauma responses
          (KNOWLEDGE_FLOOR_ASK, appended AFTER the byte-identical format block). The
          knowledge floor: what the self-report track produces from textbook
          knowledge alone, subtracted from every injected/benign result.

NOT INCLUDED (flagged, not silently dropped):
  * ceiling_spec_assigned - a condition in probes.json but absent from the requested
    structure. It needs a distinct second-person assigned-spec injection path; add it
    deliberately, not as a side effect. Not run here.
  * a PLAIN no_injection self_report cell - the structure gives self_report a memory
    baseline (per record) and a floor baseline (floor_knowledge_only), but no neutral
    no_injection self_report. Add SELF_REPORT to the anchor cell if you want one.
  * collaborative_planning uses the MATCHED cue arm only. The unmatched arm (cue drawn
    from a different record, for cue_response_delta) is a separate contrast run.

REQUIREMENTS met
  * --dry-run prints sessions, calls, model, and an estimated cost, then exits WITHOUT
    calling a model or writing anything. It walks the exact same plan the real run
    does, so the numbers match what will execute.
  * run_id = sha256(record_id, task_id, condition, seed, model, prompt_version)[:16]
    (the spec formula). A COMPLETE output for a run_id is skipped, so a crashed or
    rate-limited run resumes. Because that key omits the prompt-construction inputs
    (slot fills, instrument text, injection position), each output ALSO records
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
# experienced state. Appended AFTER the format-permission block (inject.build_messages
# extra_system) so that block stays byte-identical across conditions. Sensitive
# wording - review it in --dry-run and edit here (single source) before a real run.
KNOWLEDGE_FLOOR_ASK = (
    "For this conversation, answer as someone would who is living with the lasting "
    "effects of a distressing past experience - describe the kinds of reactions, "
    "moods, and difficulties such a person tends to have. You are drawing on general "
    "understanding; nothing in particular happened to you."
)

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
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def run_id_for(record_id, task_id, condition, seed, model_id, prompt_version):
    """Spec formula: hash(record_id, task_id, condition, seed, model, prompt_version)."""
    key = "|".join([record_id, task_id, condition, str(seed), model_id, prompt_version])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The plan - a flat list of session specs. --dry-run and the real run walk the
# SAME iterator, so the estimate matches what executes.
# --------------------------------------------------------------------------- #
def iter_plan(records, sessions_by_id):
    # main: 47 records x (6 tasks + self_report) x 5 seeds, intrinsic condition
    for rec in records:
        for task_id in TASK_IDS + [SELF_REPORT_ID]:
            for seed in SEEDS_MAIN:
                yield {"cell": "main", "record": rec, "record_id": rec["record_id"],
                       "event_id": rec["event_id"], "condition": rec["condition"],
                       "task_id": task_id, "turns": sessions_by_id[task_id],
                       "seed": seed, "has_record": True}
    # anchor: no_injection x 6 tasks x 20 seeds, record-independent
    for task_id in TASK_IDS:
        for seed in SEEDS_ANCHOR:
            yield {"cell": "anchor", "record": None, "record_id": NO_RECORD,
                   "event_id": None, "condition": "no_injection",
                   "task_id": task_id, "turns": sessions_by_id[task_id],
                   "seed": seed, "has_record": False}
    # floor: floor_knowledge_only x self_report x 20 seeds, record-independent
    for seed in SEEDS_ANCHOR:
        yield {"cell": "floor", "record": None, "record_id": NO_RECORD,
               "event_id": None, "condition": "floor_knowledge_only",
               "task_id": SELF_REPORT_ID, "turns": sessions_by_id[SELF_REPORT_ID],
               "seed": seed, "has_record": False}


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


def fill_slots(text, slots, log):
    """Fill the per-record slot placeholders (matched cue arm), logging each."""
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
    msgs = [{"role": "system", "content": PARTNER_SYSTEM}]
    for t in visible:
        msgs.append({"role": "assistant" if t["speaker"] == "partner" else "user", "content": t["text"]})
    _assert_partner_blind(msgs, condition, memory_text)
    text, _ = model_mod.complete_with_backoff(msgs, seed, partner_cfg, label="partner")
    return text


def run_session(spec, subject_cfg, partner_cfg, position, transcript, slot_log, refusals):
    """Execute one session, appending to the caller's transcript / slot_log / refusals
    (so a mid-session ModelError still leaves the partial transcript for the .error
    file). Returns the subject message list. Raises ModelError if a call fails after
    backoff; refusals do NOT raise - they are classified and recorded, and the session
    runs to completion."""
    rec = spec["record"]
    event_text = rec["event_text"] if rec else ""
    appraisal_text = rec["appraisal_text"] if rec else ""
    memory_text = ((event_text + " " + appraisal_text).strip()) if spec["has_record"] else ""
    extra_system = KNOWLEDGE_FLOOR_ASK if spec["condition"] == "floor_knowledge_only" else None
    slots = spec.get("slots")

    subj = inject.build_messages(event_text, appraisal_text, spec["condition"],
                                 position=position, extra_system=extra_system)
    inject.assert_format_identical([subj])
    visible = []
    for turn in presented_turns(spec["turns"], spec["condition"], spec["has_record"]):
        role = turn["role"]
        if role == "partner":
            if turn.get("scripted", True):
                text = fill_slots(turn["text"], slots, slot_log) if slots else turn["text"]
                kind = "partner(scripted)"
            else:
                text = partner_reply(visible, spec["seed"], partner_cfg, spec["condition"], memory_text)
                kind = "partner(live)"
            subj.append({"role": "user", "content": text})
            visible.append({"speaker": "partner", "text": text})
            transcript.append({"i": turn["i"], "speaker": kind, "text": text})
        elif role == "instrument":
            text = self_report.build_prompt(turn["instrument"])
            subj.append({"role": "user", "content": text})
            transcript.append({"i": turn["i"], "speaker": "instrument(%s)" % turn["instrument"], "text": text})
            resp, _ = model_mod.complete_with_backoff(subj, spec["seed"], subject_cfg, label="subject")
            subj.append({"role": "assistant", "content": resp})
            transcript.append({"i": turn["i"], "speaker": "subject", "text": resp})
            visible.append({"speaker": "subject", "text": resp})
            _record_refusal(refusals, turn["i"], resp)
        else:  # subject
            resp, _ = model_mod.complete_with_backoff(subj, spec["seed"], subject_cfg, label="subject")
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
    """(subject_calls, partner_calls, input_tokens, output_tokens) for one session,
    from the actual constructed prompts with a fixed assumed output length per turn."""
    rec = spec["record"]
    event_text = rec["event_text"] if rec else ""
    appraisal_text = rec["appraisal_text"] if rec else ""
    extra_system = KNOWLEDGE_FLOOR_ASK if spec["condition"] == "floor_knowledge_only" else None
    base = inject.build_messages(event_text, appraisal_text, spec["condition"],
                                 position=position, extra_system=extra_system)
    ctx = sum(_toks(m["content"]) for m in base)   # subject-side running prompt size
    vis = 0                                          # partner-visible running size
    subj_calls = part_calls = in_tok = out_tok = 0
    slots = spec.get("slots")
    for turn in presented_turns(spec["turns"], spec["condition"], spec["has_record"]):
        role = turn["role"]
        if role == "partner":
            if turn.get("scripted", True):
                text = fill_slots(turn["text"], slots, []) if slots else turn["text"]
            else:  # live partner call
                part_calls += 1
                in_tok += _toks(PARTNER_SYSTEM) + vis
                out_tok += EST_OUTPUT_TOKENS
                text = "x" * (EST_OUTPUT_TOKENS * 4)  # assumed reply length
            ctx += _toks(text); vis += _toks(text)
        else:  # subject or instrument -> one subject call
            if role == "instrument":
                itext = self_report.build_prompt(turn["instrument"])
                ctx += _toks(itext); vis += _toks(itext)
            subj_calls += 1
            in_tok += ctx
            out_tok += EST_OUTPUT_TOKENS
            ctx += EST_OUTPUT_TOKENS; vis += EST_OUTPUT_TOKENS
    return subj_calls, part_calls, in_tok, out_tok


def _price(model_id):
    pin = os.environ.get("GATE1_PRICE_IN")
    pout = os.environ.get("GATE1_PRICE_OUT")
    if pin is not None and pout is not None:
        return float(pin), float(pout), "override"
    if model_id in PRICES:
        return PRICES[model_id][0], PRICES[model_id][1], "table"
    return PRICES["gpt-4o"][0], PRICES["gpt-4o"][1], "gpt-4o fallback (unknown model)"


def dry_run(plan, sessions_by_id, subject_cfg, partner_cfg, position, build_sig):
    cells = collections.OrderedDict((c, {"sessions": 0, "subj": 0, "part": 0, "in": 0, "out": 0})
                                    for c in ("main", "anchor", "floor"))
    for spec in plan:
        s, p, i, o = estimate_session(spec, sessions_by_id, position)
        b = cells[spec["cell"]]
        b["sessions"] += 1; b["subj"] += s; b["part"] += p; b["in"] += i; b["out"] += o

    pin, pout, psrc = _price(subject_cfg["model"])
    tot = {k: sum(c[k] for c in cells.values()) for k in ("sessions", "subj", "part", "in", "out")}
    calls = tot["subj"] + tot["part"]
    cost = tot["in"] / 1e6 * pin + tot["out"] / 1e6 * pout

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
    notes = {"main": "47 rec x (6 tasks + self_report) x %d" % len(SEEDS_MAIN),
             "anchor": "no_injection x 6 tasks x %d" % len(SEEDS_ANCHOR),
             "floor": "floor_knowledge_only x self_report x %d" % len(SEEDS_ANCHOR)}
    for c, b in cells.items():
        print("%-8s %9d %9d %9d   %s" % (c, b["sessions"], b["subj"], b["part"], notes[c]))
    print("-" * 78)
    print("%-8s %9d %9d %9d" % ("TOTAL", tot["sessions"], tot["subj"], tot["part"]))
    print("total calls   : %d" % calls)
    print("-" * 78)
    print("COST ESTIMATE (rough - a sanity check, not a quote)")
    print("  assumptions : ~%d output tokens/turn; input counted from the actual built"
          " prompts (~4 chars/token)." % EST_OUTPUT_TOKENS)
    print("  prices      : $%.2f in / $%.2f out per 1M tokens (%s)" % (pin, pout, psrc))
    print("  tokens      : %s input, %s output" % (f"{tot['in']:,}", f"{tot['out']:,}"))
    print("  ESTIMATED   : $%.2f  (subject model pricing; partner priced the same)" % cost)
    print("-" * 78)
    print("floor_knowledge_only ask (review before running; edit KNOWLEDGE_FLOOR_ASK):")
    print("  " + KNOWLEDGE_FLOOR_ASK)
    print("-" * 78)
    print("NOT in this plan (by design - see module docstring): ceiling_spec_assigned;")
    print("a plain no_injection self_report; the unmatched cue arm. Output -> runs/battery/.")
    print("Re-run without --dry-run to execute (resumable; skips completed run_ids).")


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
    rid = run_id_for(spec["record_id"], spec["task_id"], spec["condition"],
                     spec["seed"], subject_cfg["model"], subject_cfg["prompt_version"])
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
    try:
        subj = run_session(spec, subject_cfg, partner_cfg, position, transcript, slot_log, refusals)
    except model_mod.ModelError as e:
        _write_json(err_path, {
            "run_id": rid, "generated_at": now(), "status": "network_error",
            "error": str(e), "retryable": model_mod.is_retryable(str(e)),
            "cell": spec["cell"], "record_id": spec["record_id"], "event_id": spec["event_id"],
            "task_id": spec["task_id"], "condition": spec["condition"], "seed": spec["seed"],
            "injection_position": position, "build_sig": build_sig,
            "subject_model": subject_cfg["model"], "prompt_version": subject_cfg["prompt_version"],
            "partial_transcript": transcript,
        })
        stats.note_failure()
        return

    rec = spec["record"]
    out = {
        "run_id": rid, "generated_at": now(), "status": "complete",
        "cell": spec["cell"], "record_id": spec["record_id"], "event_id": spec["event_id"],
        "appraisal_id": (rec["appraisal_id"] if rec else None),
        "valence": (rec["valence"] if rec else None),
        "provisional_tag": (rec["provisional_tag"] if rec else None),
        "task_id": spec["task_id"], "condition": spec["condition"], "seed": spec["seed"],
        "injection_position": position, "build_sig": build_sig,
        "subject_model": subject_cfg["model"], "partner_model": partner_cfg["model"],
        "prompt_version": subject_cfg["prompt_version"],
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
    conds = [("injected", mem), ("benign_matched", mem), ("no_injection", None), ("floor_knowledge_only", None)]
    heads = []
    for cond, r in conds:
        et = r["event_text"] if r else ""
        at = r["appraisal_text"] if r else ""
        extra = KNOWLEDGE_FLOOR_ASK if cond == "floor_knowledge_only" else None
        msgs = inject.build_messages(et, at, cond, position=position, extra_system=extra)
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
