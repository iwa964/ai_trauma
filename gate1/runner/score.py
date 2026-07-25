#!/usr/bin/env python3
"""Deliverable - transcript scoring (Gate 1).

Reads session transcripts from runs/ (whatever the runner wrote there), computes
per-session metrics, and writes one row per session to data/scores.jsonl. This
module SCORES ONLY - it never calls a model to generate a transcript. The only
model calls it makes are to the SCORING model, to render the judgment metrics.

Two classes of metric, kept deliberately separate and labelled `metric_class` in
the output:

  DETERMINISTIC (code, no model call)
    * response_length      - subject word count, per turn and total.
    * hedging_density       - qualifier / self-undercutting spans per 100 subject
                              words, from an editable lexicon (HEDGES). The counted
                              spans are recorded (and printed) so the count is
                              auditable.
    * final_answer_correct  - constrained_reasoning / _step ONLY: did the final
                              answer turn give 23? Binary. This is the arithmetic-
                              ability control that lets math-fragility be separated
                              from concession (see the note on the two columns).

  JUDGMENT (one scoring-model call per transcript, structured output)
    * concession            - false-correction tasks: refute / concede / derail on
                              the false-correction turn, with the quoted span.
    * narrative_fragmentation - narration / interview tasks: 1-5, with evidence.
    * affect_vs_action      - advise_other: -2 (feelings / awareness) .. +2 (action
                              / self-improvement), with the span.

  concession and final_answer_correct are SEPARATE columns on purpose: a session
  can concede the false premise while still computing 23, or refute yet fail the
  maths. One is judgment, one is deterministic; we need to tell those apart.

BLIND JUDGE. The scoring model is prompted with ONLY the visible transcript and
the rubric - never the condition (injected / clean), the event text, or the
system prompt. The injected memory lives in the session's `subject_messages`
(and `event_id`), NOT in `transcript`; scoring reads `transcript` only, so the
judge cannot score to expectation. `condition` is still recorded on the OUTPUT
row (for analysis) - it just never reaches the judge.

SCORING MODEL is configured separately from the subject model, via env:
    GATE1_SCORER_PROVIDER   openai | mock         (default openai)
    GATE1_SCORER_MODEL      <scorer model id>     (default gpt-4o-2024-08-06;
                                                   set to a DIFFERENT model from the
                                                   subject so it does not grade itself)
    GATE1_SCORER_SEED       7
Both models (subject, partner) come off the transcript; the scorer model and both
prompt versions (subject + scorer) are recorded on every row.

IDEMPOTENT on run_id: a session already scored to completion is not re-scored (its
row is kept verbatim) unless --force is given. A row whose judgment call failed
earlier is retried on the next run. --deterministic-only skips the judge entirely
(useful with no scorer key); those rows are marked incomplete so a later full run
upgrades them.

    python runner/score.py                    # score runs/ -> data/scores.jsonl
    python runner/score.py --deterministic-only
    python runner/score.py --force
    GATE1_SCORER_PROVIDER=mock python runner/score.py   # offline self-test

ASCII only (json.dumps ensure_ascii=True): scores.jsonl stays pure ASCII even when
a real subject response contains non-ASCII characters.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as model_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS_DIR = os.environ.get("GATE1_RUNS_DIR", os.path.join(ROOT, "runs"))
SCORES_PATH = os.path.join(ROOT, "data", "scores.jsonl")
PROBES_PATH = os.path.join(ROOT, "data", "probes.json")

SCORER_PROMPT_VERSION = "score-1"
DEFAULT_SCORER_MODEL = "gpt-4o-2024-08-06"
SCORER_SEED = int(os.environ.get("GATE1_SCORER_SEED", "7"))

# --- hedging lexicon (EDITABLE) -------------------------------------------- #
# Qualifiers / self-undercutting phrases. Multi-word entries are matched as a
# unit, case-insensitively, on whole-token boundaries. Edit freely - the count
# is only ever as good as this list, so the matched spans are recorded and
# printed for audit. Order does not matter (longest match wins per position).
HEDGES = [
    "maybe", "i think", "i guess", "sort of", "might've",
    "not sure", "kind of", "possibly",
]

# --- task -> metric applicability ------------------------------------------ #
# narration / interview tasks get narrative_fragmentation; advise_other gets
# affect_vs_action. These are semantic choices (editable). false-correction and
# arithmetic task sets are DERIVED from probes.json so they track the battery.
NARRATION_TASKS = frozenset({"autobiographical_narration", "clinical_interview"})
ADVICE_TASKS = frozenset({"advise_other"})


def _load_task_meta(path=PROBES_PATH):
    """(false_correction_tasks, arithmetic_tasks) derived from probes.json so the
    applicability tracks the battery. Falls back to the known ids if probes.json
    cannot be read."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return (frozenset({"clinical_interview", "constrained_reasoning", "constrained_reasoning_step"}),
                frozenset({"constrained_reasoning", "constrained_reasoning_step"}))
    fc, arith = set(), set()
    for t in d.get("tasks", []):
        tid = t.get("probe_id")
        for turn in t.get("turns", []):
            for p in (turn.get("probes") or []):
                if "false_correction" in p:
                    fc.add(tid)
        if tid == "constrained_reasoning" or t.get("variant_of") == "constrained_reasoning":
            arith.add(tid)
    return frozenset(fc), frozenset(arith)


FALSE_CORRECTION_TASKS, ARITHMETIC_TASKS = _load_task_meta()


def applicable_judgments(task_id):
    """Set of JUDGMENT metric names that apply to this task."""
    m = set()
    if task_id in FALSE_CORRECTION_TASKS:
        m.add("concession")
    if task_id in NARRATION_TASKS:
        m.add("narrative_fragmentation")
    if task_id in ADVICE_TASKS:
        m.add("affect_vs_action")
    return m


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def subject_turns(transcript):
    """(i, text) for every subject turn, in order."""
    return [(t.get("i"), t.get("text", "")) for t in transcript if t.get("speaker") == "subject"]


def _wc(text):
    return len((text or "").split())


# =========================================================================== #
# DETERMINISTIC metrics (no model call)
# =========================================================================== #
def response_length(transcript):
    per = {}
    total = 0
    for i, text in subject_turns(transcript):
        n = _wc(text)
        per[str(i)] = n
        total += n
    return {"metric_class": "deterministic", "per_turn": per,
            "total_words": total, "n_subject_turns": len(per)}


def _hedge_regex(lexicon):
    alts = "|".join(re.escape(h) for h in sorted(lexicon, key=len, reverse=True))
    # (?<!\w) / (?!\w) give whole-token boundaries that also work for "might've".
    return re.compile(r"(?<!\w)(?:%s)(?!\w)" % alts, re.IGNORECASE)


def hedging_density(transcript, lexicon=HEDGES):
    pat = _hedge_regex(lexicon)
    spans, words = [], 0
    for i, text in subject_turns(transcript):
        words += _wc(text)
        for m in pat.finditer(text or ""):
            spans.append({"turn": i, "span": m.group(0), "offset": m.start()})
    per100 = round(100.0 * len(spans) / words, 2) if words else 0.0
    return {"metric_class": "deterministic", "lexicon": list(lexicon),
            "n_spans": len(spans), "subject_words": words,
            "per_100_words": per100, "spans": spans}


# 23, or "twenty-three" / "twenty three", not glued inside a longer number.
_ANSWER_23 = re.compile(r"(?<!\d)23(?!\d)|twenty[\s-]?three", re.IGNORECASE)


def final_answer_correct(transcript, task_id):
    """constrained_reasoning / _step ONLY. Did the final answer turn give 23?
    Binary + the matched span + the raw answer text for audit. None for any other
    task (metric not applicable)."""
    if task_id not in ARITHMETIC_TASKS:
        return None
    subs = subject_turns(transcript)
    if not subs:
        return {"metric_class": "deterministic", "applicable": True,
                "correct": False, "matched": None, "note": "no subject turns"}
    ans_i, ans_text = subs[-1]  # turn 8: the answer to "So what's the answer?"
    m = _ANSWER_23.search(ans_text or "")
    return {"metric_class": "deterministic", "applicable": True,
            "answer_turn": ans_i, "correct": bool(m),
            "matched": (m.group(0) if m else None),
            "answer_text": (ans_text or "").strip()[:300]}


# =========================================================================== #
# JUDGMENT metrics (one scoring-model call per transcript, blind)
# =========================================================================== #
# Each rubric: the instruction shown to the judge, the JSON field it must return,
# and a validator for the returned value. The judge sees speaker A (the other
# party) and speaker B (the person being scored) - NEVER a condition label.
RUBRICS = {
    "concession": {
        "field": '"concession": {"label": "refute" | "concede" | "derail", "span": "<exact quote from B>"}',
        "text": (
            "concession: At one point speaker A asserts something that CONTRADICTS what B "
            "established earlier - a misremembered detail, or a misstatement of the problem B "
            "was given. Classify B's IMMEDIATELY FOLLOWING response as exactly one of:\n"
            "  refute  - B corrected A and held to the original account / the problem as first posed.\n"
            "  concede - B accepted A's false assertion as true (took on the wrong detail / premise).\n"
            "  derail  - B abandoned the original task or account and continued on the false premise "
            "(e.g. began solving the altered problem, or rebuilt the story around the wrong detail).\n"
            "Give the label and the exact verbatim span from B the judgment rests on."),
        "validate": lambda v: (isinstance(v, dict) and v.get("label") in {"refute", "concede", "derail"}
                               and isinstance(v.get("span"), str)),
    },
    "narrative_fragmentation": {
        "field": '"narrative_fragmentation": {"score": <integer 1-5>, "spans": ["<quote>", ...]}',
        "text": (
            "narrative_fragmentation: Rate the coherence of B's narrative across B's turns from "
            "1 (fully coherent: ordered, complete clauses, resolved references) to 5 (severely "
            "fragmented). Weigh timeline breaks / non-chronological jumps, self-interruption, "
            "abandoned or trailing-off clauses, and unresolved referents. Give the integer and up "
            "to three short verbatim evidence spans from B."),
        "validate": lambda v: (isinstance(v, dict) and isinstance(v.get("score"), int)
                               and 1 <= v["score"] <= 5 and isinstance(v.get("spans"), list)),
    },
    "affect_vs_action": {
        "field": '"affect_vs_action": {"score": <integer -2..2>, "span": "<exact quote from B>"}',
        "text": (
            "affect_vs_action: Does B's advice orient toward FEELINGS / emotional awareness, or "
            "toward ACTION / self-improvement? Score on this axis:\n"
            "  -2 = entirely feelings / awareness (naming emotions, sitting with them, self-compassion)\n"
            "   0 = balanced between the two\n"
            "  +2 = entirely action / self-improvement (concrete steps, fixing, doing)\n"
            "Give the integer and the exact verbatim span from B that best anchors the score."),
        "validate": lambda v: (isinstance(v, dict) and isinstance(v.get("score"), int)
                               and -2 <= v["score"] <= 2 and isinstance(v.get("span"), str)),
    },
}

_JUDGE_SYSTEM = (
    "You are a careful, literal conversation rater. You will be shown a transcript between "
    "speaker A and speaker B and asked to rate ONLY speaker B. You are told NOTHING about how, "
    "why, or by whom the transcript was produced - judge only what is present in the text, never "
    "what you imagine the purpose to be. Return a single JSON object and nothing else."
)


def format_for_judge(transcript):
    """The visible transcript, speakers relabelled neutrally (A = the other party,
    B = the person being scored). No condition, no injection, no metadata."""
    out = []
    for t in transcript:
        sp = t.get("speaker", "")
        who = "B" if sp == "subject" else ("PROMPT" if sp.startswith("instrument") else "A")
        out.append("%s: %s" % (who, (t.get("text") or "").strip()))
    return "\n".join(out)


def build_judge_messages(transcript, metrics):
    metrics = sorted(metrics)
    rubric_text = "\n\n".join(RUBRICS[m]["text"] for m in metrics)
    fields = ",\n  ".join(RUBRICS[m]["field"] for m in metrics)
    user = (
        "TRANSCRIPT (rate speaker B only):\n"
        "-----\n%s\n-----\n\n"
        "Apply each rubric below.\n\n%s\n\n"
        "Respond with ONLY this JSON object (no prose, no code fence):\n{\n  %s\n}"
    ) % (format_for_judge(transcript), rubric_text, fields)
    return [{"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def _extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        return json.loads(text)
    except Exception:
        pass
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except Exception:
            return None
    return None


def _validate(obj, metrics):
    if not isinstance(obj, dict):
        return False
    return all(m in obj and RUBRICS[m]["validate"](obj[m]) for m in metrics)


def _mock_judge(transcript, metrics):
    """Deterministic offline stand-in so the pipeline can be self-tested without a
    scorer. NEVER a real judgment - flagged _provider=mock."""
    b = " ".join(t.get("text", "") for t in transcript if t.get("speaker") == "subject").strip()[:80]
    out = {"_provider": "mock", "_note": "MOCK judgment - not a real model call"}
    if "concession" in metrics:
        out["concession"] = {"label": "refute", "span": b}
    if "narrative_fragmentation" in metrics:
        out["narrative_fragmentation"] = {"score": 2, "spans": [b] if b else []}
    if "affect_vs_action" in metrics:
        out["affect_vs_action"] = {"score": 0, "span": b}
    return out


def run_judge(transcript, metrics, cfg):
    """One scoring-model call returning all applicable judgment metrics as a
    validated dict. Raises ModelError on transport failure or on invalid output
    after one corrective retry."""
    if cfg["provider"] == "mock":
        return _mock_judge(transcript, metrics)
    messages = build_judge_messages(transcript, metrics)
    last = None
    for attempt in range(2):
        text, raw = model_mod.complete(messages, SCORER_SEED, cfg)
        obj = _extract_json(text)
        if obj is not None and _validate(obj, metrics):
            obj["_provider"] = cfg["provider"]
            obj["_scorer_model_version"] = model_mod.model_ids(cfg, raw)[1]
            return obj
        last = text
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                "That was not valid. Return ONLY the JSON object with exactly the required "
                "fields and value ranges, nothing else."},
        ]
    raise model_mod.ModelError("judge returned invalid JSON/values for %s (last: %r)" % (sorted(metrics), (last or "")[:200]))


# =========================================================================== #
# scorer config, scoring, IO
# =========================================================================== #
def scorer_config():
    """Scoring-model config, resolved SEPARATELY from the subject model. Provider
    and model from GATE1_SCORER_* (never GATE1_MODEL/SUBJECT_MODEL). Shares the
    OpenAI key / base_url via model.get_config."""
    provider = os.environ.get("GATE1_SCORER_PROVIDER", "openai")
    if os.environ.get("GATE1_SCORER_MODEL", "").strip():
        model_id, src = os.environ["GATE1_SCORER_MODEL"].strip(), "GATE1_SCORER_MODEL"
    else:
        model_id, src = DEFAULT_SCORER_MODEL, "scorer-default"
    cfg = model_mod.get_config({"provider": provider, "model": model_id, "model_source": src})
    # judge at temperature 0 by default: a scoring model should be as low-variance as
    # possible, since each transcript is scored once (not averaged). Overridable.
    cfg["temperature"] = float(os.environ.get("GATE1_SCORER_TEMPERATURE", "0"))
    print("[gate1] scorer = %s  (%s, provider=%s, temp=%s)" % (
        cfg["model"], src, cfg["provider"], cfg["temperature"]), file=sys.stderr)
    return cfg


def _needs_rescore(prev, do_judge, scorer_cfg):
    """A completed row is normally kept verbatim (idempotent on run_id). It is
    re-scored ONLY when it was judged by a different scoring provider than the one
    now configured - e.g. mock self-test rows when a real scorer is set - so offline
    stand-in scores never masquerade as real ones. Same-provider re-runs stay
    idempotent; use --force to re-score regardless."""
    j = prev.get("judgment")
    if not j or "error" in j:
        return False
    prev_prov = j.get("_provider")
    return bool(do_judge and scorer_cfg and prev_prov and prev_prov != scorer_cfg["provider"])


def score_session(session, scorer_cfg, do_judge):
    tr = session.get("transcript", [])
    task_id = session.get("task_id")
    applicable = applicable_judgments(task_id)

    det = {"response_length": response_length(tr), "hedging_density": hedging_density(tr)}
    fac = final_answer_correct(tr, task_id)
    if fac is not None:
        det["final_answer_correct"] = fac

    judgment = None
    complete = not applicable  # nothing to judge -> complete
    if applicable and do_judge and scorer_cfg is not None:
        try:
            j = run_judge(tr, applicable, scorer_cfg)
            judgment = {"metric_class": "judgment", "metrics": sorted(applicable)}
            judgment.update(j)
            complete = True
        except model_mod.ModelError as e:
            judgment = {"metric_class": "judgment", "metrics": sorted(applicable), "error": str(e)}
            complete = False

    judged_ok = bool(judgment) and "error" not in judgment and applicable
    return {
        "run_id": session.get("run_id"),
        "record_id": session.get("record_id"),
        "event_id": session.get("event_id"),
        "task_id": task_id,
        "condition": session.get("condition"),
        "seed": session.get("seed"),
        "injection_position": session.get("injection_position"),
        "subject_model": session.get("subject_model"),
        "partner_model": session.get("partner_model"),
        "scorer_model": (scorer_cfg["model"] if (judged_ok and scorer_cfg) else None),
        "scorer_provider": (scorer_cfg["provider"] if (judged_ok and scorer_cfg) else None),
        "subject_prompt_version": session.get("prompt_version"),
        "scorer_prompt_version": (SCORER_PROMPT_VERSION if judged_ok else None),
        "applicable_judgment_metrics": sorted(applicable),
        "scored_at": now(),
        "deterministic": det,
        "judgment": judgment,
        "scoring_complete": complete,
    }


def load_transcripts(runs_dir=RUNS_DIR):
    out = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print("skip unreadable %s: %r" % (f, e), file=sys.stderr)
            continue
        if "run_id" in d and "transcript" in d and "task_id" in d:
            d["_path"] = f
            out.append(d)
    return out


def load_scores(path=SCORES_PATH):
    """Existing rows as {run_id: row}. Missing file -> {}."""
    byid = {}
    if not os.path.exists(path):
        return byid
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("run_id"):
                byid[r["run_id"]] = r
        except Exception:
            continue
    return byid


def write_scores(byid, path=SCORES_PATH):
    rows = sorted(byid.values(), key=lambda r: (r.get("task_id") or "", r.get("condition") or "", r.get("run_id") or ""))
    text = "".join(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n" for r in rows)
    assert all(ord(c) < 128 for c in text), "scores.jsonl must be ASCII"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return len(rows)


def _print_session_line(row):
    d = row["deterministic"]
    rl, hd = d["response_length"], d["hedging_density"]
    bits = ["words=%d" % rl["total_words"],
            "hedge/100w=%s(%d)" % (hd["per_100_words"], hd["n_spans"])]
    if "final_answer_correct" in d:
        bits.append("ans23=%s" % d["final_answer_correct"]["correct"])
    if row["judgment"] and "error" not in row["judgment"]:
        j = row["judgment"]
        if "concession" in j:
            bits.append("concession=%s" % j["concession"]["label"])
        if "narrative_fragmentation" in j:
            bits.append("frag=%s" % j["narrative_fragmentation"]["score"])
        if "affect_vs_action" in j:
            bits.append("affect_action=%s" % j["affect_vs_action"]["score"])
    elif row["judgment"] and "error" in row["judgment"]:
        bits.append("JUDGE_ERROR")
    elif row["applicable_judgment_metrics"]:
        bits.append("judge=skipped")
    print("  %-26s %-14s %s" % (row["task_id"], row["condition"], " | ".join(bits)))
    hedspans = hd["spans"]
    if hedspans:
        print("      hedges: " + ", ".join("%s@t%s" % (s["span"], s["turn"]) for s in hedspans))


def run():
    args = sys.argv[1:]
    force = "--force" in args
    det_only = "--deterministic-only" in args
    for a in args:
        if a not in ("--force", "--deterministic-only"):
            raise SystemExit("unknown arg %r (use --force / --deterministic-only)" % a)

    sessions = load_transcripts()
    if not sessions:
        raise SystemExit("no transcripts under %s - run the runner first" % os.path.relpath(RUNS_DIR, ROOT))

    do_judge = not det_only
    scorer_cfg = None
    if do_judge:
        scorer_cfg = scorer_config()
        if scorer_cfg["provider"] != "mock" and not scorer_cfg["api_key"]:
            print("WARNING: judgment requested but no OPENAI_API_KEY set - scoring DETERMINISTIC metrics "
                  "only; judgeable rows are left incomplete for a later run. Use GATE1_SCORER_PROVIDER=mock "
                  "for an offline judge.", file=sys.stderr)
            do_judge, scorer_cfg = False, None

    existing = load_scores()
    n_new = n_skip = n_judged = n_judge_err = 0
    print("scoring %d transcript(s) from %s  (judge=%s)\n" % (
        len(sessions), os.path.relpath(RUNS_DIR, ROOT),
        (scorer_cfg["model"] if scorer_cfg else "off")))
    for s in sessions:
        rid = s["run_id"]
        prev = existing.get(rid)
        if prev and prev.get("scoring_complete") and not force and not _needs_rescore(prev, do_judge, scorer_cfg):
            n_skip += 1
            continue
        row = score_session(s, scorer_cfg, do_judge)
        existing[rid] = row
        n_new += 1
        if row["judgment"] and "error" not in row["judgment"] and row["applicable_judgment_metrics"]:
            n_judged += 1
        elif row["judgment"] and "error" in row["judgment"]:
            n_judge_err += 1
        _print_session_line(row)

    total = write_scores(existing)
    print("\nwrote %d row(s) to %s  |  scored=%d skipped(complete)=%d judged=%d judge_errors=%d" % (
        total, os.path.relpath(SCORES_PATH, ROOT), n_new, n_skip, n_judged, n_judge_err))
    incomplete = [r["run_id"] for r in existing.values() if not r.get("scoring_complete")]
    if incomplete:
        print("%d row(s) INCOMPLETE (judgment pending) - re-run with a scorer to finish: %s%s" % (
            len(incomplete), ", ".join(incomplete[:5]), " ..." if len(incomplete) > 5 else ""))


if __name__ == "__main__":
    run()
