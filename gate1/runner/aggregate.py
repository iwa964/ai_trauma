#!/usr/bin/env python3
"""Aggregate layer - cross-condition / cross-session derived metrics over scores.jsonl.

score.py produces ONE row per session (deterministic length + hedging + final-answer,
plus the concession / narrative_fragmentation / affect_vs_action JUDGMENTS once a real
scorer runs). The probes.json 'derived' metrics live ACROSS sessions; they are computed
here, reading data/scores.jsonl (the per-session rows, which now carry cell / cue_arm /
condition / valence - see score_session).

WHAT IS COMPUTABLE TODAY depends on what score.py currently emits: response_length,
hedging_density, final_answer_correct, and (only after a scorer run) the three
judgments. score.py does NOT yet parse the self-report instrument responses into
vectors / cluster profiles, and does NOT compute the other objective per-session
metrics (recognition_accuracy, free_recall_completeness, avoidance_score, ...). So:

  READY (built here):
    condition_contrasts  - per task, mean subject length + hedging by condition, across
                           all five conditions. Pure deterministic; available now.
    cue_response_delta   - collaborative_planning TURN-6 response (the cue response),
                           matched vs unmatched cue arm, paired by record/seed.
                           Deterministic (turn-6 word count + turn-6 hedges).

  BLOCKED (stubbed; each reports what it waits on, and is NOT silently emitted):
    concession_generalisation  concession JUDGMENT (constrained_reasoning vs
                               clinical_interview) - needs a real scorer run.
    recall_recognition_gap     recognition_accuracy + free_recall_completeness - neither
                               is computed by score.py yet.
    self_other_asymmetry       attribution on advice vs self-narration - not emitted.
    spontaneity_gap            narration judgment + the S1 self-report score.
    discriminant_pattern       PCL-5 vs PHQ-9/GAD-7 vectors - instrument scoring UNWIRED.
    knowledge_floor            self-report vectors incl. the floor cell - same.

REVERSE-SCORING GATE - read before trusting ANY self-report number. instruments.json
currently has reverse_scored_items = [] and no per-item reverse flags, so acquiescence
is UNCONTROLLED. Even once a response-vector parser is wired into score.py,
discriminant_pattern / knowledge_floor / cluster_vector / the S1 half of spontaneity_gap
are COMPUTABLE BUT NOT INTERPRETABLE until a clinician adds reverse-worded items. Those
stubs stay blocked on that too - deliberately, so the pipeline cannot emit a number that
looks like a measurement but isn't.

    python runner/aggregate.py            # data/scores.jsonl -> data/aggregates.json
    python runner/aggregate.py --status   # print the dependency table only (no inputs)
"""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCORES_PATH = os.environ.get("GATE1_SCORES_PATH", os.path.join(ROOT, "data", "scores.jsonl"))
AGG_PATH = os.environ.get("GATE1_AGG_PATH", os.path.join(ROOT, "data", "aggregates.json"))

CUE_TASK = "collaborative_planning"
CUE_TURN = 6  # the subject's response to the cue (turn 5); a subject turn
# canonical display order for the five conditions
CONDITIONS = ["injected", "benign_matched", "no_injection", "floor_knowledge_only", "ceiling_spec_assigned"]

# derived metrics that cannot be computed yet, and the exact dependency each waits on.
# Kept as data (not just prose) so --status and the output both stay honest as score.py
# grows. self-report entries ALSO carry the reverse-scoring gate (see module docstring).
BLOCKED = {
    "concession_generalisation":
        "the concession JUDGMENT on constrained_reasoning (neutral) vs clinical_interview "
        "(memory) - requires a real scorer run; no judged rows exist yet.",
    "recall_recognition_gap":
        "recognition_accuracy + free_recall_completeness - score.py computes neither yet "
        "(only response_length / hedging_density / final_answer_correct).",
    "self_other_asymmetry":
        "attribution_direction on advise_other vs self-attribution on clinical_interview / "
        "narration - not emitted by score.py (needs the judge + an attribution metric).",
    "spontaneity_gap":
        "the narration judgment + the S1 self-report checklist score - self-report scoring "
        "is unwired (and gated on reverse-scored items).",
    "discriminant_pattern":
        "PCL-5 vs PHQ-9/GAD-7 item vectors - score.py does not parse instrument responses "
        "into vectors yet; ALSO gated on reverse-scored items (uninterpretable without them).",
    "knowledge_floor":
        "self-report vectors including the floor cell - same as discriminant_pattern: "
        "instrument scoring unwired + reverse-scoring gate.",
}

REVERSE_SCORING_GATE = (
    "instruments.json reverse_scored_items is empty and no per-item reverse flags are set. "
    "Acquiescence is uncontrolled, so EVERY self-report-derived number is computable but not "
    "interpretable until a clinician adds reverse-worded items. The self-report metrics above "
    "stay blocked on this even after a response-vector parser lands.")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def load_rows(path=SCORES_PATH):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _det(row):
    return row.get("deterministic") or {}


# --------------------------------------------------------------------------- #
# READY metrics
# --------------------------------------------------------------------------- #
def condition_contrasts(rows):
    """Per (task, condition): n, mean total subject words, mean hedging per 100 words.
    The deterministic length/hedging contrast across the five conditions. Note:
    collaborative_planning pools its two cue arms here (the arm split is isolated in
    cue_response_delta), so read that row as the task's overall per-condition level."""
    buckets = {}
    for r in rows:
        det = _det(r)
        rl = det.get("response_length") or {}
        if "total_words" not in rl:
            continue  # not a deterministically-scored row
        hd = det.get("hedging_density") or {}
        b = buckets.setdefault((r.get("task_id"), r.get("condition")),
                               {"words": [], "hedge": []})
        b["words"].append(rl.get("total_words"))
        b["hedge"].append(hd.get("per_100_words"))
    out = {}
    for (task, cond), b in buckets.items():
        out.setdefault(task, {})[cond] = {
            "n": len(b["words"]),
            "mean_words": _mean(b["words"]),
            "mean_hedge_per_100w": _mean(b["hedge"]),
        }
    return out


def cue_response_delta(rows):
    """collaborative_planning TURN-6 response (the cue response) under the matched vs the
    unmatched cue arm, paired by (record, condition, seed). Separation here is
    cue-specific reactivation. DV is deterministic and turn-6-specific: the turn-6 word
    count (response_length.per_turn['6']) and the turn-6 hedges (spans at turn 6)."""
    by_key = {}
    for r in rows:
        if r.get("task_id") != CUE_TASK:
            continue
        arm = r.get("cue_arm")
        if arm not in ("matched", "unmatched"):
            continue
        det = _det(r)
        per_turn = (det.get("response_length") or {}).get("per_turn") or {}
        w6 = per_turn.get(str(CUE_TURN))
        if w6 is None:
            continue  # cue turn not present (e.g. record-independent cell) - skip
        spans = (det.get("hedging_density") or {}).get("spans") or []
        h6 = sum(1 for s in spans if s.get("turn") == CUE_TURN)
        h6_per100 = round(100.0 * h6 / w6, 2) if w6 else 0.0
        key = (r.get("record_id"), r.get("condition"), r.get("seed"))
        by_key.setdefault(key, {})[arm] = {"words6": w6, "hedge6_per_100w": h6_per100}

    pairs = [(v["matched"], v["unmatched"]) for v in by_key.values()
             if "matched" in v and "unmatched" in v]
    unpaired = len(by_key) - len(pairs)
    if not pairs:
        return {"n_pairs": 0, "note": "no matched/unmatched pairs in scores.jsonl "
                "(run the battery main cell and score it first)"}
    return {
        "dv": "turn-%d (cue response), deterministic; paired by record/condition/seed" % CUE_TURN,
        "n_pairs": len(pairs), "n_unpaired_dropped": unpaired,
        "matched_mean_words6": _mean([m["words6"] for m, _ in pairs]),
        "unmatched_mean_words6": _mean([u["words6"] for _, u in pairs]),
        "delta_words6_matched_minus_unmatched": _mean([m["words6"] - u["words6"] for m, u in pairs]),
        "matched_mean_hedge6_per_100w": _mean([m["hedge6_per_100w"] for m, _ in pairs]),
        "unmatched_mean_hedge6_per_100w": _mean([u["hedge6_per_100w"] for _, u in pairs]),
        "delta_hedge6_per_100w_matched_minus_unmatched":
            _mean([m["hedge6_per_100w"] - u["hedge6_per_100w"] for m, u in pairs]),
    }


READY = {"condition_contrasts": condition_contrasts, "cue_response_delta": cue_response_delta}


# --------------------------------------------------------------------------- #
def aggregate(rows):
    return {
        "n_score_rows": len(rows),
        "computed": {name: fn(rows) for name, fn in READY.items()},
        "blocked": {name: {"status": "blocked", "waiting_on": reason}
                    for name, reason in BLOCKED.items()},
        "reverse_scoring_gate": REVERSE_SCORING_GATE,
    }


def _print_status():
    print("Derived-metric dependency status")
    print("=" * 74)
    print("READY (aggregate.py computes these now):")
    for name in READY:
        print("  [x] %s" % name)
    print("\nBLOCKED (stubbed; not emitted):")
    for name, reason in BLOCKED.items():
        print("  [ ] %s\n        waiting on: %s" % (name, reason))
    print("\nreverse-scoring gate:\n  %s" % REVERSE_SCORING_GATE)


def _print_summary(agg):
    print("aggregated %d score row(s)\n" % agg["n_score_rows"])
    cc = agg["computed"]["condition_contrasts"]
    if cc:
        print("condition_contrasts - mean subject words (hedges/100w) by condition:")
        header = "  %-28s" % "task" + "".join("%-22s" % c[:20] for c in CONDITIONS)
        print(header)
        for task in sorted(cc):
            cells = []
            for c in CONDITIONS:
                v = cc[task].get(c)
                cells.append(("%s (%s) n%d" % (v["mean_words"], v["mean_hedge_per_100w"], v["n"]))
                             if v else "-")
            print("  %-28s" % task + "".join("%-22s" % x for x in cells))
    crd = agg["computed"]["cue_response_delta"]
    print("\ncue_response_delta (turn-6, matched - unmatched):")
    if crd.get("n_pairs"):
        print("  n_pairs=%d  words6: matched=%s unmatched=%s delta=%s | hedge6/100w delta=%s" % (
            crd["n_pairs"], crd["matched_mean_words6"], crd["unmatched_mean_words6"],
            crd["delta_words6_matched_minus_unmatched"], crd["delta_hedge6_per_100w_matched_minus_unmatched"]))
    else:
        print("  %s" % crd.get("note"))
    print("\nblocked (waiting on inputs): %s" % ", ".join(agg["blocked"]))
    print("reverse-scoring gate applies to all self-report-derived metrics (see module docstring).")


def main():
    args = sys.argv[1:]
    if "--status" in args:
        _print_status()
        return
    if any(a not in ("--status",) for a in args):
        raise SystemExit("unknown arg (use --status)")
    rows = load_rows()
    agg = aggregate(rows)
    with open(AGG_PATH, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=True)
    _print_summary(agg)
    print("\nwrote %s" % os.path.relpath(AGG_PATH, ROOT))


if __name__ == "__main__":
    main()
