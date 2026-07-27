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
                           Compared over COMMON TURNS ONLY (see below).
    cue_response_delta   - collaborative_planning TURN-6 response (the cue response),
                           matched vs unmatched cue arm, paired by record/seed.
                           Deterministic (turn-6 word count + turn-6 hedges).

TWO COMPARABILITY RULES, both load-bearing - a figure here is meaningless without them:

  * COMMON TURNS ONLY. run.py drops the memory-presupposing turns (and the subject turns
    answering them) in the record-independent conditions, so sessions are NOT the same
    length in every condition: clinical_interview runs 6 subject turns under injected but
    4 under no_injection / ceiling; collaborative_planning 5 vs 4. Summing a whole session
    would report those conditions as shorter because FEWER TURNS RAN - a ~50% artefact on
    the two tasks that matter most, in the same direction as the hypothesis. Contrasts are
    therefore computed only over turns present in EVERY condition for that task, with the
    compared and excluded turns reported, plus a per-turn normalised figure.
  * WITHIN EXPERIMENT STRATUM. run_id includes the subject model and prompt version, so a
    scores.jsonl can legitimately hold several experiments; re-running after changing
    GATE1_MODEL ADDS rows rather than replacing them. Everything is computed within a
    stratum (subject_model, subject_prompt_version, build_sig, injection_position) and
    cue pairs are KEYED by it - otherwise two experiments collide on one pair key and a
    'pair' can combine a matched arm from one with an unmatched arm from another.

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


# --- experiment identity ----------------------------------------------------- #
# A scores.jsonl may legitimately hold rows from MORE THAN ONE experiment: run.py's
# run_id includes the subject model and prompt version, so re-running after changing
# GATE1_MODEL adds rows rather than replacing them. Pooling those into one mean would
# silently average two different subjects. Every aggregate is therefore computed WITHIN
# a stratum, and pairs are keyed by it. (pilot_refusal.aggregate solves the same problem
# by filtering to the current config; aggregate.py has no "current" config to filter
# against - it only reads rows - so it stratifies and reports every stratum instead.)
STRATUM_FIELDS = ("subject_model", "subject_prompt_version", "build_sig", "injection_position",
                  "partner")
# Sentinel for the partner component on sessions where no LIVE partner call was made -
# the same value run.py uses in its run_id partner_key, so the two agree by construction.
NO_LIVE_PARTNER = "none"


def live_partner_tasks(rows):
    """Task ids whose sessions actually made a LIVE partner call, detected from the
    recorded partner_model_versions (run.py populates it only on a real partner call).
    Derived from the data rather than hardcoding clinical_interview, so it stays true if
    the battery gains another live-partner task.

    This scoping matters in BOTH directions: the partner model shapes only these tasks,
    so including partner identity everywhere would split every scripted-partner task into
    spurious strata whenever GATE1_PARTNER_MODEL changed, while omitting it here would
    average responses elicited by different live partners into one clinical figure. It
    mirrors run.py's _has_live_partner, which scopes the run_id partner_key the same way.

    Rows written before partner_model_versions existed carry no signal, so no task
    registers as live-partner and the partner component stays the sentinel - the previous
    behaviour, never a spurious split."""
    return frozenset(r.get("task_id") for r in rows if r.get("partner_model_versions"))


def _stratum(row, live_tasks=frozenset()):
    partner = ("%s/%s" % (row.get("partner_model"), row.get("partner_provider"))
               if row.get("task_id") in live_tasks else NO_LIVE_PARTNER)
    return (row.get("subject_model"), row.get("subject_prompt_version"),
            row.get("build_sig"), row.get("injection_position"), partner)


def _stratum_label(st):
    return " | ".join(str(v) for v in st)


def _turn_keys(row):
    """Subject-turn indices present in this session, as a set of strings."""
    return set(((_det(row).get("response_length") or {}).get("per_turn") or {}).keys())


def _restricted(row, turns):
    """(words, hedge_spans) counted over ONLY the given subject turns. Both measures come
    from data score.py already records per turn (response_length.per_turn and the hedge
    spans' turn index), so a like-for-like subset can be recomputed without re-scoring."""
    per_turn = (_det(row).get("response_length") or {}).get("per_turn") or {}
    words = sum(v for k, v in per_turn.items() if k in turns)
    spans = (_det(row).get("hedging_density") or {}).get("spans") or []
    hedges = sum(1 for s in spans if str(s.get("turn")) in turns)
    return words, hedges


# --------------------------------------------------------------------------- #
# READY metrics
# --------------------------------------------------------------------------- #
def condition_contrasts(rows, live_tasks=frozenset()):
    """Deterministic length + hedging by condition, per task, WITHIN an experiment stratum.

    Two comparability rules, both load-bearing:

    * COMMON TURNS ONLY. run.py drops the memory-presupposing partner turns (and the
      subject turns answering them) in the record-independent conditions, so a session is
      not the same length everywhere: clinical_interview runs 6 subject turns under
      injected but 4 under no_injection / ceiling, collaborative_planning 5 vs 4. Summing
      the whole session would report those conditions as structurally shorter because
      FEWER TURNS RAN, not because the responses got shorter - a ~50% artefact on exactly
      the two tasks that matter most. So the contrast is computed only over the turns
      present in EVERY condition for that task, and the compared/dropped turns are
      reported. mean_words_per_turn is given too, as a normalised cross-check.
    * WITHIN STRATUM. Rows are grouped by experiment identity first (see STRATUM_FIELDS),
      never pooled across subject models / prompt versions / builds / injection positions.

    collaborative_planning still pools its two cue arms here (the arm split is isolated in
    cue_response_delta) - read that row as the task's overall per-condition level."""
    # group rows by (stratum, task), keeping only deterministically-scored sessions
    groups = {}
    for r in rows:
        if not _turn_keys(r):
            continue  # not a deterministically-scored row
        groups.setdefault((_stratum(r, live_tasks), r.get("task_id")), []).append(r)

    out = {}
    for (st, task), grp in groups.items():
        # turns present in every CONDITION of this task (intersect per-condition turn sets)
        by_cond = {}
        for r in grp:
            by_cond.setdefault(r.get("condition"), []).append(r)
        per_cond_turns = [set.intersection(*(_turn_keys(r) for r in rs)) for rs in by_cond.values()]
        common = set.intersection(*per_cond_turns) if per_cond_turns else set()
        all_turns = set.union(*per_cond_turns) if per_cond_turns else set()
        if not common:
            continue
        entry = out.setdefault(_stratum_label(st), {
            "experiment": dict(zip(STRATUM_FIELDS, st)), "tasks": {}})
        conds = {}
        for cond, rs in by_cond.items():
            words, hedges = [], []
            for r in rs:
                w, h = _restricted(r, common)
                words.append(w)
                hedges.append(round(100.0 * h / w, 2) if w else 0.0)
            mw = _mean(words)
            conds[cond] = {
                "n": len(rs),
                "mean_words_common_turns": mw,
                "mean_words_per_turn": (round(mw / len(common), 3) if mw is not None else None),
                "mean_hedge_per_100w_common_turns": _mean(hedges),
            }
        entry["tasks"][task] = {
            "turns_compared": sorted(common, key=int),
            "n_turns_compared": len(common),
            "turns_excluded_not_in_all_conditions": sorted(all_turns - common, key=int),
            "conditions": conds,
        }
    return out


def cue_response_delta(rows, live_tasks=frozenset()):
    """collaborative_planning TURN-6 response (the cue response) under the matched vs the
    unmatched cue arm. Separation here is cue-specific reactivation. DV is deterministic
    and turn-6-specific: the turn-6 word count (response_length.per_turn['6']) and the
    turn-6 hedges (spans at turn 6) - so the two arms are turn-comparable by construction
    (both are main-cell sessions that ran the cue turn).

    Pairs are keyed by (STRATUM, record, condition, seed) and reported per stratum. The
    stratum is load-bearing, not cosmetic: run_id distinguishes runs from different
    subject models / prompt versions, so without it two rows from DIFFERENT experiments
    collide on one key, each arm overwrites by file order, and the 'pair' can end up
    combining a matched response from one experiment with an unmatched response from
    another - a meaningless delta counted as valid."""
    by_stratum = {}
    for r in rows:
        if r.get("task_id") != CUE_TASK:
            continue
        arm = r.get("cue_arm")
        if arm not in ("matched", "unmatched"):
            continue
        turn_key = str(CUE_TURN)
        if turn_key not in _turn_keys(r):
            continue  # the cue turn did not RUN (record-independent cell) - not scorable
        # A turn that ran but produced an empty answer is words6 == 0: a real, and
        # possibly the most interesting, outcome (going silent under a matched cue).
        # Presence of the turn key, never truthiness of the count, decides inclusion -
        # a `if not w6` test would silently drop exactly those cases and bias the delta.
        w6, h6 = _restricted(r, {turn_key})
        key = (r.get("record_id"), r.get("condition"), r.get("seed"))
        st = by_stratum.setdefault(_stratum(r, live_tasks), {})
        st.setdefault(key, {})[arm] = {
            "words6": w6, "hedge6_per_100w": (round(100.0 * h6 / w6, 2) if w6 else 0.0)}

    out = {}
    for st, by_key in by_stratum.items():
        pairs = [(v["matched"], v["unmatched"]) for v in by_key.values()
                 if "matched" in v and "unmatched" in v]
        if not pairs:
            continue
        out[_stratum_label(st)] = {
            "experiment": dict(zip(STRATUM_FIELDS, st)),
            "dv": "turn-%d (cue response), deterministic; paired by record/condition/seed "
                  "within this experiment stratum" % CUE_TURN,
            "n_pairs": len(pairs), "n_unpaired_dropped": len(by_key) - len(pairs),
            "matched_mean_words6": _mean([m["words6"] for m, _ in pairs]),
            "unmatched_mean_words6": _mean([u["words6"] for _, u in pairs]),
            "delta_words6_matched_minus_unmatched": _mean([m["words6"] - u["words6"] for m, u in pairs]),
            "matched_mean_hedge6_per_100w": _mean([m["hedge6_per_100w"] for m, _ in pairs]),
            "unmatched_mean_hedge6_per_100w": _mean([u["hedge6_per_100w"] for _, u in pairs]),
            "delta_hedge6_per_100w_matched_minus_unmatched":
                _mean([m["hedge6_per_100w"] - u["hedge6_per_100w"] for m, u in pairs]),
        }
    if not out:
        return {"n_strata": 0, "note": "no matched/unmatched pairs in scores.jsonl "
                "(run the battery main cell and score it first)"}
    return out


READY = {"condition_contrasts": condition_contrasts, "cue_response_delta": cue_response_delta}


# --------------------------------------------------------------------------- #
def _experiment_cores(rows, live):
    """Distinct strata IGNORING the partner component. Within ONE battery run the partner
    component legitimately differs by task (a live-partner task carries the partner id, a
    scripted-partner task carries the sentinel), so counting raw strata would cry
    'more than one experiment' on a perfectly clean single run - and a false contamination
    alarm is worse than none, because it teaches the reader to ignore the real one. Only a
    difference in the CORE (subject model / prompt version / build / position), or two
    different partners within one core, means genuinely mixed data."""
    return {_stratum(r, live)[:-1] for r in rows}


def _mixed_partners(rows, live):
    """core -> set of partner ids, for cores where a live-partner task ran under MORE THAN
    ONE partner. That is real mixing and is worth flagging on its own."""
    seen = {}
    for r in rows:
        if r.get("task_id") not in live:
            continue
        st = _stratum(r, live)
        seen.setdefault(st[:-1], set()).add(st[-1])
    return {core: sorted(ps) for core, ps in seen.items() if len(ps) > 1}


def aggregate(rows):
    live = live_partner_tasks(rows)
    strata = sorted({_stratum_label(_stratum(r, live)) for r in rows})
    cores = _experiment_cores(rows, live)
    mixed = _mixed_partners(rows, live)
    return {
        "n_score_rows": len(rows),
        "n_strata": len(strata),
        "strata": strata,
        "stratum_fields": list(STRATUM_FIELDS),
        "live_partner_tasks": sorted(live),
        # genuine-mixing signals, distinct from the benign per-task partner split
        "n_experiment_cores": len(cores),
        "mixed_partners_within_core": {_stratum_label(c): ps for c, ps in mixed.items()},
        "stratum_note": ("Every computed figure is WITHIN one experiment stratum. Rows from "
                         "different subject models / prompt versions / builds / injection "
                         "positions are never pooled, and cue pairs are keyed by stratum so a "
                         "pair can never mix arms from two experiments. The partner component "
                         "applies to live-partner tasks only (see live_partner_tasks); "
                         "elsewhere the partner never touched the session, so it is the "
                         "sentinel '%s' rather than a spurious split." % NO_LIVE_PARTNER),
        "computed": {name: fn(rows, live) for name, fn in READY.items()},
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
    print("aggregated %d score row(s) in %d stratum/strata (%d experiment core(s))\n" % (
        agg["n_score_rows"], agg["n_strata"], agg["n_experiment_cores"]))
    if agg["n_experiment_cores"] > 1:
        print("NOTE: scores.jsonl holds MORE THAN ONE experiment (subject model / prompt "
              "version / build / position).\n      Every figure below is within a single "
              "stratum - they are never pooled across experiments.\n")
    if agg["mixed_partners_within_core"]:
        for core, ps in sorted(agg["mixed_partners_within_core"].items()):
            print("NOTE: live-partner sessions under [%s] ran with MORE THAN ONE partner (%s);\n"
                  "      those are reported as separate strata, never averaged together.\n"
                  % (core, ", ".join(ps)))
    elif agg["n_strata"] > agg["n_experiment_cores"]:
        print("(One experiment; the extra stratum is just the partner component - live-partner\n"
              " tasks carry the partner id, scripted-partner tasks carry '%s'. Not mixed data.)\n"
              % NO_LIVE_PARTNER)
    cc = agg["computed"]["condition_contrasts"]
    for label, strat in sorted(cc.items()):
        print("condition_contrasts [%s]" % label)
        print("  mean subject words over COMMON turns (hedges/100w on those turns), by condition:")
        print("  %-26s %-7s" % ("task", "turns") + "".join("%-22s" % c[:20] for c in CONDITIONS))
        for task in sorted(strat["tasks"]):
            t = strat["tasks"][task]
            cells = []
            for c in CONDITIONS:
                v = t["conditions"].get(c)
                cells.append(("%s (%s) n%d" % (v["mean_words_common_turns"],
                                               v["mean_hedge_per_100w_common_turns"], v["n"]))
                             if v else "-")
            print("  %-26s %-7s" % (task, len(t["turns_compared"])) + "".join("%-22s" % x for x in cells))
            if t["turns_excluded_not_in_all_conditions"]:
                print("      (excluded turns %s - not run in every condition, so not comparable)"
                      % ",".join(t["turns_excluded_not_in_all_conditions"]))
        print("")
    crd = agg["computed"]["cue_response_delta"]
    print("cue_response_delta (turn-%d, matched - unmatched):" % CUE_TURN)
    if crd.get("note"):
        print("  %s" % crd["note"])
    else:
        for label, d in sorted(crd.items()):
            print("  [%s] n_pairs=%d  words6: matched=%s unmatched=%s delta=%s | hedge6/100w delta=%s" % (
                label, d["n_pairs"], d["matched_mean_words6"], d["unmatched_mean_words6"],
                d["delta_words6_matched_minus_unmatched"],
                d["delta_hedge6_per_100w_matched_minus_unmatched"]))
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
