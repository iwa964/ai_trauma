#!/usr/bin/env python3
"""Self-report instrument loader + scoring (Gate 1 self_report session).

Items are transcribed VERBATIM from the official instruments into
data/instruments.json (provenance + licensing in data/SOURCES.md). This module
loads them, applies the time-anchor rewrite, presents the instrument, and scores
it.

  * TIME-ANCHOR REWRITE. Each instrument stores its verbatim instruction and the
    exact calendar phrase it uses (time_anchor_original). On load, that phrase is
    replaced with "since that happened" (time_anchor). The result is deterministic
    and condition-independent -> byte-identical across ALL conditions including
    no_injection. load() refuses any instrument whose presented instruction still
    carries a calendar anchor.
  * SCORING is item-level. score() returns the response VECTOR and, for PCL-5, the
    B/C/D/E cluster profile. Grand totals are DISCARDED as a severity DV - a total
    is computed only inside the cutoff-leakage check (cutoff_clustering), never
    reported as a score. reverse_scored items (per-item flag, or id listed in the
    top-level reverse_scored_items) are inverted on the response scale.
  * Scoring stays separate from execution: build_prompt() is the only thing the
    runner calls.

    python runner/self_report.py     # offline self-test
"""
from __future__ import annotations

import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INSTRUMENTS_PATH = os.path.join(ROOT, "data", "instruments.json")

CANONICAL_ANCHOR = "since that happened"
# every calendar phrase that must NOT survive the rewrite (defensive)
_FORBIDDEN_ANCHORS = ("in the past month", "in the last month",
                      "over the last 2 weeks", "over the last two weeks")
INSTRUMENT_NAMES = ("pcl5", "phq9", "gad7")

_LOADED = None


def _present_instruction(inst):
    """Verbatim instruction with the instrument's calendar phrase replaced by the
    canonical anchor. Case-sensitive replace of the exact recorded phrase."""
    return inst["instruction_original"].replace(
        inst["time_anchor_original"], inst.get("time_anchor", CANONICAL_ANCHOR))


def _assert_anchored(name, instruction):
    low = instruction.lower()
    if any(p in low for p in _FORBIDDEN_ANCHORS):
        raise AssertionError("%s: a calendar anchor survived the rewrite: %r" % (name, instruction))
    if CANONICAL_ANCHOR not in instruction:
        raise AssertionError("%s: instruction lost the '%s' anchor" % (name, CANONICAL_ANCHOR))


def load(path=INSTRUMENTS_PATH, force=False):
    """Load instruments.json, apply the time-anchor rewrite, validate. Cached."""
    global _LOADED
    if _LOADED is not None and not force:
        return _LOADED
    doc = json.load(open(path, encoding="utf-8"))
    reverse_ids = set(doc.get("reverse_scored_items", []))
    instruments = {}
    for name in INSTRUMENT_NAMES:
        inst = dict(doc[name])
        inst["instruction"] = _present_instruction(inst)
        _assert_anchored(name, inst["instruction"])
        instruments[name] = inst
    _LOADED = {"instruments": instruments, "reverse_scored_items": reverse_ids}
    return _LOADED


def build_prompt(name, loaded=None):
    """Subject-facing instrument text. No condition argument -> byte-identical
    across every condition including no_injection (the runner asserts this)."""
    loaded = loaded or load()
    inst = loaded["instruments"][name]
    scale = "; ".join("%d = %s" % (i, lab) for i, lab in enumerate(inst["response_scale"]))
    lines = [inst["instruction"], "(%s)" % scale]
    for n, it in enumerate(inst["items"], 1):
        lines.append("%2d. %s" % (n, it["text"]))
    return "\n".join(lines)


# --- scoring (analysis-time, pure - never called during execution) ----------
def _is_reverse(item, reverse_ids):
    return bool(item.get("reverse_scored")) or item["id"] in reverse_ids


def score(name, responses, loaded=None):
    """Item-level response VECTOR + (PCL-5) B/C/D/E cluster profile. Grand total is
    DISCARDED as a DV and is not returned. reverse_scored items are inverted."""
    loaded = loaded or load()
    inst = loaded["instruments"][name]
    items, hi = inst["items"], len(inst["response_scale"]) - 1
    rev = loaded["reverse_scored_items"]
    if len(responses) != len(items):
        raise ValueError("%s: %d responses for %d items" % (name, len(responses), len(items)))
    vector, clusters = [], collections.defaultdict(int)
    for it, r in zip(items, responses):
        if not (0 <= r <= hi):
            raise ValueError("%s: response %r out of scale [0,%d]" % (it["id"], r, hi))
        v = (hi - r) if _is_reverse(it, rev) else r
        vector.append(v)
        if it.get("cluster"):
            clusters[it["cluster"]] += v
    return {"instrument": name, "n_items": len(items), "vector": vector,
            "cluster_profile": dict(clusters)}  # {} for phq9 / gad7


def pcl5_total(responses, loaded=None):
    """Sum of PCL-5 item scores. Delegates to score() so the SAME length + scale
    validation applies - malformed input (short vector, out-of-scale value) raises
    instead of silently producing a corrupt total that would poison the leakage
    check. Used ONLY by the cutoff-leakage check below - never a reported severity
    score."""
    return sum(score("pcl5", responses, loaded)["vector"])


def cutoff_clustering(totals, cutoff=None, window=5, concentration=0.5, loaded=None):
    """Flag whether a set of PCL-5 totals CLUSTERS near the clinical cutoff - a tell
    that the model is performing a diagnosis at the number it knows rather than
    reporting a state (cutoff leakage). Totals are used ONLY here. Fraction within
    +/-window of the cutoff; flagged if that fraction >= concentration."""
    loaded = loaded or load()
    if cutoff is None:
        cutoff = loaded["instruments"]["pcl5"].get("clinical_cutoff")
    totals = list(totals)
    if not totals:
        return {"n": 0, "flagged": False, "note": "no totals"}
    near = [t for t in totals if abs(t - cutoff) <= window]
    frac = len(near) / len(totals)
    mean = sum(totals) / len(totals)
    var = sum((t - mean) ** 2 for t in totals) / len(totals)
    flagged = frac >= concentration
    return {"n": len(totals), "cutoff": cutoff, "window": window, "near_cutoff": len(near),
            "near_cutoff_fraction": round(frac, 3), "mean": round(mean, 2),
            "variance": round(var, 2), "flagged": flagged,
            "note": ("clusters near cutoff - possible performed diagnosis (cutoff leakage)"
                     if flagged else "no clustering at cutoff")}


def _self_test():
    loaded = load()
    print("loaded %d instruments from %s\n" % (len(loaded["instruments"]), os.path.relpath(INSTRUMENTS_PATH, ROOT)))
    for name in INSTRUMENT_NAMES:
        inst = loaded["instruments"][name]
        print("=" * 74)
        print("%s  (%d items)  source: %s" % (name.upper(), len(inst["items"]), inst["source"]))
        print("=" * 74)
        print("\n".join(build_prompt(name, loaded).splitlines()[:3]) + "\n  ...")
        assert CANONICAL_ANCHOR in build_prompt(name, loaded)

    cl = collections.Counter(it["cluster"] for it in loaded["instruments"]["pcl5"]["items"])
    print("\nPCL-5 cluster counts (DSM-5):", dict(cl))

    print("\n-- scoring demo (synthetic responses; item vectors + cluster profile, totals discarded) --")
    for name, resp in (("pcl5", [2] * 20), ("phq9", [1] * 9), ("gad7", [1] * 7)):
        sc = score(name, resp, loaded)
        print("%-5s vector[:6]=%s cluster_profile=%s" % (name, sc["vector"][:6], sc["cluster_profile"] or "-"))

    print("\n-- PCL-5 cutoff-clustering check (totals used ONLY here) --")
    for label, totals in (("piled near cutoff", [31, 32, 33, 33, 34, 35, 33, 32]),
                          ("well spread", [4, 12, 20, 33, 48, 55, 8, 70])):
        r = cutoff_clustering(totals, loaded=loaded)
        print("  %-18s -> flagged=%s (%d/%d within +/-5, var=%.1f, cutoff=%s)" % (
            label, r["flagged"], r["near_cutoff"], r["n"], r["variance"], r["cutoff"]))


if __name__ == "__main__":
    _self_test()
