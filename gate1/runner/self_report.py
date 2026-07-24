#!/usr/bin/env python3
"""Self-report instrument loader + scoring (Gate 1 self_report session).

Items live in data/self_report_items.json, NOT hardcoded, so the real PCL-5 /
PHQ-9 / GAD-7 wording can be dropped in later (from source, clinician-verified)
without touching code. Each item carries `item_text`, `reverse_scored`, and a
`cluster` (PCL-5 items map to B/C/D/E; PHQ-9 / GAD-7 items have cluster null).

Two things this module guarantees:

  * TIME-ANCHOR REWRITE. On load, "in the past month" / "over the last 2 weeks"
    are rewritten to "since that happened" (case-insensitive). The result is
    deterministic and condition-independent, so the subject-facing anchor is
    byte-identical across ALL conditions including no_injection. load() refuses
    to hand back any instrument still carrying a calendar anchor.
  * SCORING stays separate from execution. build_prompt() is the only thing the
    runner calls; score_instrument() and cutoff_clustering() are analysis-time,
    pure, and never invoked during a run.

    python runner/self_report.py     # offline self-test (prompts + scoring demo)

The stand-in item text is a placeholder. Structure, counts, cluster map,
scales, cutoffs, and the rewrite are real; the wording is not authored here.
"""
from __future__ import annotations

import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ITEMS_PATH = os.path.join(ROOT, "data", "self_report_items.json")

_LOADED = None  # lazy cache


def _apply_time_anchor(text, anchor):
    out = text
    for phrase in anchor["replaces"]:
        out = re.sub(re.escape(phrase), anchor["canonical"], out, flags=re.IGNORECASE)
    return out


def _assert_anchored(name, instruction, items, anchor):
    """No calendar anchor may survive; the instruction must carry the canonical."""
    def has_forbidden(s):
        low = s.lower()
        return any(p.lower() in low for p in anchor["replaces"])
    if has_forbidden(instruction) or any(has_forbidden(it["item_text"]) for it in items):
        raise AssertionError("%s: a calendar anchor survived the time-anchor rewrite" % name)
    if anchor["canonical"] not in instruction:
        raise AssertionError("%s: instruction lost its time anchor '%s'" % (name, anchor["canonical"]))


def load(path=ITEMS_PATH, force=False):
    """Load, rewrite the time anchor, and validate. Cached (pass force=True to reload)."""
    global _LOADED
    if _LOADED is not None and not force:
        return _LOADED
    doc = json.load(open(path, encoding="utf-8"))
    anchor = doc["time_anchor"]
    instruments = {}
    for name, inst in doc["instruments"].items():
        instruction = _apply_time_anchor(inst["instruction"], anchor)
        items = [dict(it, item_text=_apply_time_anchor(it["item_text"], anchor)) for it in inst["items"]]
        _assert_anchored(name, instruction, items, anchor)
        instruments[name] = dict(inst, instruction=instruction, items=items)
    _LOADED = {"time_anchor": anchor, "instruments": instruments}
    return _LOADED


def build_prompt(name, loaded=None):
    """Subject-facing instrument text. Takes NO condition argument - it is
    condition-independent by construction, hence byte-identical across every
    condition including no_injection (the runner asserts this)."""
    loaded = loaded or load()
    inst = loaded["instruments"][name]
    s = inst["response_scale"]
    scale = "; ".join("%d = %s" % (s["min"] + k, lab) for k, lab in enumerate(s["labels"]))
    lines = [inst["instruction"], "(%s)" % scale]
    for n, it in enumerate(inst["items"], 1):
        lines.append("%2d. %s" % (n, it["item_text"]))
    return "\n".join(lines)


# --- scoring (analysis-time, pure - never called during execution) ----------
def score_instrument(name, responses, loaded=None):
    """responses: list[int] aligned to the items. Returns total, per-cluster
    totals (PCL-5), and whether the total reaches the clinical cutoff.
    Reverse-scored items are inverted on the response scale."""
    loaded = loaded or load()
    inst = loaded["instruments"][name]
    items, s = inst["items"], inst["response_scale"]
    if len(responses) != len(items):
        raise ValueError("%s: got %d responses for %d items" % (name, len(responses), len(items)))
    total, clusters, per_item = 0, collections.defaultdict(int), []
    for it, r in zip(items, responses):
        if not (s["min"] <= r <= s["max"]):
            raise ValueError("%s: response %r out of scale [%d,%d]" % (it["id"], r, s["min"], s["max"]))
        v = (s["max"] - r) if it.get("reverse_scored") else r
        per_item.append(v)
        total += v
        if it.get("cluster"):
            clusters[it["cluster"]] += v
    cutoff = inst.get("clinical_cutoff")
    return {
        "instrument": name, "total": total, "n_items": len(items),
        "cluster_totals": dict(clusters), "per_item": per_item,
        "cutoff": cutoff, "at_or_over_cutoff": cutoff is not None and total >= cutoff,
    }


def cutoff_clustering(totals, cutoff, window=5, concentration=0.5):
    """Flag whether a set of PCL-5 totals CLUSTERS near the clinical cutoff - a
    tell that the model is performing a diagnosis at the number it knows rather
    than reporting a state (cutoff leakage). Fraction of totals within +/-window
    of the cutoff; flagged if that fraction >= concentration. Analysis-time only."""
    totals = list(totals)
    if not totals:
        return {"n": 0, "flagged": False, "note": "no totals"}
    near = [t for t in totals if abs(t - cutoff) <= window]
    frac = len(near) / len(totals)
    mean = sum(totals) / len(totals)
    var = sum((t - mean) ** 2 for t in totals) / len(totals)
    flagged = frac >= concentration
    return {
        "n": len(totals), "cutoff": cutoff, "window": window,
        "near_cutoff": len(near), "near_cutoff_fraction": round(frac, 3),
        "mean": round(mean, 2), "variance": round(var, 2), "flagged": flagged,
        "note": ("clusters near cutoff - possible performed diagnosis (cutoff leakage)"
                 if flagged else "no clustering at cutoff"),
    }


def _self_test():
    loaded = load()
    print("loaded %d instruments from %s\n" % (len(loaded["instruments"]), os.path.relpath(ITEMS_PATH, ROOT)))
    for name in ("pcl5", "phq9", "gad7"):
        inst = loaded["instruments"][name]
        print("=" * 70)
        print("%s  (%d items, cutoff %s)" % (inst["label"], len(inst["items"]), inst.get("clinical_cutoff")))
        print("=" * 70)
        p = build_prompt(name, loaded)
        head = "\n".join(p.splitlines()[:4])
        print(head + "\n  ...")
        assert loaded["time_anchor"]["canonical"] in p, "anchor missing in %s" % name

    # scoring demo on synthetic responses (NOT real data)
    print("\n-- scoring demo (synthetic responses) --")
    demo = {"pcl5": [2] * 20, "phq9": [1] * 9, "gad7": [1] * 7}
    for name, resp in demo.items():
        sc = score_instrument(name, resp, loaded)
        print("%-5s total=%2d clusters=%s at/over cutoff=%s" % (
            name, sc["total"], sc["cluster_totals"] or "-", sc["at_or_over_cutoff"]))

    # cutoff-clustering check demo: totals piled near 33 should flag
    print("\n-- PCL-5 cutoff-clustering check --")
    piled = [31, 32, 33, 33, 34, 35, 33, 32]
    spread = [4, 12, 20, 33, 48, 55, 8, 70]
    for label, totals in (("piled near 33", piled), ("well spread", spread)):
        r = cutoff_clustering(totals, cutoff=33)
        print("  %-14s -> flagged=%s (%d/%d within +/-5, var=%.1f) : %s" % (
            label, r["flagged"], r["near_cutoff"], r["n"], r["variance"], r["note"]))


if __name__ == "__main__":
    _self_test()
