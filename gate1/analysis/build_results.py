#!/usr/bin/env python3
"""Build analysis/results.html - a single self-contained results page.

Same construction as dist/gate1.html: one file, data inlined as
<script type="application/json">, no external requests, CSS variables + a
prefers-color-scheme block for theming, event delegation instead of per-element
listeners, and click-anything-opens-a-drawer.

    python analysis/build_results.py data/scores_2026-07-29_finalrun.jsonl
    python analysis/build_results.py <scores.jsonl> [--aggregates data/aggregates.json]
                                                    [--out analysis/results.html]
                                                    [--stratum-build eaf66051]

The scores path is a REQUIRED ARGUMENT - the live data/scores.jsonl is never
hardcoded, so rebuilding the page cannot silently pick up a half-scored working
file.

TWO embedded arrays, not one:
  data-cells    per-cell aggregates for rendering - task x condition x
                {n, mean/sd words, mean/sd hedge per 100w}, plus the
                benign_matched-normalised index used by panel 1.
  data-sessions a SLIMMED per-session list for the drawers - run_id, task,
                condition, words, concession label + span, ans23. The full hedge
                span lists are dropped: they are ~20x the size of everything else
                and nothing on the page reads them.

METHOD NOTES ENFORCED HERE (not left to the reader)
  * STRATIFY, NEVER POOL. Every figure is computed within one experiment stratum
    (subject_model, prompt_version, build_sig, injection_position, partner), using
    aggregate.py's own _stratum so the split is identical to data/aggregates.json.
    clinical_interview lands in a DIFFERENT stratum from every other task because it
    is the only task with a live partner - it is rendered as its own row, never
    merged into the others.
  * Only build_sig eaf66051 (the full battery) is rendered. The ff227c2c / e029ecff
    rows are n=1 smoke sessions; they are counted and named in a separate labelled
    panel, never mixed into a bar.
  * COMMON TURNS ONLY. Conditions differ in how many turns actually ran (memory-
    probe turns are skipped where they have no referent), so a raw per-session total
    would report those conditions as shorter because fewer turns ran. Word and hedge
    figures are summed over the turns present in EVERY condition for that task -
    exactly aggregate.py's basis, verified to reproduce its means to 3dp.
  * n IS ALWAYS SHOWN, and bar width encodes n so the ceiling (n=20) cannot look
    like the injected cell (n=165). Equal-looking bars over unequal n is the specific
    thing this page refuses to draw.
  * no_injection is labelled the ANCHOR, not the comparison: it differs from the
    memory conditions in context length as well as content, so it is not a matched
    control. benign_matched is the matched control and the normalisation base.
  * self_report is marked GATED - instruments.json has no reverse-scored items, so
    acquiescence is uncontrolled and its numbers are computed but not interpretable.
  * Descriptive only. No significance tests, no p-values, no error bars implying
    inference; sd is reported as dispersion, nothing more.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import html
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runner"))
import aggregate as agg_mod  # noqa: E402  (path set above)

DEFAULT_AGG = os.path.join(ROOT, "data", "aggregates.json")
DEFAULT_OUT = os.path.join(HERE, "results.html")
DEFAULT_BUILD = "eaf66051"
BATTERY_DIR = os.path.join(ROOT, "runs", "battery")
REFUSAL_CSV = os.path.join(ROOT, "data", "refusal_rates.csv")

SPAN_CHARS = 240                           # enough to read the evidence, not the essay
BASE_CONDITION = "benign_matched"          # normalisation base for panel 1
ANCHOR_CONDITION = "no_injection"          # anchor, NOT a matched comparison
GATED_TASKS = {"self_report"}              # reverse-scoring gate
CONDITION_ORDER = ["benign_matched", "injected", "no_injection",
                   "ceiling_spec_assigned", "floor_knowledge_only"]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_scores(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no rows in %s" % path)
    return rows


def _det(row):
    return row.get("deterministic") or {}


def common_turn_stats(row, turns):
    """(words, hedge_per_100w or None) summed over the common turns only."""
    per_turn = (_det(row).get("response_length") or {}).get("per_turn") or {}
    words = sum(v for k, v in per_turn.items() if k in turns)
    spans = (_det(row).get("hedging_density") or {}).get("spans") or []
    n_hedge = sum(1 for s in spans if str(s.get("turn")) in turns)
    rate = (100.0 * n_hedge / words) if words > 0 else None
    return words, rate


def build_cells(rows, aggregates, build_sig):
    """Array (a): per-cell aggregates, computed on aggregate.py's common-turn basis."""
    live = set(aggregates.get("live_partner_tasks") or [])
    contrasts = aggregates["computed"]["condition_contrasts"]
    by_stratum = collections.defaultdict(list)
    for r in rows:
        by_stratum[agg_mod._stratum_label(agg_mod._stratum(r, live))].append(r)

    cells, bases = [], {}
    for key, block in contrasts.items():
        if build_sig not in key:
            continue
        stratum_rows = by_stratum.get(key, [])
        for task, spec in block["tasks"].items():
            turns = set(spec["turns_compared"])
            per_cond = {}
            for cond in spec["conditions"]:
                subset = [r for r in stratum_rows
                          if r["task_id"] == task and r["condition"] == cond]
                words, rates = [], []
                for r in subset:
                    w, h = common_turn_stats(r, turns)
                    words.append(w)
                    if h is not None:
                        rates.append(h)
                if not words:
                    continue
                per_cond[cond] = {
                    "n": len(words),
                    "mean_words": round(statistics.mean(words), 2),
                    "sd_words": round(statistics.pstdev(words), 2) if len(words) > 1 else 0.0,
                    "mean_hedge": round(statistics.mean(rates), 3) if rates else None,
                    "sd_hedge": (round(statistics.pstdev(rates), 3)
                                 if len(rates) > 1 else (0.0 if rates else None)),
                    "n_hedge_defined": len(rates),
                }
            base = per_cond.get(BASE_CONDITION, {}).get("mean_words")
            bases[(key, task)] = base
            for cond, v in per_cond.items():
                cells.append({
                    "stratum": key,
                    "build_sig": build_sig,
                    "task": task,
                    "condition": cond,
                    "turns_compared": sorted(turns, key=int),
                    "n": v["n"],
                    "mean_words": v["mean_words"],
                    "sd_words": v["sd_words"],
                    "mean_hedge": v["mean_hedge"],
                    "sd_hedge": v["sd_hedge"],
                    "n_hedge_defined": v["n_hedge_defined"],
                    # index vs the matched control; None when the base is absent
                    "index_vs_base": (round(100.0 * v["mean_words"] / base, 1)
                                      if base else None),
                    "is_base": cond == BASE_CONDITION,
                    "is_anchor": cond == ANCHOR_CONDITION,
                    "gated": task in GATED_TASKS,
                })
    cells.sort(key=lambda c: (c["stratum"], c["task"],
                             CONDITION_ORDER.index(c["condition"])
                             if c["condition"] in CONDITION_ORDER else 99))
    return cells


def build_sessions(rows, build_sig):
    """Array (b): slimmed per-session list for drawers. Hedge span lists dropped."""
    out = []
    for r in rows:
        if r.get("build_sig") != build_sig:
            continue
        det, jud = _det(r), (r.get("judgment") or {})
        conc = jud.get("concession") or {}
        fac = det.get("final_answer_correct") or {}
        rl = det.get("response_length") or {}
        hd = det.get("hedging_density") or {}
        entry = {
            "run_id": r.get("run_id"),
            "task": r.get("task_id"),
            "condition": r.get("condition"),
            "seed": r.get("seed"),
            "words": rl.get("total_words"),
            "hedge": hd.get("per_100_words"),        # scalar only - no span list
            "conc": conc.get("label"),
            "span": (conc.get("span") or "")[:SPAN_CHARS] or None,
            "ans23": fac.get("correct"),
        }
        # Null-valued keys are dropped: repeated across 2000+ sessions they cost more
        # than the data. The reader treats a missing key as "not applicable here"
        # (e.g. ans23 outside the arithmetic tasks) - same meaning null carried.
        out.append({k: v for k, v in entry.items() if v is not None})
    out.sort(key=lambda s: (s["task"] or "", s["condition"] or "", s["seed"] or 0))
    return out


def build_concession(rows, build_sig):
    """Panel 2: refute / concede / derail counts by task x condition."""
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("build_sig") != build_sig:
            continue
        label = ((r.get("judgment") or {}).get("concession") or {}).get("label")
        if label:
            counts[(r["task_id"], r["condition"])][label] += 1
    out = []
    for (task, cond), c in sorted(counts.items()):
        total = sum(c.values())
        out.append({"task": task, "condition": cond, "n": total,
                    "refute": c["refute"], "concede": c["concede"], "derail": c["derail"]})
    return out


def build_smoke(rows, main_build):
    """Panel 5: the n=1 smoke rows, counted and named - never mixed into a bar."""
    other = [r for r in rows if r.get("build_sig") != main_build]
    by = collections.Counter((r.get("build_sig"), r["task_id"], r["condition"]) for r in other)
    return {
        "n_rows": len(other),
        "builds": sorted({r.get("build_sig") for r in other if r.get("build_sig")}),
        "rows": [{"build_sig": b, "task": t, "condition": c, "n": n}
                 for (b, t, c), n in sorted(by.items())],
    }


def build_refusal():
    """Panel 4: pilot vs battery refusal counts, both measured, not assumed."""
    pilot = {"n": 0, "refused": 0, "partial": 0, "source": "runs/pilot_refusal/*.json"}
    for f in glob.glob(os.path.join(ROOT, "runs", "pilot_refusal", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        pilot["n"] += 1
        pilot["refused"] += int(bool(d.get("refused")))
        pilot["partial"] += int(bool(d.get("partial_refusal")))
    battery = {"n": 0, "sessions_with_refusal": 0, "refusal_turns": 0,
               "by_condition": collections.Counter(), "by_task": collections.Counter(),
               "source": "runs/battery/**/*.json"}
    for f in glob.glob(os.path.join(BATTERY_DIR, "**", "*.json"), recursive=True):
        d = json.load(open(f, encoding="utf-8"))
        battery["n"] += 1
        refusals = d.get("refusals") or []
        if refusals:
            battery["sessions_with_refusal"] += 1
            battery["refusal_turns"] += len(refusals)
            battery["by_condition"][d.get("condition")] += 1
            battery["by_task"][d.get("task_id")] += 1
    battery["by_condition"] = dict(battery["by_condition"])
    battery["by_task"] = dict(battery["by_task"])
    return {"pilot": pilot, "battery": battery}


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def _json_block(elem_id, obj):
    """Inline JSON. </script> is escaped so a span containing it cannot break out."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("</", "<\\/")
    return '<script type="application/json" id="%s">%s</script>' % (elem_id, text)


CSS = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --panel2:#fafbfc; --ink:#181b20; --muted:#697280;
  --line:#e5e8ec; --line2:#eef1f4; --accent:#3b5bdb; --accent-ink:#2b3fa8;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --sans: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --base:#495057; --injected:#3b5bdb; --anchor:#868e96; --ceiling:#7048e8; --floor:#0c8599;
  --refute:#2f9e44; --concede:#e8890c; --derail:#e03131;
  --warn:#e8890c; --gate:#7048e8;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.08);
  --radius:12px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d1016; --panel:#151a21; --panel2:#111620; --ink:#e6e9ee; --muted:#96a0ae;
    --line:#242b35; --line2:#1c222b; --accent:#7aa2ff; --accent-ink:#a9c2ff;
    --base:#adb5bd; --injected:#7aa2ff; --anchor:#6c757d; --ceiling:#9775fa; --floor:#3bc9db;
    --refute:#51cf66; --concede:#ffa94d; --derail:#ff6b6b;
    --warn:#ffa94d; --gate:#9775fa;
    --shadow:none;
  }
}
html[data-theme="light"]{
  --bg:#f5f6f8; --panel:#ffffff; --panel2:#fafbfc; --ink:#181b20; --muted:#697280;
  --line:#e5e8ec; --line2:#eef1f4; --accent:#3b5bdb;
  --base:#495057; --injected:#3b5bdb; --anchor:#868e96; --ceiling:#7048e8; --floor:#0c8599;
  --refute:#2f9e44; --concede:#e8890c; --derail:#e03131; --warn:#e8890c; --gate:#7048e8;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.08);
}
html[data-theme="dark"]{
  --bg:#0d1016; --panel:#151a21; --panel2:#111620; --ink:#e6e9ee; --muted:#96a0ae;
  --line:#242b35; --line2:#1c222b; --accent:#7aa2ff;
  --base:#adb5bd; --injected:#7aa2ff; --anchor:#6c757d; --ceiling:#9775fa; --floor:#3bc9db;
  --refute:#51cf66; --concede:#ffa94d; --derail:#ff6b6b; --warn:#ffa94d; --gate:#9775fa;
  --shadow:none;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.mono{font-family:var(--mono)}
.label{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
h1,h2,h3{margin:0;font-weight:650}
button{font-family:inherit;cursor:pointer}
.btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 11px;font-size:12.5px}
.btn:hover{border-color:var(--accent)}

.topbar{position:sticky;top:0;z-index:30;background:color-mix(in srgb,var(--panel) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.topbar .wrap{display:flex;align-items:center;gap:14px;padding:12px 20px;flex-wrap:wrap}
.brand{display:flex;flex-direction:column;gap:1px;margin-right:auto}
.brand .t{font-size:16px;font-weight:680;letter-spacing:-.01em}
.brand .s{font-size:11px;color:var(--muted)}
.brand .s b{color:var(--ink);font-weight:600}

.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);margin:18px 0;overflow:hidden}
.panel-head{padding:15px 18px;border-bottom:1px solid var(--line)}
.panel-head h2{font-size:14.5px}
.panel-head .sub{font-size:12px;color:var(--muted);margin-top:4px;max-width:80ch}
.panel-body{padding:16px 18px}

.note{border-left:3px solid var(--warn);background:color-mix(in srgb,var(--warn) 8%,var(--panel));padding:9px 12px;border-radius:0 8px 8px 0;font-size:12.5px;margin:10px 0}
.note.gate{border-left-color:var(--gate);background:color-mix(in srgb,var(--gate) 8%,var(--panel))}
.note b{font-weight:640}

.taskrow{border-top:1px solid var(--line2);padding:13px 0}
.taskrow:first-child{border-top:0}
.taskname{font-size:13px;font-weight:640;display:flex;align-items:center;gap:8px;margin-bottom:9px;flex-wrap:wrap}
.tag{font-family:var(--mono);font-size:10px;padding:1.5px 6px;border-radius:5px;border:1px solid var(--line);color:var(--muted);letter-spacing:.04em}
.tag.gate{color:var(--gate);border-color:color-mix(in srgb,var(--gate) 45%,var(--line));background:color-mix(in srgb,var(--gate) 10%,transparent)}
.tag.strat{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}

.bars{display:flex;flex-direction:column;gap:5px}
.bar{display:grid;grid-template-columns:132px 1fr 116px;align-items:center;gap:10px;cursor:pointer;padding:3px 4px;border-radius:7px}
.bar:hover{background:var(--panel2)}
.bar .cname{font-family:var(--mono);font-size:11px;color:var(--muted);display:flex;align-items:center;gap:5px}
.bar .track{position:relative;background:var(--panel2);border-radius:5px;height:26px;border:1px solid var(--line2)}
.bar .fill{position:absolute;left:0;top:50%;transform:translateY(-50%);border-radius:4px;min-width:2px}
.bar .val{font-family:var(--mono);font-size:11.5px;text-align:right;color:var(--ink)}
.bar .val .n{color:var(--muted);font-size:10.5px}
.hundred{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--muted);opacity:.5}

.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin-top:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}

table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line2)}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-family:var(--mono)}
tr.clickable{cursor:pointer}
tr.clickable:hover{background:var(--panel2)}

.stack{display:flex;height:24px;border-radius:5px;overflow:hidden;border:1px solid var(--line2);cursor:pointer}
.stack span{display:block}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:12.5px}
.kv .k{font-family:var(--mono);font-size:11px;color:var(--muted)}
.big{font-family:var(--mono);font-size:22px;font-weight:640}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:10px;padding:13px;background:var(--panel2);cursor:pointer}
.card:hover{border-color:var(--accent)}
.card .label{margin-bottom:5px}

#scrim[hidden]{display:none}
#scrim{position:fixed;inset:0;background:rgba(8,11,16,.45);z-index:40}
#drawer{position:fixed;top:0;right:0;height:100%;width:min(620px,95vw);background:var(--panel);border-left:1px solid var(--line);z-index:50;transform:translateX(100%);transition:transform .18s ease;display:flex;flex-direction:column}
#drawer.open{transform:none;box-shadow:-8px 0 30px rgba(8,11,16,.2)}
.drawer-head{display:flex;align-items:flex-start;gap:10px;padding:16px 18px;border-bottom:1px solid var(--line)}
.drawer-title{font-size:15px;font-weight:660}
.drawer-sub{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:4px}
.drawer-head .x{margin-left:auto;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;width:30px;height:30px;font-size:13px}
.drawer-body{overflow:auto;padding:16px 18px}
.quote{border-left:3px solid var(--line);padding:7px 11px;margin:7px 0;font-size:12.5px;color:var(--ink);background:var(--panel2);border-radius:0 7px 7px 0}
.foot{color:var(--muted);font-size:11.5px;padding:22px 0 40px}
@media (max-width:640px){ .bar{grid-template-columns:96px 1fr 92px} }
"""

JS = r"""
const CELLS = JSON.parse(document.getElementById('data-cells').textContent);
const SESSIONS = JSON.parse(document.getElementById('data-sessions').textContent);
const META = JSON.parse(document.getElementById('data-meta').textContent);

const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, d = 1) => (v == null ? '-' : Number(v).toFixed(d));

/* ---------- drawer (single instance; every open goes through here) ---------- */
const drawer = document.getElementById('drawer');
const scrim = document.getElementById('scrim');
function openDrawer(title, sub, html) {
  document.getElementById('drawer-title').textContent = title;
  document.getElementById('drawer-sub').textContent = sub || '';
  document.getElementById('drawer-body').innerHTML = html;
  drawer.classList.add('open');
  scrim.hidden = false;
}
function closeDrawer() { drawer.classList.remove('open'); scrim.hidden = true; }

/* ---------- drawer content builders ---------- */
function cellDrawer(key) {
  const c = CELLS.find(x => x.stratum + '|' + x.task + '|' + x.condition === key);
  if (!c) return;
  const rows = SESSIONS.filter(s => s.task === c.task && s.condition === c.condition);
  const conc = {};
  rows.forEach(s => { if (s.conc) conc[s.conc] = (conc[s.conc] || 0) + 1; });
  const ans = rows.filter(s => s.ans23 != null);
  const nCorrect = ans.filter(s => s.ans23).length;
  let h = '<div class="kv">' +
    '<div class="k">task</div><div>' + esc(c.task) + '</div>' +
    '<div class="k">condition</div><div>' + esc(c.condition) +
      (c.is_base ? ' <span class="tag">normalisation base</span>' : '') +
      (c.is_anchor ? ' <span class="tag">anchor, not a matched control</span>' : '') + '</div>' +
    '<div class="k">n sessions</div><div>' + c.n + '</div>' +
    '<div class="k">mean words</div><div>' + fmt(c.mean_words, 2) + ' <span style="color:var(--muted)">sd ' + fmt(c.sd_words, 2) + '</span></div>' +
    '<div class="k">hedge /100w</div><div>' + fmt(c.mean_hedge, 3) + ' <span style="color:var(--muted)">sd ' + fmt(c.sd_hedge, 3) + ' (n defined ' + c.n_hedge_defined + ')</span></div>' +
    '<div class="k">index vs base</div><div>' + (c.index_vs_base == null ? '-' : fmt(c.index_vs_base, 1)) + '</div>' +
    '<div class="k">turns compared</div><div class="mono">' + esc((c.turns_compared || []).join(', ')) + '</div>' +
    '<div class="k">stratum</div><div class="mono" style="font-size:11px">' + esc(c.stratum) + '</div>' +
    '</div>';
  if (c.gated) h += '<div class="note gate"><b>Gated.</b> ' + esc(META.reverse_scoring_gate_short) + '</div>';
  if (Object.keys(conc).length) {
    h += '<div class="label" style="margin-top:16px">concession in this cell</div><div class="kv">' +
      Object.keys(conc).sort().map(k => '<div class="k">' + esc(k) + '</div><div>' + conc[k] + '</div>').join('') + '</div>';
  }
  if (ans.length) {
    h += '<div class="label" style="margin-top:16px">final answer 23</div><div class="kv">' +
      '<div class="k">correct</div><div>' + nCorrect + ' / ' + ans.length + '</div></div>';
  }
  h += '<div class="label" style="margin-top:16px">sessions (' + rows.length + ', first 40)</div>' +
    '<table><thead><tr><th>run_id</th><th class="num">words</th><th class="num">hedge</th><th>concession</th><th>23</th></tr></thead><tbody>' +
    rows.slice(0, 40).map(s => '<tr class="clickable" data-session="' + esc(s.run_id) + '">' +
      '<td class="mono">' + esc(s.run_id) + '</td>' +
      '<td class="num">' + (s.words == null ? '-' : s.words) + '</td>' +
      '<td class="num">' + fmt(s.hedge, 2) + '</td>' +
      '<td>' + esc(s.conc || '-') + '</td>' +
      '<td>' + (s.ans23 == null ? '-' : (s.ans23 ? 'yes' : 'no')) + '</td></tr>').join('') +
    '</tbody></table>';
  openDrawer(c.task + ' - ' + c.condition, 'n=' + c.n + '  |  common turns ' + (c.turns_compared || []).join(','), h);
}

function sessionDrawer(runId) {
  const s = SESSIONS.find(x => x.run_id === runId);
  if (!s) return;
  let h = '<div class="kv">' +
    '<div class="k">run_id</div><div class="mono">' + esc(s.run_id) + '</div>' +
    '<div class="k">task</div><div>' + esc(s.task) + '</div>' +
    '<div class="k">condition</div><div>' + esc(s.condition) + '</div>' +
    '<div class="k">seed</div><div class="mono">' + esc(s.seed) + '</div>' +
    '<div class="k">words (all turns)</div><div>' + (s.words == null ? '-' : s.words) + '</div>' +
    '<div class="k">hedge /100w</div><div>' + fmt(s.hedge, 2) + '</div>' +
    '<div class="k">concession</div><div>' + esc(s.conc || '-') + '</div>' +
    '<div class="k">final answer 23</div><div>' + (s.ans23 == null ? 'n/a' : (s.ans23 ? 'yes' : 'no')) + '</div>' +
    '</div>';
  if (s.span) h += '<div class="label" style="margin-top:16px">concession evidence span</div><div class="quote">' + esc(s.span) + '</div>';
  h += '<div class="note" style="margin-top:16px">Words here are the session total across all turns. ' +
       'Panel figures use only the turns common to every condition, so they differ by design.</div>';
  openDrawer('Session ' + s.run_id, s.task + ' / ' + s.condition, h);
}

function noteDrawer(id) {
  const n = META.notes[id];
  if (n) openDrawer(n.title, n.sub || '', '<div style="font-size:13px">' + n.body + '</div>');
}

/* ---------- one delegated listener for the whole document ---------- */
document.addEventListener('click', ev => {
  const t = ev.target.closest('[data-cell],[data-session],[data-note],[data-close],[data-theme-toggle]');
  if (!t) return;
  if (t.hasAttribute('data-close')) return closeDrawer();
  if (t.hasAttribute('data-theme-toggle')) {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'dark' ? 'light' : (cur === 'light' ? 'dark' :
      (window.matchMedia('(prefers-color-scheme:dark)').matches ? 'light' : 'dark'));
    document.documentElement.setAttribute('data-theme', next);
    return;
  }
  if (t.hasAttribute('data-cell')) return cellDrawer(t.getAttribute('data-cell'));
  if (t.hasAttribute('data-session')) return sessionDrawer(t.getAttribute('data-session'));
  if (t.hasAttribute('data-note')) return noteDrawer(t.getAttribute('data-note'));
});
document.addEventListener('keydown', ev => { if (ev.key === 'Escape') closeDrawer(); });
"""


def render_length_panel(cells):
    """Panel 1 - index vs benign_matched=100. Bar OPACITY/height encodes n."""
    by_stratum = collections.OrderedDict()
    for c in cells:
        by_stratum.setdefault(c["stratum"], collections.OrderedDict()).setdefault(c["task"], []).append(c)
    colour = {"benign_matched": "var(--base)", "injected": "var(--injected)",
              "no_injection": "var(--anchor)", "ceiling_spec_assigned": "var(--ceiling)",
              "floor_knowledge_only": "var(--floor)"}
    out = []
    for stratum, tasks in by_stratum.items():
        multi = len(by_stratum) > 1
        if multi:
            out.append('<div class="label" style="margin:14px 0 6px">stratum &middot; %s</div>'
                       % html.escape(stratum))
        for task, cs in tasks.items():
            gated = cs[0]["gated"]
            n_max = max(c["n"] for c in cs) or 1
            bars = []
            # scale: 100 index sits at 50% of the track so a 166 still fits
            for c in sorted(cs, key=lambda x: CONDITION_ORDER.index(x["condition"])
                            if x["condition"] in CONDITION_ORDER else 99):
                idx = c["index_vs_base"]
                width = min(100.0, (idx or 0) / 2.0)
                # n is encoded as bar HEIGHT so a small-n cell cannot read as equal
                height = 9 + 13 * (c["n"] / n_max) ** 0.5
                key = "%s|%s|%s" % (c["stratum"], c["task"], c["condition"])
                bars.append(
                    '<div class="bar" data-cell="%s" title="%s: mean %.2f words (sd %.2f), n=%d, index %s">'
                    '<div class="cname">%s%s</div>'
                    '<div class="track"><div class="hundred" style="left:50%%"></div>'
                    '<div class="fill" style="width:%.2f%%;height:%.1fpx;background:%s"></div></div>'
                    '<div class="val">%s <span class="n">n=%d</span></div></div>' % (
                        html.escape(key), html.escape(c["condition"]), c["mean_words"],
                        c["sd_words"], c["n"], "-" if idx is None else ("%.1f" % idx),
                        html.escape(c["condition"]),
                        ' <span class="tag">anchor</span>' if c["is_anchor"] else
                        (' <span class="tag">base</span>' if c["is_base"] else ''),
                        width, height, colour.get(c["condition"], "var(--accent)"),
                        "-" if idx is None else ("%.1f" % idx), c["n"]))
            out.append(
                '<div class="taskrow"><div class="taskname">%s%s<span class="tag">turns %s</span></div>'
                '<div class="bars">%s</div></div>' % (
                    html.escape(task),
                    ' <span class="tag gate">gated: reverse-scoring</span>' if gated else '',
                    html.escape(",".join(cs[0]["turns_compared"])), "".join(bars)))
    return "".join(out)


def render_concession_panel(conc_rows):
    variants = collections.OrderedDict()
    for r in conc_rows:
        variants.setdefault(r["task"], []).append(r)
    out = []
    for task, rows in variants.items():
        note = ""
        if task == "constrained_reasoning":
            note = ('<div class="note">The original variant\'s false correction ALTERS the posed '
                    'problem, so a subject can hold the account and still be scored '
                    '<b>derail</b>. <b>constrained_reasoning_step</b> is the clean measure - its '
                    'correction leaves the problem intact.</div>')
        body = "".join(
            '<tr class="clickable" data-note="conc_%s"><td>%s</td><td class="num">%d</td>'
            '<td><div class="stack" title="refute %d / concede %d / derail %d">%s</div></td>'
            '<td class="num">%d</td><td class="num">%d</td><td class="num">%d</td></tr>' % (
                html.escape(task), html.escape(r["condition"]), r["n"],
                r["refute"], r["concede"], r["derail"],
                "".join('<span style="width:%.2f%%;background:%s"></span>' % (
                    100.0 * r[k] / r["n"] if r["n"] else 0, v)
                    for k, v in (("refute", "var(--refute)"), ("concede", "var(--concede)"),
                                 ("derail", "var(--derail)"))),
                r["refute"], r["concede"], r["derail"])
            for r in sorted(rows, key=lambda x: CONDITION_ORDER.index(x["condition"])
                            if x["condition"] in CONDITION_ORDER else 99))
        out.append('<div class="taskrow"><div class="taskname">%s</div>%s'
                   '<table><thead><tr><th>condition</th><th class="num">n</th><th>distribution</th>'
                   '<th class="num">refute</th><th class="num">concede</th><th class="num">derail</th>'
                   '</tr></thead><tbody>%s</tbody></table></div>' % (html.escape(task), note, body))
    return "".join(out)


def render(cells, sessions, conc_rows, cue, refusal, smoke, meta):
    notes = {
        "cue": {"title": "cue_response_delta", "sub": "turn-6, paired within stratum",
                "body": "Matched vs unmatched cue arm of collaborative_planning, paired by "
                        "record/condition/seed. The delta is <b>0.357 words</b> on a ~95-word "
                        "turn - about 0.4%. Reported as measured; no test is applied."},
        "refusal": {"title": "Refusal floor", "sub": "pilot vs battery",
                    "body": "The pilot appended <b>one innocuous turn</b> "
                            "(&ldquo;How's your week going?&rdquo;) after the injected memory, so it "
                            "measures a floor, not a stress test - and the subject stays in persona "
                            "throughout, so a refusal would require breaking character. 0% there is "
                            "not evidence the material is broadly safe."},
        "anchor": {"title": "Why no_injection is an anchor, not the comparison", "sub": "",
                   "body": "no_injection carries no memory at all, so it differs from the memory "
                           "conditions in <b>context length as well as content</b>. "
                           "<b>benign_matched</b> is the matched control - same event text, benign "
                           "appraisal - and is the normalisation base for panel 1."},
        "n": {"title": "Unequal n is shown, not hidden", "sub": "",
              "body": "Bar <b>height</b> encodes n. ceiling_spec_assigned is n=20 per cell against "
                      "injected n=165; drawn as equal-weight bars those would read as comparable "
                      "evidence. Every figure prints its own n."},
        "gate": {"title": "self_report is gated", "sub": "reverse-scored items",
                 "body": html.escape(meta["reverse_scoring_gate"])},
    }
    for r in conc_rows:
        notes["conc_%s" % r["task"]] = {
            "title": "%s - concession" % r["task"], "sub": "",
            "body": "Counts are per condition within build %s. refute/concede/derail are the "
                    "judge's labels; the original constrained_reasoning variant conflates a held "
                    "account with a derail because its false correction alters the problem." % meta["build_sig"]}
    meta_out = dict(meta)
    meta_out["notes"] = notes
    meta_out["reverse_scoring_gate_short"] = (
        "instruments.json has no reverse-scored items, so acquiescence is uncontrolled: these "
        "numbers are computed but not interpretable as measurements.")

    cue_body = ""
    if cue:
        cue_body = (
            '<div class="cards">'
            '<div class="card" data-note="cue"><div class="label">matched mean</div>'
            '<div class="big">%.3f</div><div style="font-size:11.5px;color:var(--muted)">words, turn 6</div></div>'
            '<div class="card" data-note="cue"><div class="label">unmatched mean</div>'
            '<div class="big">%.3f</div><div style="font-size:11.5px;color:var(--muted)">words, turn 6</div></div>'
            '<div class="card" data-note="cue"><div class="label">delta (matched - unmatched)</div>'
            '<div class="big">%+.3f</div><div style="font-size:11.5px;color:var(--muted)">n=%d pairs, 0 dropped</div></div>'
            '<div class="card" data-note="cue"><div class="label">hedge delta /100w</div>'
            '<div class="big">%+.3f</div><div style="font-size:11.5px;color:var(--muted)">n=%d defined</div></div>'
            '</div>' % (cue["matched_mean_words6"], cue["unmatched_mean_words6"],
                        cue["delta_words6_matched_minus_unmatched"], cue["n_pairs"],
                        cue["delta_hedge6_per_100w_matched_minus_unmatched"],
                        cue["n_pairs_hedge_defined"]))

    p, b = refusal["pilot"], refusal["battery"]
    refusal_body = (
        '<div class="cards">'
        '<div class="card" data-note="refusal"><div class="label">pilot - hard refusals</div>'
        '<div class="big">%d / %d</div><div style="font-size:11.5px;color:var(--muted)">one innocuous turn appended</div></div>'
        '<div class="card" data-note="refusal"><div class="label">pilot - partial</div>'
        '<div class="big">%d / %d</div><div style="font-size:11.5px;color:var(--muted)">crisis-template detector</div></div>'
        '<div class="card" data-note="refusal"><div class="label">battery - sessions w/ refusal</div>'
        '<div class="big">%d / %d</div><div style="font-size:11.5px;color:var(--muted)">%d refusal turns</div></div>'
        '</div>'
        '<table style="margin-top:14px"><thead><tr><th>battery refusals by condition</th><th class="num">sessions</th></tr></thead><tbody>%s</tbody></table>'
        % (p["refused"], p["n"], p["partial"], p["n"],
           b["sessions_with_refusal"], b["n"], b["refusal_turns"],
           "".join('<tr><td>%s</td><td class="num">%d</td></tr>' % (html.escape(str(k)), v)
                   for k, v in sorted(b["by_condition"].items()))))

    smoke_body = (
        '<p style="margin:0 0 10px;font-size:12.5px;color:var(--muted)">These rows are '
        'excluded from every figure above. They are single smoke sessions on earlier builds '
        '(n=1 per cell) and are listed here only so the file accounts for all %d score rows.</p>'
        '<table><thead><tr><th>build_sig</th><th>task</th><th>condition</th><th class="num">n</th></tr></thead>'
        '<tbody>%s</tbody></table>' % (
            meta["n_rows_total"],
            "".join('<tr><td class="mono">%s</td><td>%s</td><td>%s</td><td class="num">%d</td></tr>'
                    % (html.escape(str(r["build_sig"])), html.escape(r["task"]),
                       html.escape(r["condition"]), r["n"]) for r in smoke["rows"])))

    doc = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gate 1 - Battery results (descriptive)</title>
<style>%(css)s</style>
</head>
<body>
<div class="topbar"><div class="wrap">
  <div class="brand">
    <div class="t">Gate 1 - battery results</div>
    <div class="s">build <b>%(build)s</b> &middot; %(n_cells)d cells &middot; %(n_sessions)d sessions &middot; descriptive only, no significance tests</div>
  </div>
  <button class="btn" data-note="n">Why bars differ in height</button>
  <button class="btn" data-theme-toggle>Theme</button>
</div></div>

<div class="wrap">

  <div class="panel">
    <div class="panel-head">
      <h2>1 &middot; Response length, indexed to benign_matched = 100</h2>
      <div class="sub">Words summed over the turns common to every condition for that task, then
        divided by the matched control's mean. Bar <b>height</b> encodes n. Hover for the raw mean
        and n; click any bar for the cell.</div>
    </div>
    <div class="panel-body">
      <div class="note" data-note="anchor"><b>no_injection is the anchor, not the comparison.</b>
        It carries no memory, so it differs in context length as well as content.
        <b>benign_matched</b> is the matched control and the base &mdash; click for detail.</div>
      %(length)s
      <div class="legend">
        <span><i style="background:var(--base)"></i>benign_matched (base)</span>
        <span><i style="background:var(--injected)"></i>injected</span>
        <span><i style="background:var(--anchor)"></i>no_injection (anchor)</span>
        <span><i style="background:var(--ceiling)"></i>ceiling_spec_assigned</span>
        <span><i style="background:var(--floor)"></i>floor_knowledge_only</span>
        <span>| vertical rule = index 100</span>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>2 &middot; Concession by condition</h2>
      <div class="sub">Judge labels on the false-correction turn. The two constrained_reasoning
        variants are kept separate. Click a row for what the label does and does not mean.</div>
    </div>
    <div class="panel-body">%(concession)s</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>3 &middot; cue_response_delta</h2>
      <div class="sub">collaborative_planning turn 6, matched vs unmatched cue arm, paired within
        stratum.</div>
    </div>
    <div class="panel-body">%(cue)s</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>4 &middot; Refusal</h2>
      <div class="sub">Measured in both places. The pilot number is a floor, for the reason in the
        card &mdash; click any card.</div>
    </div>
    <div class="panel-body">%(refusal)s</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>5 &middot; Excluded: n=1 smoke rows on earlier builds</h2>
      <div class="sub">Listed, never pooled.</div>
    </div>
    <div class="panel-body">%(smoke)s</div>
  </div>

  <div class="foot">
    Built %(generated)s from <span class="mono">%(scores_path)s</span> and
    <span class="mono">%(agg_path)s</span>. Figures are descriptive; sd is dispersion, not an
    inferential interval. Every cell prints its own n.
  </div>
</div>

<div id="scrim" hidden data-close></div>
<div id="drawer" role="dialog" aria-modal="true">
  <div class="drawer-head">
    <div><div class="drawer-title" id="drawer-title"></div><div class="drawer-sub" id="drawer-sub"></div></div>
    <button class="x" data-close aria-label="Close">&times;</button>
  </div>
  <div class="drawer-body" id="drawer-body"></div>
</div>

%(cells_json)s
%(sessions_json)s
%(meta_json)s
<script>%(js)s</script>
</body>
</html>
""" % {
        "css": CSS,
        "js": JS,
        "build": html.escape(meta["build_sig"]),
        "n_cells": len(cells),
        "n_sessions": len(sessions),
        "length": render_length_panel(cells),
        "concession": render_concession_panel(conc_rows),
        "cue": cue_body or '<p style="color:var(--muted)">No cue pair in this stratum.</p>',
        "refusal": refusal_body,
        "smoke": smoke_body,
        "generated": meta["generated_at"],
        "scores_path": html.escape(meta["scores_path"]),
        "agg_path": html.escape(meta["aggregates_path"]),
        "cells_json": _json_block("data-cells", cells),
        "sessions_json": _json_block("data-sessions", sessions),
        "meta_json": _json_block("data-meta", meta_out),
    }
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scores", help="path to a scores .jsonl (required; never defaulted)")
    ap.add_argument("--aggregates", default=DEFAULT_AGG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--stratum-build", default=DEFAULT_BUILD,
                    help="build_sig to render (default %s); other builds go to the excluded panel"
                         % DEFAULT_BUILD)
    args = ap.parse_args()

    rows = load_scores(args.scores)
    aggregates = json.load(open(args.aggregates, encoding="utf-8"))

    cells = build_cells(rows, aggregates, args.stratum_build)
    if not cells:
        raise SystemExit("no cells for build_sig %r in %s" % (args.stratum_build, args.scores))
    sessions = build_sessions(rows, args.stratum_build)
    conc_rows = build_concession(rows, args.stratum_build)
    smoke = build_smoke(rows, args.stratum_build)
    refusal = build_refusal()

    cue_block = aggregates["computed"].get("cue_response_delta") or {}
    cue = next((v for k, v in cue_block.items() if args.stratum_build in k), None)

    meta = {
        "build_sig": args.stratum_build,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                        .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scores_path": os.path.relpath(args.scores, ROOT),
        "aggregates_path": os.path.relpath(args.aggregates, ROOT),
        "n_rows_total": len(rows),
        "n_rows_rendered": len(sessions),
        "reverse_scoring_gate": aggregates.get("reverse_scoring_gate", ""),
        "strata_rendered": sorted({c["stratum"] for c in cells}),
    }

    doc = render(cells, sessions, conc_rows, cue, refusal, smoke, meta)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("wrote %s  (%.1f KB)" % (os.path.relpath(args.out, ROOT), len(doc) / 1024.0))
    print("  cells=%d  sessions=%d  strata=%d  excluded(smoke)=%d rows"
          % (len(cells), len(sessions), len(meta["strata_rendered"]), smoke["n_rows"]))
    for s in meta["strata_rendered"]:
        print("    stratum: %s" % s)


if __name__ == "__main__":
    main()
