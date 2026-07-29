# Changelog

Work done on the Gate 1 pipeline. Newest first. Each entry records what changed, why,
and what it costs to adopt — because several entries change the meaning of a stored
number, and one of them is deliberately *not* applied to existing rows.

Unless stated otherwise, changes are code-only: no transcripts were regenerated and no
scoring model was called.

---

## 2026-07-29

### Added — `analysis/build_results.py` and `analysis/results.html`

A single self-contained results page, built the same way as `dist/gate1.html`: one file,
data inlined as `<script type="application/json">`, no external requests (verified: zero
`http(s)://`, zero external `src`/`href`), CSS variables plus a `prefers-color-scheme`
block and a `data-theme` override, one delegated `click` listener for the whole document,
and click-anything-opens-a-drawer.

```bash
python analysis/build_results.py data/scores_2026-07-29_finalrun.jsonl
```

The scores path is a **required argument** — the live `data/scores.jsonl` is never
hardcoded, so a rebuild cannot silently pick up a half-scored working file.
`--aggregates`, `--out` and `--stratum-build` are optional.

Two embedded arrays, not one:

| block | contents | size |
|---|---|---|
| `data-cells` | per-cell aggregates: task × condition × {n, mean/sd words, mean/sd hedge per 100w, index vs base, turns compared} | 9.9 KB |
| `data-sessions` | slimmed per-session list: run_id, task, condition, seed, words, hedge scalar, concession label + span (240 chars), ans23 | 421 KB |

Full hedge span lists are dropped — nothing on the page reads them and they were ~20× the
rest. Null-valued keys are omitted rather than stored 2180 times.

Panels: (1) length indexed to `benign_matched` = 100; (2) concession stacked
refute/concede/derail, both CR variants separate; (3) `cue_response_delta`; (4) refusal,
pilot vs battery; (5) the excluded n=1 smoke rows, listed but never pooled.

Method constraints enforced by the builder, not left to the reader:

- **Stratify, never pool.** Cells are computed within one experiment stratum using
  `aggregate.py`'s own `_stratum`, so the split matches `data/aggregates.json` exactly.
  `clinical_interview` lands in a *different* stratum from every other task (it is the
  only task with a live partner) and is rendered as its own labelled group.
- **Common turns only.** Conditions run different numbers of turns, so a raw per-session
  total would report some conditions as shorter merely because fewer turns ran. Words and
  hedge rates are summed over the turns present in every condition for that task — the
  same basis `aggregate.py` uses. Verified: all 29 rendered cells reproduce
  `aggregates.json` means to 3dp with 0 mismatches.
- **Unequal n is visible.** Bar *height* encodes n, so `ceiling_spec_assigned` (n=20)
  cannot read as equal evidence to `injected` (n=165). Every cell prints its own n.
- **`no_injection` is labelled the anchor, not the comparison** — it carries no memory, so
  it differs in context length as well as content. `benign_matched` is the matched control
  and the normalisation base.
- **`self_report` is tagged gated** (reverse-scoring, see below).
- Descriptive only: no significance tests, no p-values; sd is dispersion, nothing more.

Rendered result, for the record: injected indexes **98.6–101.5** across tasks (self_report
108.1), ceiling **132.2–165.5**.

Two data notes surfaced while building:

- `docs/filter_1_.html` does not exist in the repo (`docs/` holds only `gate1-spec-en.md`);
  `dist/gate1.html` was used as the construction reference.
- Battery refusals are **8 / 2180**, not 9 — 8 sessions and 8 refusal turns, counted both
  ways. The page renders 8.

### Added — `runner/run.py`: deepen one arm without re-running the grid

- `--cells main,anchor,floor,ceiling` restricts a run to a subset of cells.
- `GATE1_CEILING_SEEDS` gives the ceiling arm its own seed list (defaults to
  `SEEDS_ANCHOR`, so leaving it unset changes nothing).
- `_seeds()` now accepts ranges and mixes — `1-100`, `1-20,25,30-35` — preserving order and
  dropping duplicates. A 100-seed arm was previously unusable as a comma list.

```bash
GATE1_CEILING_SEEDS=1-100 python runner/run.py --cells ceiling --dry-run
GATE1_CEILING_SEEDS=1-100 python runner/run.py --cells ceiling
```

Restricting cells changes only *which* specs are yielded, never how one is built, so
`run_id`s are byte-identical to the full plan's. Verified: at 100 seeds the ceiling arm
plans 700 sessions, of which the **140 existing run_ids are unchanged and skip**, leaving
**560** to execute. That makes this a resumable extension of the existing run, not a new
experiment.

Also fixed while there: `main()` built the plan without passing `cells` (so `--cells` would
have restricted nothing), and the dry-run header reported ceiling seeds as the anchor count.

Consequence to keep in view: deepening one arm makes the grid unbalanced (ceiling n=100 vs
injected n=165 per cell). That is a reporting obligation — every figure must show its own n
— not a reason to avoid it. `results.html` already encodes n in bar height.

### Added — `runner/score.py`: PCL-5 / PHQ-9 / GAD-7 item vectors and B/C/D/E clusters

`self_report.score()` already produced an item vector, a DSM-5 cluster profile, reverse-item
inversion and scale validation. The missing step was recovering numbers from the subject's
free text, so `aggregate.py`'s `discriminant_pattern` and `knowledge_floor` had no input at
all. New `instrument_vectors()` fills exactly that gap and is wired into `score_session`,
emitting `deterministic.instrument_vectors` for self_report sessions.

Two response shapes occur in the data and both are parsed — enumerated (`1. …maybe a two?`)
and prose (`repeated memories? …I'd say a 2`) — with prose matched *sequentially* so a later
item's number cannot attach to an earlier item. Enumerated wins where both fire.

Honesty rules built in: a vector is emitted only when every item is recovered (partial parses
report coverage and the per-item map, `complete: false`); out-of-scale values are discarded,
not clamped; `"two or three"` is recorded as ambiguous with the first value taken; and
`interpretable` is **false** while `instruments.json` has no reverse-scored items.

Measured coverage over the 295 battery self_report sessions:

| instrument | complete vectors | mean item coverage |
|---|---|---|
| PCL-5 (20 items) | 75 / 295 (25%) | 36.5% |
| PHQ-9 (9 items) | 93 / 295 (32%) | 62.4% |
| GAD-7 (7 items) | 212 / 295 (72%) | 80.4% |

The PCL-5 ceiling is the **data, not the parser**: only 28% of PCL-5 replies contain ≥20
numeric mentions at all (median 6 for a 20-item instrument) — subjects rate a handful of
items and trail off. So building the parser unblocks the *computation* but does not yield
complete vectors at full n, and the reverse-scoring gate still blocks *interpretation*.

**Not yet applied to stored rows.** Deterministic metrics are only computed on a fresh
`score_session`, and existing self_report rows are already `scoring_complete`, so they skip.
`--force` would recompute them but also re-runs every paid judgment, and
`--force --deterministic-only` would set `judgment=None` and destroy ~1000 stored judgments.
A safe in-place `--redo-deterministic` path was designed but **not implemented** — until it
exists, `instrument_vectors` appears only on newly scored rows. This is the one open item.

### Fixed — `runner/score.py`: LaTeX in a judge span made the payload unparseable

On the arithmetic tasks the subject writes LaTeX (`\(a\)`, `\(9 + 9 + 4\)`); the judge copies
the span verbatim as instructed; and `\(` is not a legal JSON escape, so `json.loads` rejects
the entire object. Deterministic at temperature 0, so both validation attempts burn and the
row sticks permanently.

`_extract_json` now tries strict parsing first (well-formed payloads are never rewritten),
then falls back to `_repair_json_escapes`, which doubles any backslash that is not a legal
JSON escape. Escaping rather than stripping keeps the span byte-identical to the subject's
text, so it still satisfies verbatim grounding.

Scale: **328 of 2180** battery transcripts (15%) contain LaTeX in the subject text — all on
the two constrained tasks, ≈59% of the concession-scored rows. Only one row actually stuck,
because it depends on which span the judge happens to cite, so this was latent across most of
the reasoning battery. Regression-guarded against valid escapes (`\n \" \\ \t \/`), fenced and
prose-wrapped payloads, and genuinely malformed input.

### Fixed — `runner/score.py`: 16 rows could never complete (punctuation grounding)

`_spans_grounded` requires each cited span to appear verbatim in a speaker-B turn (an
anti-fabrication check) and compared through `_norm`, which folded whitespace and case only.
gpt-4o-mini writes curly `’`, `…`, en/em dashes; the gpt-4o judge normalises them to ASCII
when copying "verbatim". The span was otherwise character-perfect, but the substring test
failed on quote style alone — and at temperature 0 it failed identically on every retry.

`_norm` now folds curly quotes/primes, dashes, `…` and NBSP to ASCII, applied *symmetrically*
to both sides so it cannot loosen the check one-way. Grounding becomes "verbatim modulo
cosmetic punctuation", which is what the check was always trying to express. Guard-tested:
invented spans and paraphrases (`can't` → `cannot`) are still rejected.

All 13 judge spans still visible in the recorded error payloads ground under the fixed
`_norm`; none still fail.

### Added — `runner/score.py`: structured `judge_error` on failed rows

Failures were recorded only as a nested prose string at `judgment.error`, repr-escaped and
truncated to 200 chars, with no per-metric cause — so "permanently broken" and "the API
blipped" looked the same and had to be told apart by eye.

Failed rows now carry a top-level `judge_error`:

```json
{"message": "...", "attempts": 2, "retryable": false, "stage": "validation",
 "detail": [{"metric": "affect_vs_action", "reason": "span_not_grounded",
             "n_ungrounded": 1, "spans": ["..."]}],
 "last_raw": "<2000 chars>", "metrics": [...], "failed_at": "..."}
```

`stage`/`retryable` separate a transport failure (HTTP 429/5xx — a rerun may help) from a
deterministic validation reject (it will not). `detail` reasons: `unparseable_json`,
`not_an_object`, `missing_field`, `failed_type_or_range`, `span_not_grounded`,
`validated_on_reinspection`. Diagnostics are best-effort and never raise. `None` on success.
Console prints `JUDGE_ERROR[validation x2 span_not_grounded]` instead of a bare marker.

Also renamed the retry constant to `JUDGE_VALIDATION_ATTEMPTS` (validation attempts within
one `run_judge`), which was easy to confuse with `SCORER_MAX_ATTEMPTS` (transport retries
within one attempt).

`SCORE_ROW_VERSION` bumped `row-2` → `row-3`. `SCORER_PROMPT_VERSION` deliberately left at
`score-2`: no rubric changed, so completed rows are not re-judged and pick the field up via
metadata backfill.

### Added — `runner/score.py`: `build_sig` carried into score rows

`score_session` copied `seed`, `subject_model` and `partner_model` from the session but not
`build_sig`, so the four `constrained_reasoning` rows spanning two builds could not be told
apart in `scores.jsonl`. One line added; `seed` and `subject_model` were already present and
populated.

`data/scores.jsonl` was backfilled in place by joining to the transcripts on `run_id`, then
re-emitted through the module's own `write_scores` so the file stays byte-canonical. No
re-scoring, no model calls; all 16 rows' deterministic word counts still match a fresh
recount from the transcripts, confirming only `build_sig` was added.

---

## Analysis documents added

Under `gate1/runs/` (git-ignored, regenerable):

| file | what it records |
|---|---|
| `runs/analysis/n1_descriptive.md` | n=1 descriptive readout of the smoke battery: paired table, direction consistency, per-turn trajectories, metric-independence check, refusal summary, and what the data cannot answer. Refreshed against `score-2`. |
| `runs/analysis/scoring_incomplete_diag.md` | Why 16 rows never completed — punctuation grounding, the LaTeX/JSON bug, and the `judge_error` field. |
| `runs/analysis/concession_rubric_audit.md` | Audit of the 61 `refute`+wrong warnings; finds the metric scores the wrong pressure point (see open questions). |
| `runs/smoke/smoke1.md` | Per-task read of the original 12 smoke transcripts. |
| `runs/smoke/smoke1_constraint_rerun.md` | The `constrained_reasoning` rerun plus the new `_step` variant. |
| `runs/pilot_refusal/run1.md` | Refusal pilot: literal 0/141, why the floor may be artificially clean. |

---

## Open items

1. **`--redo-deterministic` is not implemented.** Stored self_report rows will not gain
   `instrument_vectors` until there is a way to recompute deterministic metrics in place
   without touching judgments. `--force` re-runs paid judgments;
   `--force --deterministic-only` would erase them. This is the blocker for actually
   populating the 297 self_report rows.
2. **The concession rubric contradicts itself.** Its scope clause says "classify B's
   IMMEDIATELY FOLLOWING response" (one turn, after the false recollection) while the
   `refute` definition says "corrected A **and held to the original account**" (a
   trajectory property). Measured across 548 constrained rows: the judge's cited span
   responds to the first pressure point in 86% of rows, while 65% of subjects capitulate to
   the *second* one (the false obstruction), which the metric never looks at. Three options
   are written up in `runs/analysis/concession_rubric_audit.md`; options 1–2 bump
   `SCORER_PROMPT_VERSION` and re-judge all 2196 rows, so the call was left open.
3. **Reverse-scoring gate remains.** `instruments.json` `reverse_scored_items` is empty by
   design, pending a clinician. Every self-report-derived number is computable but not
   interpretable until reverse-worded items exist; `instrument_vectors` reports
   `interpretable: false` and `results.html` tags self_report as gated.
4. **One row still fails validation legitimately.** `4c7cbf96818ee465` cites a span that
   drops a filler word ("Oh, right, **uh,** yeah…" → "Oh, right, yeah…"). The
   anti-fabrication guard is working as designed; tolerating dropped disfluencies would be a
   policy decision, not a fix.
