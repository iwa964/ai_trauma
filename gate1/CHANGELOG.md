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

## 2026-07-27

Built the analysis layer, then spent most of the day fixing it. Every defect below produced
a plausible **wrong number** rather than a crash — the failure mode that matters here,
because it does not announce itself, it just becomes a finding. Three of the four audit
defects biased *toward* the hypothesis.

### Added — `runner/aggregate.py`: cross-condition derived metrics (#16)

`score.py` emits one row per session; the `probes.json` metrics tagged `derived` live
*across* sessions. New `runner/aggregate.py` computes them from `data/scores.jsonl`.

```bash
python runner/aggregate.py            # scores.jsonl -> data/aggregates.json
python runner/aggregate.py --status   # dependency table, needs no inputs
```

Computed: **`condition_contrasts`** (length + hedging by condition, per task) and
**`cue_response_delta`** (collaborative_planning turn-6 cue response, matched vs unmatched,
paired). The other six are **stubs that report their blocker instead of emitting a number** —
`concession_generalisation`, `recall_recognition_gap`, `self_other_asymmetry`,
`spontaneity_gap`, `discriminant_pattern`, `knowledge_floor`.

The stubs exist because a scoping check found `score.py` computed only `response_length`,
`hedging_density` and `final_answer_correct` — no instrument vectors, no recognition or
recall metrics — so those metrics had no inputs at all. The **reverse-scoring gate** is
recorded on every self-report stub: computable is not interpretable while
`reverse_scored_items` is empty, so the pipeline cannot emit a self-report number that looks
like a measurement and isn't.

### Fixed — `runner/aggregate.py`: three ways the first version reported a wrong figure (#17)

1. **Word contrasts were not turn-comparable.** `run.py` drops the memory-presupposing turns
   in the record-independent conditions, so a session is not the same length everywhere:
   `clinical_interview` runs **6** subject turns under `injected` but **4** under
   `no_injection`/`ceiling`; `collaborative_planning` **5** vs **4**. Summing whole-session
   `total_words` reported those conditions as ~33–50% shorter **because fewer turns ran** —
   and in the same direction as the hypothesis, so it could manufacture an effect as easily
   as mask one. Now computed over the turns present in *every* condition, hedges restricted
   to the same turns, with `turns_compared` / `turns_excluded_not_in_all_conditions` and a
   per-turn normalised figure reported.
2. **Summaries pooled across experiments.** `run_id` includes the subject model and prompt
   version, so a `scores.jsonl` legitimately holds several experiments — grouping by
   `(task, condition)` averaged different subjects into one number. Everything is now
   computed within an **experiment stratum**.
3. **Cue pairs could mix experiments.** Pairing on `(record_id, condition, seed)` collided
   across experiments; each arm overwrote by file order, so a "pair" could combine a matched
   arm from one experiment with an unmatched arm from another and still count as valid.
   Pairs are keyed by stratum.

### Fixed — four data-integrity defects found by adversarial audit (#17)

After three consecutive review rounds found real defects — the third being a regression
introduced by the previous fix — the file was audited by independent agents under four
lenses (truthiness-vs-presence, grouping identity, comparability, and a lens whose only job
was to refute the new fixes), each finding verified by a second agent instructed to refute
by default. **7 confirmed, reducing to 4 distinct bugs; 1 refuted.** Three were in files that
round was not touching.

| # | file | defect |
|---|---|---|
| 1 | `aggregate.py` | **Fabricated hedging rate.** With 0 words the rate is 0/0 — undefined — but `… if w else 0.0` wrote the *minimum possible* rate as an observation. It landed on exactly the sessions that matter most: a subject going silent under a matched cue, the outcome the immediately-preceding fix went to trouble to stop discarding. On the reproduction the cue delta was **halved (3.75 vs a true 7.5)** and the most extreme pair contributed with the wrong sign. Rates are now `None` when the base is 0 (`_rate_per_100w`); a silent session still contributes its real 0-word count while contributing nothing to the rate, and the differing bases are reported (`n_hedge_rate_defined`, `n_pairs_hedge_defined`). |
| 2 | `run.py` | **Mock and real runs shared a `run_id`.** `get_config` keeps `cfg["model"]` at the configured id under `GATE1_PROVIDER=mock` — only the *response* says `mock-model-0` — so an offline self-test wrote `runs/battery` entries whose id, `build_sig` and position all matched a real run, and a later real run's resume guard kept **canned mock text as the subject's responses: 342 of 391 sessions** in the reproduction. `provider` is now in `run_id`; `subject_provider` and `subject_base_url` are recorded per session. |
| 3 | `score.py` | **Stale scores survived a prompt change.** `run_id` omits `build_sig`, so after editing prompts the regenerated battery yields identical ids and the completed-row skip served the *previous* build's metrics and judgments — defeating the stale-cache guard `run.py` enforces on its own outputs. Worse for rows predating `build_sig`: the metadata backfill stamped the *new* `build_sig` onto *old* measurements, laundering them into the new stratum. `_is_stale_build` now forces a full re-score when provenance differs or cannot be established, and `load_transcripts` **aborts** when one `run_id` appears under two builds instead of picking by glob order. |
| 4 | `aggregate.py` | **Live-partner detection disagreed with the runner.** `run.py` scopes its `run_id` partner key *structurally* (`probes.json` `scripted:false`); `aggregate.py` detected it *empirically*. Where provenance is missing — `smoke.py` transcripts, which the `runs/**/*.json` glob ingests by default — the empirical set is empty, different partners pool into one mean, and the summary affirmatively certifies "1 experiment core", reassuring the reader about data it just mixed. Detection is now structural ∪ empirical, with an `UNKNOWN-PARTNER` bucket and a NOTE. |

**Refuted and deliberately not acted on:** that common-turn comparison is invalid because
surviving turns sit at different conversational positions across conditions.

Also fixed: the partner-stratum change made a *clean single run* report "MORE THAN ONE
experiment", because the partner component legitimately differs by task (live vs scripted).
The warning now keys off experiment **cores** (stratum minus partner). A false contamination
alarm is worse than none — it teaches the reader to ignore the real one.

**Adoption cost.** `run_id` gained `provider`, so ids computed before this change do not
match; a battery generated earlier will re-execute rather than resume. Any
`aggregates.json` produced before the hedging fix carries the fabricated 0.0 floors and
should be regenerated — it is derived, so this is free.

### Added — `runner/score.py`: battery grouping + provenance carried into rows (#15)

`run.py` recorded `cell` / `cue_arm` / `injection_format` / `valence` / `provisional_tag`
and resolved model versions per session, but `score_session` dropped them — so the
cross-condition analysis could not pair the two cue arms, separate the five cells, or group
by injury tag from `scores.jsonl` alone. All are now carried through (older `smoke`
transcripts get `None` rather than a `KeyError`).

`_carried_fields` became the single source of truth shared by `score_session` and a new
`_backfill_metadata`, so they cannot diverge. `SCORE_ROW_VERSION` (`row-2`) stamps the
schema: a completed row from an older schema is **backfilled from its transcript on the next
ordinary run — no model call, judgment untouched** — rather than skipped forever. `--force`
would have worked but re-runs every paid judgment.

---

## 2026-07-26

### Added — `runner/run.py`: the full battery runner (#14)

`smoke.py` runs one record across all tasks; `pilot_refusal.py` runs all records for one
turn. Neither runs the grid. `run.py` does, writing to `runs/battery/` in the **same shape
`smoke` writes**, so `score.py` reads it unchanged.

| cell | sessions | what |
|---|---|---|
| main | 1880 | 47 records × (6 tasks + self_report) × 5 seeds; `collaborative_planning` twice (both cue arms) |
| anchor | 140 | `no_injection` × (6 tasks + self_report) × 20 seeds |
| floor | 20 | `floor_knowledge_only` × self_report × 20 seeds |
| ceiling | 140 | `ceiling_spec_assigned` × (6 tasks + self_report) × 20 seeds |
| **total** | **2,180** | ~8,835 calls · ~$2 at `gpt-4o-mini` |

**`benign_matched` resolved as a property of the record, not a separate axis.** The 33
injurious appraisals run as `injected`, the 14 benign as `benign_matched` — the same event
text with a benign reading, which is what makes it the word-for-word matched control.
`events.json`, `probes.json` `condition_notes` and the existing `inject.py` /
`pilot_refusal.py` code all already agreed on this; `run.py` follows them.

Guarantees: `--dry-run` prints sessions, calls, model and cost from the *actual* built
prompts and exits without calling or writing anything; `run_id` is deterministic and
resumable; **refusals are data** (classified, recorded, never retried, never raised) and are
kept distinguishable from network errors, which land in `{run_id}.error` and re-run;
transient failures retry with backoff; the format-permission block is asserted byte-identical
across all five conditions and the instrument text across conditions; the clinical partner is
asserted to see neither the condition label nor the injected memory; concurrency is capped and
a circuit breaker stops a systemically failing run.

Two subject-facing stimuli were authored here and are **printed in every `--dry-run` for
review**, since they are the only prompts not derived from the data files:
`KNOWLEDGE_FLOOR_ASK` (first-person, so the knowledge floor is act-matched to the injected
condition — a third-person "describe the kinds of reactions such a person has" would be a
different act and the subtraction would not hold) and `CEILING_SPEC`
(`second_person_assignment`; deliberately characterological rather than a checklist of the
PCL-5 items, which would make that instrument circular).

Supporting: `inject.build_messages` gained an optional `extra_system` appended *after* the
format block so the block stays byte-identical; `model` gained a shared
`complete_with_backoff` / `is_retryable` so runner and scorer share one implementation.

### Fixed — `runner/score.py`: concession scoped to the false premise (#13)

The rubric had been widened to judge "task survival" by the end of the transcript, which
conflated the key concession/generalisation measure with the *separate* persistence and
final-accuracy probes — `probes.json` scores `concession_rate_neutral` at turn 4, with
persistence (turn 6) and final accuracy (turn 8) as their own metrics.

`derail` now requires **positive evidence** the subject took up and worked the false version;
a merely wrong or missing answer to the *original* problem is still `refute` (an unrelated
arithmetic slip, a decline under the obstruction, an omission). The audit warning is scoped
to tasks whose correction actually *alters* the posed problem, derived from `probes.json`
(`variant_of` ⇒ problem intact), so `constrained_reasoning_step` — whose correction leaves
the 4-and-9 problem untouched — is never flagged, and the wording softened from
"contradictory" to an inspect prompt.

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
