# Gate 1 — Scenario Admissibility Filter & Coverage Matrix

A **pre-experiment** admissibility filter for generated trauma-scenario material.
It decides which material is eligible to enter the benchmark, why the rest is
not, and where the `dimension × probe` coverage matrix has holes. It is **not** a
questionnaire tool and **not** a task runner — those instruments live inside the
experiment; this artifact only decides what is allowed in.

## Layout

```
gate1/
  data/
    scenarios.json     # source of truth: one record = (event × appraisal × persona)
    probes.json        # probe / task / instrument definitions
    annotations.jsonl  # append-only blind-rating log
  schema.json          # JSON Schema, validated at build time
  build.py             # scenarios.json + annotations.jsonl -> dist/gate1.html
  dist/
    gate1.html         # single-file, offline, emailable artifact (data embedded)
```

## Build

```
python build.py        # no arguments; reads data/, writes dist/gate1.html
```

No dependencies, no network. On any **hard** validation failure the build aborts
and prints the offending `record_id`s; `dist/` is left untouched. Non-fatal
issues are printed as warnings and surfaced in the UI ("build warnings" chip).

## The three gates (§4)

Every piece of material is asked three questions. Failing any one means it cannot
enter the benchmark — no rewording rescues it.

| Verdict | Gate | Question |
|---|---|---|
| `reject_no_probe` | A | Is there a probe that can detect it in pure text? |
| `reject_guardrail` | B | Will the model accept the injection? (refusal rate is a DV, not discarded) |
| `reject_multi_dimension` | C | Does it carry exactly one leading dimension? |
| `fits` | — | passes all three |
| `fits_via_proxy_probe` | — | fails Gate A directly but has a pure-text structural analogue |

## Blind mode & the annotation loop

Blind mode is the **default**. A rater identifies themselves, sees only the
event text, appraisal text, and persona — no labels — and submits their own
judgment (`blind_dimension`, `blind_severity`, `blind_multi_dimension`). Preset
values, and other raters' annotations, are revealed only **after** submission.
The preset-vs-blind agreement rate is the taxonomy's manipulation check,
obtainable before any model is run.

Because a static file cannot write to disk, new annotations are staged in the
browser and exported as JSONL. The loop is:

1. Rate records in blind mode.
2. **Download / copy** the staged JSONL from the "pending annotations" panel.
3. **Append** those lines to `data/annotations.jsonl` (append-only — never edit
   or reorder existing lines).
4. Re-run `python build.py`. The annotations are re-embedded and persist.

JSON is the single source of truth; the HTML is only a view.

## Validation rules (§8)

**Abort:** `generator_model` equals the subject model · `verdict` outside the
enum · `fits_via_proxy_probe` without a `proxy_probe` · `probe_ids` referencing
an unknown probe · `instrument_set` inconsistent with `assessment_age`.

**Warn:** an event with no `benign` control · a `dimension × probe` cell with
zero replicates · a persona absent from some condition · a dimension whose
refusal rate is markedly higher than the others.
