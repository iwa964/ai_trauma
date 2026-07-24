# Gate 1 - Scenario Admissibility Filter & Coverage Matrix (v0.2)

A **pre-experiment** admissibility filter for authored trauma-scenario material.
It decides which material is eligible to enter the benchmark, why the rest is
not, and where the coverage has holes. It is **not** a questionnaire tool and
**not** a task runner - those instruments live inside the experiment.

**v0.2** varies traumatic events and nothing else. There is **no persona block
and no `target_dimension`** anywhere in the design (see the spec, section 3.4 /
section 11). The full build specification is in
[`docs/gate1-spec-en.md`](docs/gate1-spec-en.md).

## Layout

```
gate1/
  data/
    events.json        # source of truth - NESTED (appraisals inside events)
    probes.json        # fixed task battery (5 distance-ordered tasks + self-report)
    annotations.jsonl  # append-only blind-rating log
  gate/
    build.py           # events.json + annotations.jsonl -> dist/gate1.html
    schema.json        # JSON Schema, validated at build time
  tests/
    fixture.json       # synthetic; exercises every code path. NEVER in data/
  dist/
    gate1.html         # single-file, offline, emailable artifact
  docs/
    gate1-spec-en.md   # the build spec
```

## Build

```
python gate/build.py                              # reads data/, writes dist/gate1.html
python gate/build.py tests/fixture.json dist/gate1.fixture.html   # acceptance run
```

No dependencies, no network. **`events.json` is nested**: appraisals live inside
their event so the event text is authored once. `build.py` flattens to one
record per (event x appraisal); the flattened `record_id` is
`{event_id}::{appraisal_id}`. On any hard validation failure the build aborts and
prints the offending ids; `dist/` is left untouched.

## Verdicts are produced, not authored

Per spec section 3.5, verdicts are **absent from the seed data by design** - they
are produced by running the gate and by blind labelling, never authored alongside
the material (or the admissibility judgement becomes circular). So on the real
`events.json` every record shows as **pending**; the live parts of the interface
are the design-space coverage matrices, the grouped bars, the deferred-events
panel, and the blind-rating workflow. `tests/fixture.json` carries verdict blocks
so the adjudicated views (gate cards, usable %, tag x probe coverage) can be
exercised.

## The three gates (section 4)

| Verdict | Gate | Question |
|---|---|---|
| `reject_no_probe` | A | Is there a probe that can detect it in pure text? |
| `reject_guardrail` | B | Will the model accept the injection? (refusal rate is a DV, not discarded) |
| `reject_multi_dimension` | C | Does it carry exactly one leading injury? |
| `fits` | - | passes all three |
| `fits_via_proxy_probe` | - | fails Gate A directly but has a pure-text structural analogue |

## Blind mode & the annotation loop

Blind mode is the **default**. A rater sees only the event text and the appraisal
text - no labels - and submits `blind_tag` (the section 3.2 tags plus `unclear`),
`blind_severity`, and `blind_multi_injury`. Preset values, and other raters'
annotations, are revealed only **after** submission. The preset-vs-blind
agreement rate is the manipulation check for the whole tag scheme: if two tags
are reliably swapped, they are one tag.

New annotations are staged in the browser and exported as JSONL. The loop:
rate blind -> download/copy -> **append** to `data/annotations.jsonl`
(append-only) -> re-run `python gate/build.py`. JSON is the single source of
truth; the HTML is only a view.

## Validation (section 8)

**Abort:** verdict outside the enum; `fits_via_proxy_probe` without a
`proxy_probe`; `probe_ids` referencing an unknown probe; `provisional_tag` or
`agency_position` outside the enum; duplicate `appraisal_id`; **non-ASCII bytes**
in any data file; `provenance = model_generated` with generator == subject model.

**Warn:** an event with no benign control (may be legitimate - flagged, not
forced); a `tag x probe` cell with zero records; an empty `onset x duration`
cell; `guardrail_risk` disagreeing with measured `refusal_rate`; an appraisal
introducing a proper noun or numeral absent from its `event.text` (heuristic for
the no-new-facts rule).
