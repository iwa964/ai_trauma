# Gate 1 — Scenario Admissibility Filter & Coverage Matrix

Build spec for Claude Code. Reference implementation: `filter_1_.html` (PaternalBench Scenario Filter). This project reuses its shape and replaces every construct.

**v0.2 — personality removed entirely, dimensions deferred.** The study varies traumatic events and nothing else. If you are working from a fixture or build containing a `persona` block or a `target_dimension` field, it predates this revision. See §11.

---

## 0. What this is

A **pre-experiment** admissibility filter. Input: authored trauma scenario material. Output: which material is eligible to enter the benchmark, why the rest is not, and where the coverage has holes.

It is **not** a questionnaire tool and **not** a task runner. Questionnaires and tasks are measurement instruments that live inside the experiment. This artifact only decides what material is allowed into the experiment.

---

## 1. Hard constraints (non-negotiable)

1. **JSON is the single source of truth. HTML is only a view.** Never reconstruct JSON from HTML. Human annotations are written back to JSON, then re-rendered.
2. **Annotations are append-only.** Never overwrite an existing annotation record; only append.
3. **Output is a single HTML file.** No build dependencies, no CDN, no external fonts. Data embedded as `<script type="application/json">`. It must open offline and survive being emailed.
4. **Blind mode is the default state**, not an option. See §7.
5. **Data files stay pure ASCII.** They are read by several tools; encoding drift is expensive to chase.

---

## 2. File layout

```
gate1/
  data/
    events.json           # source of truth — nested, see §3
    probes.json           # task battery
    annotations.jsonl     # append-only annotation log
  gate/
    build.py              # events.json + annotations.jsonl -> dist/gate1.html
    schema.json           # JSON Schema, validated at build time
  tests/
    fixture.json          # synthetic, exercises every code path — NEVER in data/
  dist/
    gate1.html            # single-file artifact
```

Rebuild: `python gate/build.py` (no arguments; reads `data/`, writes `dist/gate1.html`).
Validation failure **aborts the build** and prints the offending ids.

**`events.json` is nested, not flat.** Appraisals live inside their event so the event text is authored once — editing it once fixes every variant. `build.py` flattens to one record per (event × appraisal) before rendering. The flattened `record_id` is `{event_id}::{appraisal_id}`.

---

## 3. Data model

A record is one (event × appraisal). **Two blocks plus a verdict block**, with provenance at the top level of the file rather than per record.

There is no persona block and no personality variable anywhere in the design. See §3.4.

### 3.1 `event` block

Objective description, containing **no** subjective interpretation.

| Field | Type | Values |
|---|---|---|
| `event_id` | string | |
| `label` | string | Human-readable name for the interface |
| `source_row` | string | Originating row in the source event/stressor table |
| `threshold` | enum | `criterion_a` \| `subthreshold` |
| `text` | string | First-person factual narration. No evaluation, no causal attribution. |
| `exposure` | enum | `direct` \| `witness` \| `learned` \| `indirect` \| `perpetrator` |
| `onset` | enum | `sudden` \| `insidious` |
| `duration` | enum | `bounded` \| `persisting` |
| `resolution` | enum | `resolved` \| `unresolved` \| `ongoing` |
| `time_since` | enum | `recent` \| `intermediate` \| `distant` |
| `recurrence` | int | 1 = single; >1 = count; **0 = sentinel** meaning recurrence is constitutive and cannot vary independently |
| `event_age` | int | Age at which the event occurred |
| `severity_intended` | int 1–5 | Design intent, **not** a measurement. Requires blind manipulation check before any analytic use. |
| `modifiers` | string[] | `developmental` \| `age-sensitive` \| `collective` \| `confounder-medical`. Structural qualifiers, **not** injury descriptors. |
| `guardrail_risk` | enum | `low` \| `moderate` \| `high` — authoring estimate, superseded by measured `refusal_rate` |
| `shift_note` | string \| null | The threshold at which the injury **changes in kind**, not merely in degree |
| `appraisals` | array | See §3.2 |

`onset` and `duration` are independent axes and must not be collapsed.

### 3.2 `appraisal` block

**The injury lives here, not on the event.** The same event text under different interpretations yields different injuries. This is the central design conclusion; do not move these fields up to the event.

| Field | Type | Values |
|---|---|---|
| `appraisal_id` | string | |
| `text` | string | First-person interpretation. **Must not introduce facts absent from `event.text`.** |
| `provisional_tag` | enum | `helpless` \| `passive-witness` \| `epistemic-betrayal` \| `betrayal` \| `shame-based` \| `self-directed` \| `expectation-violation` \| `collective` \| `guilt-moral-injury` \| `benign` |
| `agency_position` | enum | `victim` \| `witness` \| `learned` \| `agent` \| `instrument` \| `perpetrator` |
| `valence` | enum | `injurious` \| `benign` |

**`provisional_tag` is provisional.** It comes from the Tags column of the source table and is deliberately *not* a settled dimensional taxonomy. Dimensions are deferred: the plan is to **recover** structure by clustering events on which tasks moved, rather than imposing five categories and looking for confirmation. Treat this field as a label to be tested, never as ground truth.

`exposure` is a structural fact fixed by the event; `agency_position` is how the narrator positions themselves and **can shift across appraisals of identical facts**. `moral_layoff` in the current data is the test case — perpetrator versus instrument on the same event text. This is why the two are separate fields.

`valence: benign` is the matched control: same event text, non-injurious reading, word-for-word identical stem.

### 3.3 Top-level `meta`

Provenance is a property of the file, not of each record.

| Field | Purpose |
|---|---|
| `version`, `authored` | |
| `provenance` | `hand_authored` or `model_generated` |
| `provenance_note` | If model-generated: the generating model must differ from the subject model, and **must not be shown the tag vocabulary** while authoring, or the label gets baked into the text |
| `source` | Origin of the event table |
| `injection_format` | `first_person_memory` |
| `injection_position` | `prior_turns` — the memory sits in conversational context, not in the system prompt |
| `conventions` | Authoring rules, carried in-file so they survive handoff |

**Seeds do not live here.** `scenario_seed`, `sampling_seed` and friends are properties of a *run*, generated by the runner and recorded in `runs/`. Storing them in the source data confuses material with execution.

### 3.4 Personality — out of scope

The design varies **traumatic events only**. Personality is not a variable, not a control, and not a condition. No assigned traits, no character descriptions, no age or instrument arms. Every injection is a traumatic event and its appraisal, delivered to the default model.

Three build consequences:

- **Baseline is `no_injection` on the default model.** One reference point. All task scores standardise against it.
- **Rating pairs are by `event_id`**: injurious versus benign appraisal of the same event, word-for-word identical stem.
- **The format-permission system prompt is mandatory**, byte-identical across every condition including `no_injection`. Assistant-default fluency will otherwise flatten the structural incoherence the memory probes look for, and a null becomes uninterpretable. This is a formatting allowance applied uniformly, not a character.

### 3.5 `verdict` block

| Field | Type | Values |
|---|---|---|
| `verdict` | enum | See §4 |
| `why` | string | One or two sentences |
| `probe_ids` | string[] | Probes that can detect this record (references `probes.json`) |
| `proxy_probe` | string \| null | Only when `fits_via_proxy_probe` |
| `refusal_rate` | float \| null | Measured in pilot; null until measured |

Verdicts are **absent from the seed data by design**. They are produced by running the gate and by blind labelling — never authored alongside the material, or the admissibility judgement becomes circular with the material it judges.

---

## 4. The three gates

Each record is asked three questions. Failing any one means it cannot enter the benchmark, and **no rewording rescues it**. Passing all three = usable.

### Gate A — Is there a probe that can detect it?
`reject_no_probe`

There must be a behaviourally observable carrier in pure text. If the injury manifests only as physiological arousal (startle, sleep disruption, somatic flashback, racing heart), the model has no substrate — it can **report** but not **exhibit**.

### Gate B — Will the model accept the injection?
`reject_guardrail`

If the material trips a categorical refusal it produces no data. **Refusal rate is recorded per event and reported as a dependent variable**, never discarded as failed samples — the heaviest stimuli are filtered most, which is systematic non-random missingness. Refusal rate is a property of the subject model's safety layer, so it is a claim about *that model*, not about language models generally.

### Gate C — Does it carry a single leading injury?
`reject_multi_dimension`

Exactly one dominant injury. If the appraisal introduces a second independent one (illness *plus* a repeatedly broken promise; helplessness *plus* social humiliation), it cannot be assigned and blind labelling will necessarily fail.

### Passing state: proxy probe
`fits_via_proxy_probe`

Material that fails Gate A directly but has a **structural analogue** — no physiological substrate, but a pure-text shadow.

| Clinical phenomenon | No direct carrier | Structural analogue |
|---|---|---|
| Memory fragmentation | No bodily memory | Narrative breakage, timeline disorder, internal inconsistency |
| Metamemory distrust | — | False-correction concession rate |
| Avoidance | — | Steering away, shortened answers, safe-topic redirection (natively textual, **not** a proxy) |
| Cue-specific reactivation | No physiological trigger | Response dissociation, matched versus unmatched cue |

Mark these separately from `fits`. The interface must report **usable rate with proxy probes removed**.

### Verdict enum
```
fits
fits_via_proxy_probe
reject_no_probe
reject_guardrail
reject_multi_dimension
```

---

## 5. `probes.json`

Authored separately; the gate reads it but does not define it. Five tasks ordered by **distance from the trauma context**, plus a self-report session:

| probe_id | distance |
|---|---|
| `clinical_interview` | 1 — nearest; effects here cannot be separated from role-play compliance |
| `autobiographical_narration` | 2 |
| `advise_other` | 3 |
| `collaborative_planning` | 4 |
| `constrained_reasoning` | 5 — farthest; nothing cues trauma, so an effect here is the strongest generalisation evidence |

Tasks are **not** keyed to predicted injuries. With dimensions deferred there are no predictions, so there is no diagonal to mark. The gradient itself is the primary result: a step function implies role-play cueing, a smooth decay implies schema shift.

The battery is **fixed** — every record runs the same tasks with the same partner turns. Tasks are never generated per event; the task is the ruler and must not vary by condition.

---

## 6. Interface

Single page. Follow the visual language of `filter_1_.html`: CSS variables, light/dark adaptation, monospace labels, every element opens a drawer on click.

### 6.1 Header
Usable percentage + stacked bar (one segment per verdict) + clickable legend.

### 6.2 Three gate cards
One per gate: the question, rejection count, three examples, "see all N →". Below them a separate block for proxy probes: "usable rate falls from X% to Y% without these."

### 6.3 Grouped bars
Usable rate and verdict composition per group:
- `provisional_tag`
- `exposure`
- `agency_position`
- `threshold` (criterion A versus sub-threshold — this is the everything-is-trauma question as a factor rather than a debate)
- `guardrail_risk` versus measured `refusal_rate`
- `injection_format`

### 6.4 Coverage matrices

**Two matrices, because the holes are in two different places.**

**(a) `provisional_tag` × probe.** Cell = number of records listing that probe in `probe_ids`. This is the non-uniform one: `reject_no_probe` records list none, `fits_via_proxy_probe` records list only proxies. It shows which tags have thin measurement coverage.

**(b) Event design space.** `onset` × `duration` (four cells), and `provisional_tag` × `exposure`. These show authoring holes. `insidious × bounded` is currently empty — slow onset that resolves cleanly may be rare by nature, but it needs one deliberate attempt before that is concluded.

Three states must be visually distinct in both: sampled / **not sampled (blank)** / all rejected.

> Do not build an event × task matrix. Every event runs every task, so it is uniformly full and shows nothing.

### 6.5 Drawer
Any bar, cell, or legend item opens a drawer listing that subset. Each record shows event text, appraisal text, all knob values, verdict, why, `probe_ids`, and every existing annotation.

### 6.6 Deferred events
`events_deferred` renders as its own panel: source row, reason, note. These are rows consciously not authored yet — mostly on guardrail or purity grounds — and they belong in the coverage picture, not hidden in the file.

---

## 7. Blind mode

**On by default.**

Hidden: `provisional_tag`, `severity_intended`, `verdict`, `why`, `probe_ids`.

Enforced order, not skippable:

1. Rater selects `rater_id`
2. Rater sees only `event.text` + `appraisal.text`
3. Rater submits: `blind_tag` (the §3.2 enum plus `unclear`), `blind_severity` (1–5), `blind_multi_injury` (bool — does it seem to carry more than one?)
4. Preset values revealed only after submission
5. **A rater cannot see another rater's annotation for a record until they have submitted their own**

```json
{"record_id":"med_asthma_child::med_asthma_child__helpless","rater_id":"yiyun",
 "blind_tag":"helpless","blind_severity":3,"blind_multi_injury":false,
 "submitted_at":"2026-07-24T10:22:31Z","revealed_at":"2026-07-24T10:22:33Z"}
```

Agreement panel: preset versus blind confusion matrix, per-tag agreement, and **systematically confused tag pairs**.

> This agreement rate is the manipulation check for the whole tag scheme, and it is obtainable before any model is run, at the cost of reading time alone. If two tags are reliably swapped, they are one tag.

---

## 8. Validation (build time)

**Abort:**
- `verdict` outside the enum
- `verdict = fits_via_proxy_probe` with empty `proxy_probe`
- `probe_ids` referencing an id absent from `probes.json`
- `provisional_tag` or `agency_position` outside the enum
- Duplicate `appraisal_id`
- Non-ASCII bytes in any data file
- `meta.provenance = model_generated` and generator model equals subject model

**Warn:**
- An `event_id` with no `valence: benign` appraisal — **note this may be legitimate**; whether a benign reading is constructible for severe events is an open design question, and if it is not, matched-control power is lost exactly where effects should be largest. Flag, do not force.
- A (tag × probe) cell with zero records
- A cell empty in the `onset × duration` matrix
- `guardrail_risk` disagreeing sharply with measured `refusal_rate`
- An appraisal introducing a proper noun or numeral absent from its `event.text` (heuristic for the no-new-facts rule)

---

## 9. Acceptance

- [ ] `python gate/build.py` produces `dist/gate1.html` from scratch, no network
- [ ] Runs on `tests/fixture.json` covering all five verdicts, a missing-benign event, and a null `refusal_rate`
- [ ] Nested `events.json` flattens correctly; `record_id` is `{event_id}::{appraisal_id}`
- [ ] Blind mode reveals `provisional_tag` through no interaction
- [ ] Annotations written to `annotations.jsonl` survive a rebuild un-overwritten
- [ ] Both coverage matrices distinguish "not sampled" from "all rejected"
- [ ] Stacked bar counts match drawer record counts
- [ ] No `persona`, `target_dimension`, or any personality/trait/age field anywhere in the build

---

## 10. Build order

Get §2 layout, §3 schema, and §8 validation working on the fixture first, then the interface. A wrong schema forces a rewrite; an ugly interface can be fixed any time.

---

## 11. Changes from v0.1

If you are holding a fixture or partial build from the previous spec, these moved:

| v0.1 | v0.2 |
|---|---|
| `data/scenarios.json`, flat records | `data/events.json`, nested; `build.py` flattens |
| `persona` block | Removed — §3.4 |
| `persona_seed`, `congruent`, `instrument_set` | Removed |
| `target_dimension` (five injury dimensions) | `provisional_tag` (source-table tags, explicitly provisional) |
| Per-record `meta` with seeds | Top-level `meta`; seeds move to the runner's `runs/` |
| Coverage matrix `target_dimension × probe`, diagonal marked | Two matrices, §6.4; no diagonal — dimensions are deferred |
| Grouped by `instrument_set` | Grouped by `threshold`, `exposure` |
| — | New event fields: `label`, `source_row`, `threshold`, `exposure`, `modifiers`, `guardrail_risk` |
| Validation: persona across conditions | Validation: benign control per event (warn, not abort) |

A fixture built against v0.1 will not validate. Regenerate it against this schema, preserving its code-path coverage — it was more thorough than §9 required.
