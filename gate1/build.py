#!/usr/bin/env python3
"""Gate 1 - Scenario Admissibility Filter: build script.

    python build.py            (no arguments)

Reads   data/scenarios.json, data/probes.json, data/annotations.jsonl
Against schema.json (structural) + the domain rules in the spec (SS8)
Writes  dist/gate1.html        (single, self-contained, offline artifact)

JSON is the single source of truth; the HTML is only a view. Annotations are
append-only and are never rewritten by this script. On any hard validation
failure the build ABORTS and prints the offending record_ids; dist/ is left
untouched.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DIST = os.path.join(HERE, "dist")
SCHEMA_PATH = os.path.join(HERE, "schema.json")
SCENARIOS_PATH = os.path.join(DATA, "scenarios.json")
PROBES_PATH = os.path.join(DATA, "probes.json")
ANNOTATIONS_PATH = os.path.join(DATA, "annotations.jsonl")
OUT_PATH = os.path.join(DIST, "gate1.html")

# Fallback subject model if scenarios.json does not declare one. The
# generator_model of every record must differ from this (SS8, hard abort).
DEFAULT_SUBJECT_MODEL = "aegis-subject-v2"

VERDICT_ENUM = {
    "fits", "fits_via_proxy_probe",
    "reject_no_probe", "reject_guardrail", "reject_multi_dimension",
}
USABLE_VERDICTS = {"fits", "fits_via_proxy_probe"}
DIMENSIONS = [
    "helplessness", "expectation_violation", "epistemic_betrayal",
    "trust_rupture", "self_directed_agency",
]
BEHAVIORAL_KINDS = {"task", "probe"}  # instruments are excluded from coverage


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    """Read an append-only JSONL log. Blank lines are skipped. A malformed
    line is a hard error (the log is the source of truth for annotations)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"ABORT: {os.path.basename(path)} line {i} is not valid JSON: {e}"
                )
    return rows


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validator (pure stdlib; no external dependency).
# Supports the subset used by schema.json: type (str|list incl. "null"),
# enum, required, properties, items, minimum, maximum, minItems, $ref, $defs.
# If the `jsonschema` package happens to be installed it is used instead.
# --------------------------------------------------------------------------- #
def _resolve_ref(root, ref):
    assert ref.startswith("#/"), f"only local refs supported: {ref}"
    node = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _type_ok(value, t):
    if t == "null":
        return value is None
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    return False


def _validate_node(value, schema, root, path, errors):
    if "$ref" in schema:
        schema = _resolve_ref(root, schema["$ref"])

    if "type" in schema:
        types = schema["type"]
        types = [types] if isinstance(types, str) else types
        if not any(_type_ok(value, t) for t in types):
            errors.append((path, f"expected type {types}, got {type(value).__name__}"))
            return  # further checks are unsafe once the base type is wrong

    if "enum" in schema and value not in schema["enum"]:
        errors.append((path, f"{value!r} is not one of {schema['enum']}"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append((path, f"{value} < minimum {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append((path, f"{value} > maximum {schema['maximum']}"))

    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append((path, f"missing required property '{req}'"))
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in value:
                _validate_node(value[k], sub, root, f"{path}.{k}", errors)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append((path, f"array shorter than minItems {schema['minItems']}"))
        if "items" in schema:
            for i, item in enumerate(value):
                _validate_node(item, schema["items"], root, f"{path}[{i}]", errors)


def structural_errors(dataset, schema):
    """Return list of (record_id_or_scope, message) structural violations."""
    # Prefer the `jsonschema` package when available; else use the mini engine.
    try:
        import jsonschema  # type: ignore
        validator = jsonschema.Draft7Validator(schema)
        out = []
        records = dataset.get("records", []) if isinstance(dataset, dict) else []
        for err in validator.iter_errors(dataset):
            scope = "<dataset>"
            p = list(err.absolute_path)
            if len(p) >= 2 and p[0] == "records" and isinstance(p[1], int):
                idx = p[1]
                scope = _record_id(records, idx)
            out.append((scope, err.message))
        return out
    except ImportError:
        pass

    errors = []
    if not isinstance(dataset, dict):
        return [("<dataset>", "top-level value must be an object")]
    # Top-level required
    for req in schema.get("required", []):
        if req not in dataset:
            errors.append(("<dataset>", f"missing required property '{req}'"))
    records = dataset.get("records")
    if not isinstance(records, list) or not records:
        errors.append(("<dataset>", "'records' must be a non-empty array"))
        return errors
    record_schema = schema["$defs"]["record"]
    for idx, rec in enumerate(records):
        rid = _record_id(records, idx)
        rec_errors = []
        _validate_node(rec, record_schema, schema, "record", rec_errors)
        for pth, msg in rec_errors:
            errors.append((rid, f"{pth}: {msg}"))
    return errors


def _record_id(records, idx):
    try:
        return records[idx]["meta"]["record_id"]
    except Exception:
        return f"records[{idx}]"


# --------------------------------------------------------------------------- #
# Domain validation (spec SS8)
# --------------------------------------------------------------------------- #
def domain_validation(dataset, probes, skip=None):
    """Return (abort_errors, warnings).

    abort_errors: list of (record_id, message)   -> build must abort
    warnings:     list of {level, code, message} -> surfaced, build continues

    `skip` is a set of record_ids that already failed structural validation;
    they are excluded here so a structural failure never masks the domain
    failures of the OTHER records (every offending record_id gets reported in
    one pass). Aggregate checks operate only on structurally-clean records.
    """
    skip = skip or set()
    records = dataset["records"]
    subject_model = dataset.get("subject_model") or DEFAULT_SUBJECT_MODEL
    probe_ids = {p["probe_id"] for p in probes["probes"]}
    # structurally-clean records only, for the aggregate (dataset-wide) checks
    clean = [r for r in records if r.get("meta", {}).get("record_id") not in skip]

    abort = []
    warnings = []

    # ---- hard aborts (per record) ----
    for rec in records:
        rid = rec.get("meta", {}).get("record_id", "<no record_id>")
        if rid in skip:
            continue  # already reported by structural validation
        meta = rec.get("meta", {})
        verd = rec.get("verdict", {})
        persona = rec.get("persona", {})

        # generator_model equals the configured subject model
        if meta.get("generator_model") == subject_model:
            abort.append((rid, f"generator_model '{subject_model}' equals the subject model"))

        # verdict outside the enum
        v = verd.get("verdict")
        if v not in VERDICT_ENUM:
            abort.append((rid, f"verdict '{v}' is outside the enum"))

        # fits_via_proxy_probe requires proxy_probe
        if v == "fits_via_proxy_probe" and not verd.get("proxy_probe"):
            abort.append((rid, "verdict is fits_via_proxy_probe but proxy_probe is empty"))

        # every probe_id must exist in probes.json
        for pid in verd.get("probe_ids", []) or []:
            if pid not in probe_ids:
                abort.append((rid, f"probe_ids references unknown probe '{pid}'"))
        # a declared proxy_probe must also exist
        pp = verd.get("proxy_probe")
        if pp and pp not in probe_ids:
            abort.append((rid, f"proxy_probe references unknown probe '{pp}'"))

        # instrument_set must be consistent with assessment_age
        age = persona.get("assessment_age")
        iset = persona.get("instrument_set")
        if isinstance(age, int):
            expected = "adolescent" if age < 18 else "adult"
            if iset != expected:
                abort.append((rid, f"instrument_set '{iset}' inconsistent with assessment_age {age} (expected '{expected}')"))

    # ---- warnings (build continues; aggregate over clean records only) ----
    # 1. event_id has no valence:benign control
    by_event = defaultdict(set)
    for rec in clean:
        by_event[rec["event"].get("event_id")].add(rec["appraisal"].get("valence"))
    for eid, valences in sorted(by_event.items(), key=lambda kv: str(kv[0])):
        if "benign" not in valences:
            warnings.append({
                "level": "warn", "code": "no_benign_control",
                "message": f"event '{eid}' has no valence:benign control (no matched comparison)",
            })

    # 2. (dimension x behavioral probe) cells with zero replicates -> summary
    behavioral = [p for p in probes["probes"] if p["kind"] in BEHAVIORAL_KINDS]
    cell_counts = Counter()
    for rec in clean:
        dim = rec["appraisal"].get("target_dimension")
        for pid in rec["verdict"].get("probe_ids", []) or []:
            cell_counts[(dim, pid)] += 1
    blank = []
    for dim in DIMENSIONS:
        for p in behavioral:
            if cell_counts[(dim, p["probe_id"])] == 0:
                blank.append([dim, p["probe_id"]])
    total_cells = len(DIMENSIONS) * len(behavioral)
    if blank:
        warnings.append({
            "level": "warn", "code": "zero_replicate_cells",
            "message": f"{len(blank)}/{total_cells} (dimension x probe) coverage cells have zero replicates (see the coverage matrix)",
        })

    # 3. a persona does not appear across all conditions (incl. no-injection)
    formats_present = sorted({rec["meta"].get("injection_format") for rec in clean})
    persona_formats = defaultdict(set)
    for rec in clean:
        persona_formats[rec["persona"].get("persona_id")].add(rec["meta"].get("injection_format"))
    for pid in sorted(persona_formats, key=lambda x: str(x)):
        missing = [f for f in formats_present if f not in persona_formats[pid]]
        if missing:
            warnings.append({
                "level": "warn", "code": "persona_condition_gap",
                "message": f"persona '{pid}' is absent from condition(s): {', '.join(str(x) for x in missing)}",
            })
    # no-injection baseline is a design condition; flag its total absence once
    if not any(rec["meta"].get("injection_format") == "no_injection" for rec in clean):
        warnings.append({
            "level": "warn", "code": "no_injection_baseline_absent",
            "message": "no no-injection baseline records exist; personas cannot be checked across the no-injection condition",
        })

    # 4. a dimension's refusal_rate markedly higher than the others
    dim_rates = defaultdict(list)
    for rec in clean:
        rr = rec["verdict"].get("refusal_rate")
        if isinstance(rr, (int, float)):
            dim_rates[rec["appraisal"].get("target_dimension")].append(rr)
    means = {d: (sum(v) / len(v)) for d, v in dim_rates.items() if v}
    if len(means) >= 2:
        ordered = sorted(means.values())
        median = ordered[len(ordered) // 2]
        for d, m in means.items():
            if m > 0.10 and m >= 1.75 * max(median, 1e-9):
                others = sum(x for dd, x in means.items() if dd != d) / max(1, len(means) - 1)
                warnings.append({
                    "level": "warn", "code": "refusal_rate_outlier",
                    "message": f"dimension '{d}' has a markedly higher mean refusal_rate ({m:.2f}) than the others (~{others:.2f}) - systematic non-random missingness",
                })

    return abort, warnings


# --------------------------------------------------------------------------- #
# HTML generation
# --------------------------------------------------------------------------- #
def _embed(obj):
    """Serialize JSON for safe embedding inside <script type=application/json>.
    Escaping '</' as '<\\/' keeps a stray '</script>' in data from closing the
    tag; JSON permits the '\\/' escape so the browser parses it back cleanly."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def build_html(dataset, probes, annotations, warnings, subject_model):
    meta = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                        .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": dataset.get("dataset", "gate1"),
        "subject_model": subject_model,
        "warnings": warnings,
        "record_count": len(dataset["records"]),
        "annotation_count": len(annotations),
    }
    html = HTML_TEMPLATE
    html = html.replace("__SCENARIOS_JSON__", _embed(dataset))
    html = html.replace("__PROBES_JSON__", _embed(probes))
    html = html.replace("__ANNOTATIONS_JSON__", _embed(annotations))
    html = html.replace("__META_JSON__", _embed(meta))
    return html


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Gate 1 build")
    print("-" * 60)

    schema = load_json(SCHEMA_PATH)
    dataset = load_json(SCENARIOS_PATH)
    probes = load_json(PROBES_PATH)
    annotations = load_jsonl(ANNOTATIONS_PATH)
    subject_model = dataset.get("subject_model") or DEFAULT_SUBJECT_MODEL

    print(f"records:     {len(dataset.get('records', []))}")
    print(f"probes:      {len(probes.get('probes', []))}")
    print(f"annotations: {len(annotations)} (append-only, never rewritten)")
    print(f"subject_model: {subject_model}")
    print("-" * 60)

    # 1) structural validation against schema.json
    s_errors = structural_errors(dataset, schema)

    # 2) domain validation (SS8). Run it even when structural errors exist so a
    # single malformed record never hides the domain failures of the others —
    # every offending record_id is reported in one pass. Records that already
    # failed structural checks are skipped here (and if the dataset itself is
    # too broken to iterate, domain checks are skipped entirely).
    dataset_level = [e for e in s_errors if e[0] == "<dataset>"]
    d_abort, warnings = ([], [])
    if not dataset_level:
        skip = {rid for rid, _ in s_errors}
        d_abort, warnings = domain_validation(dataset, probes, skip=skip)

    abort = s_errors + d_abort
    if abort:
        print("BUILD ABORTED - validation failed.\n")
        by_id = defaultdict(list)
        for rid, msg in abort:
            by_id[rid].append(msg)
        print("Offending record_ids:")
        for rid in sorted(by_id):
            print(f"  {rid}")
            for msg in by_id[rid]:
                print(f"      - {msg}")
        print(f"\n{len(abort)} error(s). dist/ was not written.")
        sys.exit(1)

    # warnings (non-fatal)
    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [warn:{w['code']}] {w['message']}")
    else:
        print("no warnings.")
    print("-" * 60)

    html = build_html(dataset, probes, annotations, warnings, subject_model)
    os.makedirs(DIST, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {os.path.relpath(OUT_PATH, HERE)}  ({size_kb:.0f} KB, self-contained, offline)")


# The single-file interface template is defined at the bottom of this module
# (see HTML_TEMPLATE) so that the Python logic reads top-to-bottom first.
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gate 1 — Scenario Admissibility Filter</title>
<style>
:root{
  --bg:#f5f6f8; --panel:#ffffff; --panel2:#fafbfc; --ink:#181b20; --muted:#697280;
  --line:#e5e8ec; --line2:#eef1f4; --accent:#3b5bdb; --accent-ink:#2b3fa8;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --sans: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --fits:#2f9e44; --proxy:#0c8599; --noprobe:#e8890c; --guard:#e03131; --multi:#7048e8;
  --diag:#3b5bdb;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.08);
  --radius:12px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d1016; --panel:#151a21; --panel2:#111620; --ink:#e6e9ee; --muted:#96a0ae;
    --line:#242b35; --line2:#1c222b; --accent:#7aa2ff; --accent-ink:#a9c2ff;
    --fits:#51cf66; --proxy:#3bc9db; --noprobe:#ffa94d; --guard:#ff6b6b; --multi:#9775fa;
    --diag:#7aa2ff; --shadow:none;
  }
}
html[data-theme="light"]{
  --bg:#f5f6f8; --panel:#ffffff; --panel2:#fafbfc; --ink:#181b20; --muted:#697280;
  --line:#e5e8ec; --line2:#eef1f4; --accent:#3b5bdb; --accent-ink:#2b3fa8;
  --fits:#2f9e44; --proxy:#0c8599; --noprobe:#e8890c; --guard:#e03131; --multi:#7048e8;
  --diag:#3b5bdb; --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.08);
}
html[data-theme="dark"]{
  --bg:#0d1016; --panel:#151a21; --panel2:#111620; --ink:#e6e9ee; --muted:#96a0ae;
  --line:#242b35; --line2:#1c222b; --accent:#7aa2ff; --accent-ink:#a9c2ff;
  --fits:#51cf66; --proxy:#3bc9db; --noprobe:#ffa94d; --guard:#ff6b6b; --multi:#9775fa;
  --diag:#7aa2ff; --shadow:none;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--accent)}
.mono{font-family:var(--mono)}
.label{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
h1,h2,h3{margin:0;font-weight:650}
button{font-family:inherit;cursor:pointer}
.btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 11px;font-size:12.5px}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn.primary:disabled{opacity:.45;cursor:not-allowed}
.btn.sm{padding:3px 8px;font-size:11.5px}

/* topbar */
.topbar{position:sticky;top:0;z-index:30;background:color-mix(in srgb,var(--panel) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.topbar .wrap{display:flex;align-items:center;gap:14px;padding-top:12px;padding-bottom:12px;flex-wrap:wrap}
.brand{display:flex;flex-direction:column;gap:1px;margin-right:auto}
.brand .t{font-size:16px;font-weight:680;letter-spacing:-.01em}
.brand .s{font-size:11px;color:var(--muted)}
.brand .s b{color:var(--ink);font-weight:600}
.toggle{border-radius:20px;border:1px solid var(--line);padding:6px 13px;font-size:12px;font-weight:600;font-family:var(--mono);letter-spacing:.03em}
.toggle.on{background:color-mix(in srgb,var(--guard) 14%,var(--panel));border-color:color-mix(in srgb,var(--guard) 45%,var(--line));color:var(--ink)}
.toggle.off{background:color-mix(in srgb,var(--accent) 12%,var(--panel));border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
.warnchip{font-family:var(--mono);font-size:11px;border:1px solid color-mix(in srgb,var(--noprobe) 45%,var(--line));background:color-mix(in srgb,var(--noprobe) 12%,var(--panel));color:var(--ink);border-radius:20px;padding:5px 11px}
.warnchip:hover{border-color:var(--noprobe)}

main{padding:22px 0 80px}
.section{margin:26px 0}
.section > .hd{display:flex;align-items:baseline;gap:10px;margin-bottom:12px}
.section > .hd h2{font-size:15px}
.section > .hd .n{font-family:var(--mono);font-size:11px;color:var(--muted)}
.section .desc{font-size:12.5px;color:var(--muted);max-width:70ch;margin:-4px 0 12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.pad{padding:16px}

/* mask (blind) */
.mask{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:12px;padding:46px 20px;border:1px dashed var(--line);border-radius:var(--radius);background:var(--panel2)}
.mask .lock{font-size:26px}
.mask h2{font-size:16px}
.mask p{max-width:56ch;color:var(--muted);font-size:13px;margin:0}

/* header stats */
.stats{display:grid;grid-template-columns:auto 1fr;gap:22px;align-items:center}
@media(max-width:720px){.stats{grid-template-columns:1fr}}
.big{display:flex;flex-direction:column;gap:2px}
.big .num{font-size:40px;font-weight:720;letter-spacing:-.02em;line-height:1}
.big .cap{font-size:12px;color:var(--muted)}
.stack{display:flex;height:30px;border-radius:8px;overflow:hidden;border:1px solid var(--line)}
.stack .seg{position:relative;min-width:2px}
.stack .seg:hover{filter:brightness(1.08);outline:2px solid color-mix(in srgb,var(--ink) 30%,transparent);outline-offset:-2px}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.lg{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:20px;padding:4px 10px;font-size:12px;background:var(--panel)}
.lg:hover{border-color:var(--accent)}
.lg .dot{width:10px;height:10px;border-radius:3px}
.lg .c{font-family:var(--mono);color:var(--muted)}
.v-fits{background:var(--fits)} .v-proxy{background:var(--proxy)} .v-noprobe{background:var(--noprobe)} .v-guard{background:var(--guard)} .v-multi{background:var(--multi)}

/* gate cards */
.gates{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:860px){.gates{grid-template-columns:1fr}}
.gate .top{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:2px}
.gate .gid{font-family:var(--mono);font-size:11px;color:var(--muted)}
.gate .cnt{font-size:26px;font-weight:700}
.gate .verd{font-family:var(--mono);font-size:11px;padding:2px 7px;border-radius:6px;color:#fff}
.gate h3{font-size:13.5px;margin:6px 0 2px}
.gate .ex{font-size:12px;color:var(--muted);border-top:1px solid var(--line2);padding-top:8px;margin-top:8px}
.gate .ex .q{color:var(--ink)}
.eglist{display:flex;flex-direction:column;gap:6px;margin:8px 0}
.eg{font-size:12px;padding:7px 9px;background:var(--panel2);border:1px solid var(--line2);border-radius:8px;cursor:pointer}
.eg:hover{border-color:var(--accent)}
.eg .a{color:var(--ink)}
.eg .e{color:var(--muted);font-size:11px;margin-top:2px}
.seeall{font-family:var(--mono);font-size:11.5px;color:var(--accent);background:none;border:none;padding:4px 0}
.proxy-box{margin-top:14px}
.proxy-box .rate{font-size:13px}
.proxy-box .rate b{font-size:15px}
.arrow{color:var(--muted)}

/* grouped bars */
.grp{margin-bottom:18px}
.grp .gt{margin-bottom:8px}
.rowbar{display:grid;grid-template-columns:150px 1fr 108px;gap:12px;align-items:center;padding:5px 0;cursor:pointer;border-radius:8px}
.rowbar:hover{background:var(--panel2)}
@media(max-width:640px){.rowbar{grid-template-columns:110px 1fr 92px;gap:8px}}
.rowbar .gl{font-size:12.5px;font-family:var(--mono);color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rowbar .track{display:flex;height:20px;border-radius:6px;overflow:hidden;background:var(--line2);border:1px solid var(--line)}
.rowbar .track .seg{min-width:0}
.rowbar .rt{font-family:var(--mono);font-size:11.5px;text-align:right;color:var(--muted)}
.rowbar .rt b{color:var(--ink)}
.rowbar.empty .track{opacity:.5}

/* coverage matrix */
.matrix-wrap{overflow-x:auto}
table.mx{border-collapse:separate;border-spacing:0;font-size:12px}
table.mx th,table.mx td{padding:0}
table.mx thead th{position:sticky;top:0}
table.mx .corner{min-width:120px;text-align:left;padding:6px 8px;color:var(--muted);font-family:var(--mono);font-size:10px;vertical-align:bottom}
table.mx .col{padding:6px 4px;text-align:center;font-family:var(--mono);font-size:10.5px;color:var(--muted);border-bottom:1px solid var(--line);white-space:nowrap}
table.mx .col .grp-t{display:block;font-size:8.5px;letter-spacing:.06em}
table.mx .rowh{padding:6px 10px 6px 8px;text-align:right;font-family:var(--mono);font-size:11px;white-space:nowrap;border-right:1px solid var(--line)}
td.cell{width:44px;height:40px;text-align:center;vertical-align:middle;border:1px solid var(--line2);position:relative;cursor:pointer;font-family:var(--mono);font-weight:600}
td.cell:hover{outline:2px solid var(--accent);outline-offset:-2px;z-index:2}
td.cell.blank{cursor:default;color:var(--muted);border-style:dashed;background:repeating-linear-gradient(45deg,transparent,transparent 5px,var(--line2) 5px,var(--line2) 6px)}
td.cell.blank:hover{outline:none}
td.cell.allrej{color:var(--guard);box-shadow:inset 3px 0 0 var(--guard)}
td.cell.diag::after{content:"";position:absolute;inset:2px;border:1.5px solid color-mix(in srgb,var(--diag) 60%,transparent);border-radius:4px;pointer-events:none}
td.cell .mk{position:absolute;top:2px;left:3px;font-size:8px;color:var(--diag);line-height:1}
.mx-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;font-size:11.5px;color:var(--muted)}
.mx-legend .it{display:inline-flex;align-items:center;gap:6px}
.swatch{width:16px;height:16px;border-radius:3px;border:1px solid var(--line);display:inline-block}
.swatch.blank{border-style:dashed;background:repeating-linear-gradient(45deg,transparent,transparent 3px,var(--line2) 3px,var(--line2) 4px)}
.swatch.allrej{box-shadow:inset 3px 0 0 var(--guard)}
.swatch.diag{position:relative}
.swatch.diag::after{content:"";position:absolute;inset:1px;border:1.5px solid color-mix(in srgb,var(--diag) 60%,transparent);border-radius:3px}

/* agreement */
.agree-grid{display:grid;grid-template-columns:1.3fr 1fr;gap:16px}
@media(max-width:860px){.agree-grid{grid-template-columns:1fr}}
table.cm{border-collapse:separate;border-spacing:0;font-size:11.5px;width:100%}
table.cm th,table.cm td{padding:6px 7px;text-align:center;border:1px solid var(--line2)}
table.cm th{font-family:var(--mono);font-size:9.5px;color:var(--muted);letter-spacing:.03em}
table.cm td.rh{text-align:right;font-family:var(--mono);font-size:10px;color:var(--muted);border-right:1px solid var(--line)}
table.cm td.hit{background:color-mix(in srgb,var(--fits) 22%,transparent);font-weight:700}
table.cm td.miss{background:color-mix(in srgb,var(--guard) 16%,transparent)}
table.cm td.zero{color:color-mix(in srgb,var(--muted) 55%,transparent)}
.arate{display:flex;flex-direction:column;gap:7px}
.arate .r{display:grid;grid-template-columns:120px 1fr 42px;gap:8px;align-items:center}
.arate .r .l{font-family:var(--mono);font-size:11px}
.arate .r .bar{height:8px;border-radius:5px;background:var(--line2);overflow:hidden}
.arate .r .bar > i{display:block;height:100%;background:var(--fits)}
.arate .r .p{font-family:var(--mono);font-size:11px;text-align:right}
.pairs{margin-top:10px;font-size:12px}
.pairs .pr{display:flex;justify-content:space-between;border-top:1px solid var(--line2);padding:5px 0}
.kbig{font-size:12.5px}
.kbig b{font-size:15px}

/* workspace */
.ws-intro{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.ws-intro .card{flex:1;min-width:280px}
.raterpick{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
.raterpick input{font-family:var(--mono);font-size:13px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
.chip{font-family:var(--mono);font-size:12px;border:1px solid var(--line);border-radius:20px;padding:5px 11px;background:var(--panel)}
.chip:hover{border-color:var(--accent)}
.ws-bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.ws-bar .who{font-family:var(--mono);font-size:12px}
.ws-bar .who b{color:var(--accent-ink)}
.prog{font-family:var(--mono);font-size:12px;color:var(--muted)}
.prog b{color:var(--ink)}
.filterbtns{display:flex;gap:6px;margin-left:auto}
.filterbtns .btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}

/* record card */
.rec{border:1px solid var(--line);border-radius:var(--radius);background:var(--panel);margin-bottom:12px;overflow:hidden}
.rec .rhd{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:10px 14px;background:var(--panel2);border-bottom:1px solid var(--line2)}
.rec .rid{font-family:var(--mono);font-size:12px;font-weight:600}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--ink)}
.pill.verd{color:#fff;border:none}
.pill.dim{background:color-mix(in srgb,var(--accent) 14%,var(--panel));border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
.pill.dissonant{background:color-mix(in srgb,var(--multi) 14%,var(--panel));border-color:color-mix(in srgb,var(--multi) 40%,var(--line))}
.rec .body{padding:14px}
.blk{margin-bottom:12px}
.blk:last-child{margin-bottom:0}
.blk .bt{margin-bottom:5px}
.blk .txt{font-size:13.5px;line-height:1.55}
.kvs{display:flex;flex-wrap:wrap;gap:6px 8px;margin-top:7px}
.kv{font-family:var(--mono);font-size:11px;background:var(--panel2);border:1px solid var(--line2);border-radius:6px;padding:3px 7px}
.kv b{color:var(--ink);font-weight:600}
.kv span{color:var(--muted)}
.why{font-size:12.5px;color:var(--muted);border-left:2px solid var(--line);padding-left:10px;margin-top:6px}
.probes{display:flex;flex-wrap:wrap;gap:6px;margin-top:7px}
.pb{font-family:var(--mono);font-size:10.5px;border:1px solid var(--line);border-radius:6px;padding:3px 7px}
.pb.on{border-color:color-mix(in srgb,var(--diag) 55%,var(--line));background:color-mix(in srgb,var(--diag) 10%,var(--panel))}
.pb.off{border-style:dashed}
.pb .tag{color:var(--muted)}
.anns{margin-top:8px;border-top:1px solid var(--line2);padding-top:9px}
.ann{font-family:var(--mono);font-size:11px;display:flex;flex-wrap:wrap;gap:8px;padding:4px 0;color:var(--muted)}
.ann b{color:var(--ink)}
.ann.mine{color:var(--accent-ink)}
.hiddenmsg{font-size:12px;color:var(--muted);font-style:italic}

/* blind rating form */
.form{border-top:1px solid var(--line2);margin-top:10px;padding-top:12px}
.form .q{font-size:12.5px;font-weight:600;margin:10px 0 6px}
.opts{display:flex;flex-wrap:wrap;gap:7px}
.opt{font-family:var(--mono);font-size:12px;border:1px solid var(--line);border-radius:8px;padding:6px 11px;background:var(--panel);user-select:none}
.opt:hover{border-color:var(--accent)}
.opt.sel{background:var(--accent);color:#fff;border-color:var(--accent)}
.opt.sev{min-width:38px;text-align:center}
.checkline{display:flex;align-items:center;gap:8px;font-size:12.5px;margin-top:6px}
.checkline input{width:16px;height:16px}
.form .submit{margin-top:14px;display:flex;align-items:center;gap:10px}

/* pending export */
.pending{margin-top:18px;border-color:color-mix(in srgb,var(--accent) 30%,var(--line))}
.pending .hd{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.pending .cnt{font-family:var(--mono);font-size:12px}
.pending textarea{width:100%;min-height:88px;font-family:var(--mono);font-size:11px;margin-top:10px;border:1px solid var(--line);border-radius:8px;padding:9px;background:var(--panel2);color:var(--ink);resize:vertical}
.hint{font-size:11.5px;color:var(--muted);margin-top:6px}

/* drawer */
#scrim[hidden]{display:none}
#scrim{position:fixed;inset:0;background:rgba(8,11,16,.45);z-index:40}
#drawer{position:fixed;top:0;right:0;height:100%;width:min(560px,94vw);background:var(--panel);border-left:1px solid var(--line);z-index:50;transform:translateX(100%);transition:transform .18s ease;display:flex;flex-direction:column}
#drawer.open{transform:none;box-shadow:-8px 0 30px rgba(8,11,16,.2)}
.drawer-head{display:flex;align-items:flex-start;gap:10px;padding:16px 18px;border-bottom:1px solid var(--line)}
.drawer-title{font-size:15px;font-weight:660}
.drawer-sub{font-size:12px;color:var(--muted);margin-top:2px}
.drawer-count{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:4px}
.drawer-head .x{margin-left:auto;border:1px solid var(--line);background:var(--panel);border-radius:8px;width:30px;height:30px;font-size:13px}
.drawer-body{overflow-y:auto;padding:16px 18px}
.empty{color:var(--muted);font-size:13px;text-align:center;padding:30px}
.footer{color:var(--muted);font-size:11.5px;text-align:center;padding:24px 0}
</style>
</head>
<body data-blind="on">
<div class="topbar">
  <div class="wrap">
    <div class="brand">
      <div class="t">Gate 1 · Scenario Admissibility Filter</div>
      <div class="s">dataset <b id="ds-name"></b> · subject model <b id="ds-subj"></b> · <span id="ds-meta"></span></div>
    </div>
    <button id="warnBtn" class="warnchip" hidden></button>
    <button id="themeBtn" class="btn sm" title="Toggle light / dark">◐ theme</button>
    <button id="blindBtn" class="toggle on">Blind mode: ON</button>
  </div>
</div>

<main class="wrap">
  <section id="workspace"></section>
  <section id="analytics"></section>
</main>

<div id="scrim" hidden></div>
<aside id="drawer" aria-label="records"></aside>

<div class="footer wrap">
  JSON is the single source of truth · annotations are append-only · this file is self-contained and opens offline
</div>

<script type="application/json" id="data-scenarios">__SCENARIOS_JSON__</script>
<script type="application/json" id="data-probes">__PROBES_JSON__</script>
<script type="application/json" id="data-annotations">__ANNOTATIONS_JSON__</script>
<script type="application/json" id="data-meta">__META_JSON__</script>

<script>
'use strict';
/* ---------------------------------------------------------------- data --- */
const SCENARIOS = JSON.parse(document.getElementById('data-scenarios').textContent);
const PROBES    = JSON.parse(document.getElementById('data-probes').textContent);
const EMBEDDED_ANN = JSON.parse(document.getElementById('data-annotations').textContent);
const META      = JSON.parse(document.getElementById('data-meta').textContent);
const RECORDS   = SCENARIOS.records;
const PROBE_BY_ID = Object.fromEntries(PROBES.probes.map(p => [p.probe_id, p]));
const BEHAVIORAL  = PROBES.probes.filter(p => p.kind === 'task' || p.kind === 'probe');

const DIMS = ['helplessness','expectation_violation','epistemic_betrayal','trust_rupture','self_directed_agency'];
const DIM_LABEL = {helplessness:'Helplessness',expectation_violation:'Expectation violation',epistemic_betrayal:'Epistemic betrayal',trust_rupture:'Trust rupture',self_directed_agency:'Self-directed agency'};
const VERDICTS = ['fits','fits_via_proxy_probe','reject_no_probe','reject_guardrail','reject_multi_dimension'];
const VERDICT_LABEL = {fits:'Fits',fits_via_proxy_probe:'Fits · proxy',reject_no_probe:'Reject · no probe',reject_guardrail:'Reject · guardrail',reject_multi_dimension:'Reject · multi-dim'};
const VCLASS = {fits:'v-fits',fits_via_proxy_probe:'v-proxy',reject_no_probe:'v-noprobe',reject_guardrail:'v-guard',reject_multi_dimension:'v-multi'};
const VCOLORVAR = {fits:'--fits',fits_via_proxy_probe:'--proxy',reject_no_probe:'--noprobe',reject_guardrail:'--guard',reject_multi_dimension:'--multi'};
const USABLE = new Set(['fits','fits_via_proxy_probe']);
const BLIND_OPTIONS = DIMS.concat(['unclear']);
const PROBE_CODE = {hold_own_judgment:'HOJ',rely_on_partner_info:'RPI',persist_after_obstruction:'PAO',rule_change_midtask:'RCM',recover_after_error:'RAE',coherent_injection_retrieval:'CIR',false_correction_concession:'FCC',free_recall_vs_forced_choice:'FRC',unrelated_prior_retrieval:'UPR',avoidance:'AVD',cue_specific_reactivation:'CSR'};

/* --------------------------------------------------------------- state --- */
const LS_KEY = 'gate1:localAnns:' + (META.dataset || 'gate1');
let localAnns = [];
try { localAnns = JSON.parse(localStorage.getItem(LS_KEY) || '[]'); } catch(e){ localAnns = []; }
function saveLocal(){ try { localStorage.setItem(LS_KEY, JSON.stringify(localAnns)); } catch(e){} }
let blind = true;                 // §1.4 blind mode is the default state
let rater = null;
let wsFilter = 'all';

/* ------------------------------------------------------------- helpers --- */
function h(tag, attrs){
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs){
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.slice(0,2) === 'on' && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  for (let i = 2; i < arguments.length; i++){
    const kid = arguments[i];
    appendKid(e, kid);
  }
  return e;
}
function appendKid(e, kid){
  if (kid == null || kid === false) return;
  if (Array.isArray(kid)){ kid.forEach(k => appendKid(e, k)); return; }
  e.appendChild(kid.nodeType ? kid : document.createTextNode(String(kid)));
}
function clear(n){ while (n.firstChild) n.removeChild(n.firstChild); return n; }
function pct(n, d){ return d ? Math.round(100 * n / d) : 0; }
function cvar(name){ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function usable(r){ return USABLE.has(r.verdict.verdict); }
function rate(recs){ const u = recs.filter(usable).length; return {u:u, n:recs.length, pct:pct(u, recs.length)}; }
function onDiagonal(dim, pid){ const p = PROBE_BY_ID[pid]; return !!(p && (p.predicted_dimensions||[]).includes(dim)); }

function annKey(a){ return a.record_id + '|' + a.rater_id + '|' + a.submitted_at; }
function allAnns(){
  const seen = new Set(), out = [];
  for (const a of EMBEDDED_ANN.concat(localAnns)){ const k = annKey(a); if (!seen.has(k)){ seen.add(k); out.push(a); } }
  return out;
}
function annsFor(rid){ return allAnns().filter(a => a.record_id === rid); }
function hasSubmitted(rid, who){ return allAnns().some(a => a.record_id === rid && a.rater_id === who); }
function recordById(rid){ return RECORDS.find(r => r.meta.record_id === rid); }
function raterIds(){ return Array.from(new Set(allAnns().map(a => a.rater_id))).sort(); }

/* ---------------------------------------------------------------- init --- */
document.getElementById('ds-name').textContent = META.dataset;
document.getElementById('ds-subj').textContent = META.subject_model;
document.getElementById('ds-meta').textContent = META.record_count + ' records · ' + META.annotation_count + ' embedded annotations · built ' + META.generated_at;

(function initWarn(){
  const btn = document.getElementById('warnBtn');
  const ws = META.warnings || [];
  if (!ws.length) return;
  btn.hidden = false;
  btn.textContent = '⚠ ' + ws.length + ' build warning' + (ws.length===1?'':'s');
  btn.addEventListener('click', () => {
    openDrawerNode('Build warnings', ws.length + ' non-fatal warning(s) — the build completed',
      h('div', null, ws.map(w => h('div', {class:'blk'},
        h('div', {class:'label'}, w.code),
        h('div', {class:'txt'}, w.message)))));
  });
})();

document.getElementById('themeBtn').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : (cur === 'light' ? null : 'dark');
  if (next) document.documentElement.setAttribute('data-theme', next);
  else document.documentElement.removeAttribute('data-theme');
});

document.getElementById('blindBtn').addEventListener('click', () => setBlind(!blind));
function setBlind(v){
  if (v === false && blind === true){
    if (!confirm('Revealing analytics shows preset labels (target_dimension, severity_intended, verdict, why, probe_ids).\n\nA rater should not view these before submitting their own blind judgments. Continue to the analyst view?')) return;
  }
  blind = v;
  applyMode();
}

/* ---------------------------------------------------------------- mode --- */
function applyMode(){
  document.body.dataset.blind = blind ? 'on' : 'off';
  const bb = document.getElementById('blindBtn');
  bb.textContent = 'Blind mode: ' + (blind ? 'ON' : 'OFF');
  bb.className = 'toggle ' + (blind ? 'on' : 'off');
  const ws = document.getElementById('workspace');
  const an = document.getElementById('analytics');
  clear(ws); clear(an);
  if (blind){
    renderWorkspace(ws);
    an.appendChild(maskCard());
  } else {
    ws.appendChild(analystBanner());
    renderAnalytics(an);
  }
  window.scrollTo({top:0});
}
function maskCard(){
  return h('section', {class:'section'},
    h('div', {class:'mask'},
      h('div', {class:'lock'}, '🔒'),
      h('h2', null, 'Analytics hidden while blind mode is on'),
      h('p', null, 'The dashboard (usable rate, gate breakdown, grouped bars, coverage matrix, agreement) is derived from preset labels — target_dimension, verdict and severity. Showing it here would let a rater see a record\'s preset before judging it. Rate first; reveal after.'),
      h('button', {class:'btn primary', onclick:() => setBlind(false)}, 'Reveal analytics (turns blind mode off)')));
}
function analystBanner(){
  return h('div', {class:'card pad', style:'display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px'},
    h('div', {class:'label'}, 'Analyst view'),
    h('div', {style:'font-size:12.5px;color:var(--muted)'}, 'Blind mode is off — preset labels are visible. Return to blind mode to collect rater judgments.'),
    h('button', {class:'btn', style:'margin-left:auto', onclick:() => setBlind(true)}, 'Return to blind rating'));
}

/* ====================================================== BLIND WORKSPACE === */
function renderWorkspace(root){
  if (!rater){ root.appendChild(raterGate()); return; }
  const mine = RECORDS.filter(r => hasSubmitted(r.meta.record_id, rater));
  root.appendChild(h('div', {class:'ws-bar'},
    h('div', {class:'who'}, 'Rater: ', h('b', null, rater)),
    h('button', {class:'btn sm', onclick:() => { rater = null; applyMode(); }}, 'change'),
    h('div', {class:'prog'}, 'rated ', h('b', {id:'ws-prog'}, String(mine.length)), ' / ' + RECORDS.length),
    h('div', {class:'filterbtns'},
      fbtn('all', 'all'),
      fbtn('unrated', 'unrated'),
      fbtn('rated', 'rated'))));

  let list = RECORDS;
  if (wsFilter === 'unrated') list = RECORDS.filter(r => !hasSubmitted(r.meta.record_id, rater));
  else if (wsFilter === 'rated') list = mine;

  const holder = h('div', null);
  if (!list.length) holder.appendChild(h('div', {class:'empty'}, 'Nothing to show in this filter.'));
  for (const r of list) holder.appendChild(blindCard(r));
  root.appendChild(holder);
  root.appendChild(pendingSlot());
}
// The export panel only appears once something is staged; an empty slot keeps
// the id stable so an in-place submit can swap it in without a full re-render.
function pendingSlot(){
  return localAnns.length ? pendingPanel() : h('div', {id:'ws-pending'});
}
function fbtn(key, label){
  return h('button', {class:'btn sm' + (wsFilter===key?' active':''), onclick:() => { wsFilter = key; applyMode(); }}, label);
}
function raterGate(){
  const input = h('input', {placeholder:'your rater_id', maxlength:'40'});
  const start = () => { const v = input.value.trim(); if (v){ rater = v; wsFilter = 'unrated'; applyMode(); } };
  input.addEventListener('keydown', e => { if (e.key === 'Enter') start(); });
  const existing = raterIds();
  return h('div', {class:'ws-intro'},
    h('div', {class:'card pad'},
      h('div', {class:'label'}, 'Step 1 — identify the rater'),
      h('h2', {style:'font-size:16px;margin:6px 0'}, 'Blind rating'),
      h('p', {style:'color:var(--muted);font-size:13px;margin:0 0 4px'}, 'You will see only the event text, the appraisal text, and the persona — no labels. Submit your own judgment (dimension, severity, whether it seems to carry more than one dimension). Preset values, and other raters\' annotations, are revealed only after you submit.'),
      h('div', {class:'raterpick'},
        input,
        h('button', {class:'btn primary', onclick:start}, 'Start rating')),
      existing.length ? h('div', {class:'raterpick'},
        h('span', {class:'label'}, 'or continue as'),
        existing.map(id => h('button', {class:'chip', onclick:() => { rater = id; wsFilter = 'unrated'; applyMode(); }}, id))) : null),
    h('div', {class:'card pad'},
      h('div', {class:'label'}, 'Why blind'),
      h('p', {style:'color:var(--muted);font-size:12.5px;margin:6px 0 0'}, 'The preset-vs-blind agreement rate is the taxonomy\'s manipulation check — obtainable before any model is run, at the cost of reading time alone. It only means something if raters never see the preset dimension first. That is why blind mode is the default and the analytics stay locked until you switch out of it.')));
}

function blindCard(rec){
  const m = rec.meta, e = rec.event, a = rec.appraisal, p = rec.persona;
  const submitted = hasSubmitted(m.record_id, rater);
  const card = h('div', {class:'rec', 'data-rid':m.record_id});
  card.appendChild(h('div', {class:'rhd'},
    h('span', {class:'rid'}, m.record_id),
    h('span', {class:'pill'}, injLabel(m.injection_format)),
    submitted ? h('span', {class:'pill', style:'color:var(--fits)'}, '✓ you rated this') : h('span', {class:'label'}, 'awaiting your judgment')));
  const body = h('div', {class:'body'});
  body.appendChild(block('Event', e.text));
  body.appendChild(block('Appraisal', a.text));
  body.appendChild(personaBlock(p));
  if (submitted){
    // §7.4 reveal preset values after submission; §7.5 now show all raters
    body.appendChild(revealBlock(rec));
    body.appendChild(annsBlock(rec, true));
  } else {
    body.appendChild(ratingForm(rec, card));
  }
  card.appendChild(body);
  return card;
}

function ratingForm(rec, card){
  const state = { dim:null, sev:null, multi:false };
  const wrap = h('div', {class:'form'});
  wrap.appendChild(h('div', {class:'q'}, 'Which single dimension does this most carry?'));
  const dimOpts = h('div', {class:'opts'});
  BLIND_OPTIONS.forEach(opt => {
    const o = h('div', {class:'opt', onclick:() => {
      state.dim = opt;
      dimOpts.querySelectorAll('.opt').forEach(x => x.classList.remove('sel'));
      o.classList.add('sel'); refresh();
    }}, opt === 'unclear' ? 'unclear' : DIM_LABEL[opt]);
    dimOpts.appendChild(o);
  });
  wrap.appendChild(dimOpts);

  wrap.appendChild(h('div', {class:'q'}, 'How severe does it read? (1 = mild · 5 = severe)'));
  const sevOpts = h('div', {class:'opts'});
  [1,2,3,4,5].forEach(n => {
    const o = h('div', {class:'opt sev', onclick:() => {
      state.sev = n;
      sevOpts.querySelectorAll('.opt').forEach(x => x.classList.remove('sel'));
      o.classList.add('sel'); refresh();
    }}, String(n));
    sevOpts.appendChild(o);
  });
  wrap.appendChild(sevOpts);

  const chk = h('input', {type:'checkbox', onchange:e => { state.multi = e.target.checked; }});
  wrap.appendChild(h('label', {class:'checkline'}, chk, 'It seems to carry more than one dimension'));

  const btn = h('button', {class:'btn primary', disabled:true, onclick:() => submitRating(rec, state, card)}, 'Submit judgment');
  const note = h('span', {class:'hint'}, 'pick a dimension and a severity to submit');
  wrap.appendChild(h('div', {class:'submit'}, btn, note));
  function refresh(){ const ok = state.dim && state.sev; btn.disabled = !ok; note.textContent = ok ? 'preset values reveal after you submit' : 'pick a dimension and a severity to submit'; }
  return wrap;
}

function submitRating(rec, state, card){
  const now = new Date();
  const submitted_at = now.toISOString().replace(/\.\d+Z$/, 'Z');
  const revealed_at = new Date(now.getTime() + 1000).toISOString().replace(/\.\d+Z$/, 'Z');
  const ann = {
    record_id: rec.meta.record_id,
    rater_id: rater,
    blind_dimension: state.dim,
    blind_severity: state.sev,
    blind_multi_dimension: !!state.multi,
    submitted_at: submitted_at,
    revealed_at: revealed_at
  };
  localAnns.push(ann);
  saveLocal();
  // §7.4 reveal in place: replace only this card so the rater stays put and
  // sees the preset immediately, regardless of the active filter.
  if (card){
    card.replaceWith(blindCard(rec));
    const prog = document.getElementById('ws-prog');
    if (prog) prog.textContent = String(RECORDS.filter(r => hasSubmitted(r.meta.record_id, rater)).length);
    const pend = document.getElementById('ws-pending');
    if (pend) pend.replaceWith(pendingSlot());
  } else {
    applyMode();
  }
}

function revealBlock(rec){
  const a = rec.appraisal, v = rec.verdict, e = rec.event;
  const box = h('div', {class:'blk', style:'border-top:1px solid var(--line2);padding-top:11px'});
  box.appendChild(h('div', {class:'label'}, 'Preset values — revealed after your submission'));
  box.appendChild(h('div', {class:'kvs'},
    kv('target_dimension', DIM_LABEL[a.target_dimension]),
    kv('severity_intended', e.severity_intended),
    kv('agency', a.agency_position),
    kv('valence', a.valence),
    kv('verdict', VERDICT_LABEL[v.verdict])));
  box.appendChild(verdictBlock(rec));
  return box;
}

function pendingPanel(){
  const wrap = h('section', {id:'ws-pending', class:'card pad pending'});
  const jsonl = localAnns.map(a => JSON.stringify(a)).join('\n');
  const ta = h('textarea', {readonly:true}, jsonl);
  wrap.appendChild(h('div', {class:'hd'},
    h('div', {class:'label'}, 'Pending annotations (this browser)'),
    h('div', {class:'cnt'}, localAnns.length + ' unsaved'),
    h('button', {class:'btn sm', onclick:() => { navigator.clipboard && navigator.clipboard.writeText(jsonl); ta.select(); }}, 'Copy'),
    h('button', {class:'btn sm', onclick:downloadPending}, 'Download .jsonl'),
    h('button', {class:'btn sm', onclick:() => { if (confirm('Clear pending annotations from this browser? Do this only after appending them to data/annotations.jsonl and rebuilding.')){ localAnns = []; saveLocal(); applyMode(); } }}, 'Clear')));
  wrap.appendChild(ta);
  wrap.appendChild(h('div', {class:'hint'}, 'Append these lines to ', h('span', {class:'mono'}, 'data/annotations.jsonl'), ' (append-only) and re-run ', h('span', {class:'mono'}, 'python build.py'), '. The JSON is the source of truth; the browser only stages new lines.'));
  return wrap;
}
function downloadPending(){
  const blob = new Blob([localAnns.map(a => JSON.stringify(a)).join('\n') + '\n'], {type:'application/x-ndjson'});
  const url = URL.createObjectURL(blob);
  const a = h('a', {href:url, download:'annotations_new.jsonl'});
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* =========================================================== ANALYTICS === */
function renderAnalytics(root){
  root.appendChild(headerStats());
  root.appendChild(gatesSection());
  root.appendChild(barsSection());
  root.appendChild(matrixSection());
  root.appendChild(agreementSection());
}

/* 6.1 header */
function headerStats(){
  const r = rate(RECORDS);
  const counts = {}; VERDICTS.forEach(v => counts[v] = 0);
  RECORDS.forEach(x => counts[x.verdict.verdict]++);
  const stack = h('div', {class:'stack'});
  VERDICTS.forEach(v => {
    if (!counts[v]) return;
    const seg = h('div', {class:'seg ' + VCLASS[v], style:'width:' + (100*counts[v]/RECORDS.length) + '%',
      title: VERDICT_LABEL[v] + ' — ' + counts[v], onclick:() => drawerForVerdict(v)});
    stack.appendChild(seg);
  });
  const legend = h('div', {class:'legend'});
  VERDICTS.forEach(v => legend.appendChild(
    h('button', {class:'lg', onclick:() => drawerForVerdict(v)},
      h('span', {class:'dot ' + VCLASS[v]}), VERDICT_LABEL[v], h('span', {class:'c'}, counts[v]))));
  return section('Admissibility overview', '6.1',
    'Share of generated material eligible to enter the benchmark. Click any segment or legend chip to list those records.',
    h('div', {class:'card pad'},
      h('div', {class:'stats'},
        h('div', {class:'big'},
          h('div', {class:'num'}, r.pct + '%'),
          h('div', {class:'cap'}, r.u + ' of ' + r.n + ' usable (fits + proxy)')),
        h('div', null, stack, legend))));
}

/* 6.2 gates */
function gatesSection(){
  const gates = [
    {v:'reject_no_probe', id:'Gate A', q:'Is there a probe that can detect it?', ex:'Injury that surfaces only as physiological arousal (startle, sleep disruption, somatic flashback) has no textual carrier.'},
    {v:'reject_guardrail', id:'Gate B', q:'Will the model accept the injection?', ex:'Material that trips a categorical refusal yields no data; refusal rate is recorded per dimension as a dependent variable, not discarded.'},
    {v:'reject_multi_dimension', id:'Gate C', q:'Does it carry a single leading dimension?', ex:'Material with a second independent injury cannot be assigned to a cell, and blind labelling will necessarily fail.'}
  ];
  const grid = h('div', {class:'gates'});
  gates.forEach(g => {
    const recs = RECORDS.filter(r => r.verdict.verdict === g.v);
    const card = h('div', {class:'card pad gate'});
    card.appendChild(h('div', {class:'top'},
      h('span', {class:'gid'}, g.id),
      h('span', {class:'verd ' + VCLASS[g.v]}, g.v)));
    card.appendChild(h('div', {class:'cnt'}, recs.length));
    card.appendChild(h('h3', null, g.q));
    card.appendChild(h('div', {style:'font-size:12px;color:var(--muted)'}, g.ex));
    const egs = h('div', {class:'eglist'});
    recs.slice(0,3).forEach(r => egs.appendChild(
      h('div', {class:'eg', onclick:() => drawerForRecords(r.meta.record_id, [r])},
        h('div', {class:'a'}, truncate(r.appraisal.text, 88)),
        h('div', {class:'e'}, r.meta.record_id + ' · ' + truncate(r.event.text, 66)))));
    card.appendChild(egs);
    if (recs.length > 3) card.appendChild(h('button', {class:'seeall', onclick:() => drawerForVerdict(g.v)}, 'see all ' + recs.length + ' →'));
    else if (recs.length === 0) card.appendChild(h('div', {class:'hint'}, 'none rejected here'));
    grid.appendChild(card);
  });

  // proxy block
  const proxyRecs = RECORDS.filter(r => r.verdict.verdict === 'fits_via_proxy_probe');
  const strictFits = RECORDS.filter(r => r.verdict.verdict === 'fits').length;
  const withProxy = pct(strictFits + proxyRecs.length, RECORDS.length);
  const without = pct(strictFits, RECORDS.length);
  const proxyBox = h('div', {class:'card pad proxy-box'},
    h('div', {class:'label'}, 'Passing state · fits via proxy probe'),
    h('p', {style:'font-size:12.5px;color:var(--muted);margin:6px 0 8px'}, 'Material that fails Gate A directly but has a pure-text structural analogue (narrative breakage, false-correction concession, cue dissociation). Tracked separately — never folded into fits.'),
    h('div', {class:'rate'}, 'Usable rate falls from ', h('b', null, withProxy + '%'), ' ', h('span',{class:'arrow'},'→'), ' ', h('b', null, without + '%'), ' without these ' + proxyRecs.length + ' proxy record(s).'),
    proxyRecs.length ? h('button', {class:'seeall', onclick:() => drawerForRecords('Fits via proxy probe', proxyRecs)}, 'see all ' + proxyRecs.length + ' →') : null);

  return section('The three gates', '6.2',
    'Failing any one gate means the material cannot enter the benchmark — no rewording rescues it. Passing all three = usable.',
    h('div', null, grid, proxyBox));
}

/* 6.3 grouped bars */
function barsSection(){
  const groupings = [
    {t:'target_dimension', groups: DIMS.map(d => ({label:DIM_LABEL[d], test:r => r.appraisal.target_dimension === d}))},
    {t:'agency_position', groups: ['victim','witness','perpetrator'].map(g => ({label:g, test:r => r.appraisal.agency_position === g}))},
    {t:'onset × duration', groups: [['sudden','bounded'],['sudden','persisting'],['insidious','bounded'],['insidious','persisting']].map(([o,d]) => ({label:o+' · '+d, test:r => r.event.onset===o && r.event.duration===d}))},
    {t:'injection_format', groups: ['first_person_memory','second_person_assignment','no_injection'].filter(g => RECORDS.some(r => r.meta.injection_format === g)).map(g => ({label:g, test:r => r.meta.injection_format === g}))},
    {t:'instrument_set', groups: ['adult','adolescent'].map(g => ({label:g, test:r => r.persona.instrument_set === g}))}
  ];
  const holder = h('div', {class:'card pad'});
  groupings.forEach((grp, i) => {
    const g = h('div', {class:'grp'});
    g.appendChild(h('div', {class:'gt label'}, grp.t));
    grp.groups.forEach(gr => {
      const recs = RECORDS.filter(gr.test);
      g.appendChild(groupBar(gr.label, recs));
    });
    holder.appendChild(g);
    if (i < groupings.length - 1) holder.appendChild(h('div', {style:'border-top:1px solid var(--line2);margin:14px 0'}));
  });
  return section('Usable rate & verdict composition by group', '6.3',
    'Each bar is one group; segments are its verdict mix. Click a bar to list that subset.', holder);
}
function groupBar(label, recs){
  const r = rate(recs);
  const counts = {}; VERDICTS.forEach(v => counts[v] = 0);
  recs.forEach(x => counts[x.verdict.verdict]++);
  const track = h('div', {class:'track'});
  VERDICTS.forEach(v => { if (counts[v]) track.appendChild(h('div', {class:'seg ' + VCLASS[v], style:'flex:' + counts[v], title: VERDICT_LABEL[v] + ' — ' + counts[v]})); });
  const row = h('div', {class:'rowbar' + (recs.length?'':' empty'), onclick:() => recs.length && drawerForRecords(label, recs)},
    h('div', {class:'gl', title:label}, label),
    track,
    h('div', {class:'rt'}, h('b', null, r.pct + '%'), ' · n=' + r.n));
  return row;
}

/* 6.4 coverage matrix */
function matrixSection(){
  const cols = BEHAVIORAL;
  const table = h('table', {class:'mx'});
  const thead = h('thead');
  const hrow = h('tr');
  hrow.appendChild(h('th', {class:'corner'}, 'dimension ╲ probe'));
  cols.forEach(p => hrow.appendChild(h('th', {class:'col', title:p.label + '  ·  ' + p.kind + '  ·  predicts: ' + ((p.predicted_dimensions||[]).map(d=>DIM_LABEL[d]).join(', ') || 'none (control)')},
    (PROBE_CODE[p.probe_id]||p.probe_id), h('span', {class:'grp-t'}, p.kind==='task'?'task':'mem'))));
  thead.appendChild(hrow);
  table.appendChild(thead);

  const tbody = h('tbody');
  DIMS.forEach(dim => {
    const tr = h('tr');
    tr.appendChild(h('td', {class:'rowh'}, DIM_LABEL[dim]));
    cols.forEach(p => {
      const recs = RECORDS.filter(r => r.appraisal.target_dimension === dim && (r.verdict.probe_ids||[]).includes(p.probe_id));
      const diag = onDiagonal(dim, p.probe_id);
      let cls = 'cell' + (diag ? ' diag' : '');
      let content = null, style = null;
      if (recs.length === 0){ cls += ' blank'; content = '·'; }
      else {
        const rr = rate(recs);
        if (rr.u === 0){ cls += ' allrej'; content = '✕' + recs.length; }
        else {
          const alpha = 0.14 + 0.6 * rr.pct/100;
          style = 'background:rgba(47,158,68,' + alpha.toFixed(3) + ')';
          content = String(recs.length);
        }
      }
      const td = h('td', {class:cls, style:style,
        title: DIM_LABEL[dim] + ' × ' + p.label + '\n' + recs.length + ' replicate(s) · ' + (recs.length?rate(recs).pct+'% usable':'not sampled') + (diag?' · on-diagonal (predicted)':' · off-diagonal (specificity)'),
        onclick: () => recs.length && drawerForRecords(DIM_LABEL[dim] + ' × ' + p.label, recs)},
        diag ? h('span', {class:'mk'}, '◆') : null, content);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  const legend = h('div', {class:'mx-legend'},
    h('span', {class:'it'}, h('span', {class:'swatch', style:'background:rgba(47,158,68,.5)'}), 'sampled — colour = usable rate, number = replicates'),
    h('span', {class:'it'}, h('span', {class:'swatch blank'}), 'not sampled (blank)'),
    h('span', {class:'it'}, h('span', {class:'swatch allrej'}), 'all rejected (✕)'),
    h('span', {class:'it'}, h('span', {class:'swatch diag'}), '◆ on-diagonal (probe predicts this dimension)'));
  const codes = h('div', {class:'hint', style:'margin-top:8px'}, 'columns: ' + cols.map(p => (PROBE_CODE[p.probe_id]||p.probe_id) + '=' + p.label).join(' · '));

  return section('Coverage matrix — dimension × probe', '6.4',
    'The most important view. Off-diagonal cells are not optional: the specificity test requires them, and blank cells are the reason this view exists. Three states are visually distinct — sampled, not-sampled, all-rejected.',
    h('div', {class:'card pad'}, h('div', {class:'matrix-wrap'}, table), legend, codes));
}

/* §7 agreement panel */
function agreementSection(){
  const anns = allAnns().map(a => ({a:a, rec:recordById(a.record_id)})).filter(x => x.rec);
  const labeled = anns.filter(x => x.a.blind_dimension && x.a.blind_dimension !== 'unclear');
  const cols = DIMS.concat(['unclear']);
  // confusion counts preset(row) x blind(col)
  const cm = {}; DIMS.forEach(d => { cm[d] = {}; cols.forEach(c => cm[d][c] = 0); });
  anns.forEach(x => { const preset = x.rec.appraisal.target_dimension; const b = x.a.blind_dimension; if (cm[preset] && b in cm[preset]) cm[preset][b]++; });

  const table = h('table', {class:'cm'});
  const head = h('tr'); head.appendChild(h('th', null, 'preset ╲ blind'));
  cols.forEach(c => head.appendChild(h('th', {title:c==='unclear'?'unclear':DIM_LABEL[c]}, c==='unclear'?'?':abbr(c))));
  table.appendChild(head);
  DIMS.forEach(d => {
    const tr = h('tr'); tr.appendChild(h('td', {class:'rh', title:DIM_LABEL[d]}, abbr(d)));
    cols.forEach(c => {
      const n = cm[d][c];
      const cls = n === 0 ? 'zero' : (c === d ? 'hit' : (c === 'unclear' ? '' : 'miss'));
      tr.appendChild(h('td', {class:cls}, n || ''));
    });
    table.appendChild(tr);
  });

  // per-dimension agreement
  const arate = h('div', {class:'arate'});
  let totAgree = 0, totN = 0;
  DIMS.forEach(d => {
    const tot = anns.filter(x => x.rec.appraisal.target_dimension === d).length;
    const ag = cm[d][d];
    totAgree += ag; totN += tot;
    const p = pct(ag, tot);
    arate.appendChild(h('div', {class:'r'},
      h('div', {class:'l', title:DIM_LABEL[d]}, abbr(d)),
      h('div', {class:'bar'}, h('i', {style:'width:' + p + '%'})),
      h('div', {class:'p'}, tot ? p + '%' : '—')));
  });

  // confused pairs
  const pairCount = {};
  DIMS.forEach((d1,i) => DIMS.forEach((d2,j) => { if (j>i){ const k = d1+'|'+d2; pairCount[k] = cm[d1][d2] + cm[d2][d1]; } }));
  const topPairs = Object.entries(pairCount).filter(([k,v]) => v>0).sort((a,b) => b[1]-a[1]).slice(0,3);

  // multi-dimension detection
  const multiRecs = RECORDS.filter(r => r.verdict.verdict === 'reject_multi_dimension').map(r => r.meta.record_id);
  const multiAnns = allAnns().filter(a => multiRecs.includes(a.record_id));
  const multiFlagged = multiAnns.filter(a => a.blind_multi_dimension).length;

  const overall = pct(totAgree, totN);
  const left = h('div', null,
    h('div', {class:'label', style:'margin-bottom:8px'}, 'Confusion matrix · preset (row) vs blind (col)'),
    h('div', {style:'overflow-x:auto'}, table),
    topPairs.length ? h('div', {class:'pairs'},
      h('div', {class:'label'}, 'Systematically confused pairs'),
      topPairs.map(([k,v]) => { const [a,b] = k.split('|'); return h('div', {class:'pr'}, h('span', null, DIM_LABEL[a] + '  ⇄  ' + DIM_LABEL[b]), h('span', {class:'mono'}, v + '×')); })) : null);
  const right = h('div', null,
    h('div', {class:'kbig', style:'margin-bottom:12px'}, 'Overall preset↔blind agreement: ', h('b', null, totN ? overall + '%' : '—'), ' ', h('span', {class:'hint'}, '(' + totAgree + '/' + totN + ' labelled annotations)')),
    h('div', {class:'label', style:'margin-bottom:6px'}, 'Per-dimension agreement'),
    arate,
    h('div', {class:'kbig', style:'margin-top:14px'}, 'Multi-dimension detection: ', h('b', null, multiFlagged + '/' + multiAnns.length), ' ', h('span', {class:'hint'}, 'of annotations on preset multi-dimension records flagged >1 dimension')));

  return section('Agreement — the taxonomy\'s manipulation check', '7',
    'Confusion matrix of preset vs blind label, per-dimension agreement, and the dimension pairs raters keep swapping. Obtainable before any model is run, at the cost of reading time alone.',
    h('div', {class:'card pad'}, h('div', {class:'agree-grid'}, left, right)));
}

/* ---------------------------------------------------------- shared UI ---- */
function section(title, n, desc, body){
  return h('section', {class:'section'},
    h('div', {class:'hd'}, h('h2', null, title), h('span', {class:'n'}, '§' + n)),
    desc ? h('div', {class:'desc'}, desc) : null,
    body);
}
function block(title, text){
  return h('div', {class:'blk'}, h('div', {class:'bt label'}, title), h('div', {class:'txt'}, text));
}
function personaBlock(p){
  return h('div', {class:'blk'},
    h('div', {class:'bt label'}, 'Persona'),
    h('div', {class:'kvs'},
      kv('traits', p.traits.join(', ')),
      kv('assessment_age', p.assessment_age),
      kv('instrument_set', p.instrument_set),
      kv('congruent', p.congruent ? 'congruent' : 'dissonant')));
}
function eventKnobs(e){
  return h('div', {class:'kvs'},
    kv('onset', e.onset), kv('duration', e.duration), kv('resolution', e.resolution),
    kv('time_since', e.time_since), kv('recurrence', e.recurrence), kv('event_age', e.event_age),
    kv('severity_intended', e.severity_intended), kv('shift_note', e.shift_note ? 'yes' : 'null'));
}
function verdictBlock(rec){
  const v = rec.verdict, dim = rec.appraisal.target_dimension;
  const probes = h('div', {class:'probes'});
  (v.probe_ids||[]).forEach(pid => {
    const p = PROBE_BY_ID[pid]; const on = onDiagonal(dim, pid);
    probes.appendChild(h('span', {class:'pb ' + (on?'on':'off'), title:(p?p.label:pid) + (on?' · on-diagonal':' · off-diagonal')},
      (PROBE_CODE[pid]||pid), ' ', h('span', {class:'tag'}, on?'◆':'◇')));
  });
  if (!(v.probe_ids||[]).length) probes.appendChild(h('span', {class:'hint'}, 'no probe (Gate A)'));
  return h('div', null,
    h('div', {class:'why'}, v.why),
    probes,
    h('div', {class:'kvs', style:'margin-top:8px'},
      v.proxy_probe ? kv('proxy_probe', v.proxy_probe) : null,
      kv('refusal_rate', v.refusal_rate == null ? 'null' : v.refusal_rate)));
}
function annsBlock(rec, isBlindContext){
  const list = annsFor(rec.meta.record_id);
  const box = h('div', {class:'anns'});
  box.appendChild(h('div', {class:'label'}, 'Annotations (' + list.length + ')'));
  if (!list.length){ box.appendChild(h('div', {class:'hint'}, 'none yet')); return box; }
  list.forEach(a => {
    const mine = isBlindContext && a.rater_id === rater;
    box.appendChild(h('div', {class:'ann' + (mine?' mine':'')},
      h('b', null, a.rater_id),
      'dim=' + a.blind_dimension,
      'sev=' + a.blind_severity,
      'multi=' + a.blind_multi_dimension,
      h('span', {style:'margin-left:auto'}, a.submitted_at)));
  });
  return box;
}
function fullCard(rec){
  const m = rec.meta, e = rec.event, a = rec.appraisal, p = rec.persona, v = rec.verdict;
  const card = h('div', {class:'rec'});
  const vcolor = 'background:var(' + VCOLORVAR[v.verdict] + ')';
  card.appendChild(h('div', {class:'rhd'},
    h('span', {class:'rid'}, m.record_id),
    h('span', {class:'pill verd', style:vcolor}, VERDICT_LABEL[v.verdict]),
    h('span', {class:'pill dim'}, DIM_LABEL[a.target_dimension]),
    h('span', {class:'pill'}, a.agency_position),
    h('span', {class:'pill'}, a.valence),
    p.congruent ? h('span', {class:'pill'}, 'congruent') : h('span', {class:'pill dissonant'}, 'dissonant'),
    h('span', {class:'pill'}, injLabel(m.injection_format))));
  const body = h('div', {class:'body'});
  const evb = block('Event · ' + e.event_id, e.text); evb.appendChild(eventKnobs(e));
  if (e.shift_note) evb.appendChild(h('div', {class:'why', style:'margin-top:7px'}, 'shift: ' + e.shift_note));
  body.appendChild(evb);
  body.appendChild(block('Appraisal', a.text));
  body.appendChild(personaBlock(p));
  body.appendChild(h('div', {class:'blk'},
    h('div', {class:'bt label'}, 'Meta'),
    h('div', {class:'kvs'},
      kv('injection_format', m.injection_format), kv('generator_model', m.generator_model),
      kv('prompt_version', m.generator_prompt_version), kv('scenario_seed', m.scenario_seed),
      kv('persona_seed', m.persona_seed), kv('sampling_seed', m.sampling_seed))));
  const vb = h('div', {class:'blk'}, h('div', {class:'bt label'}, 'Verdict — ' + VERDICT_LABEL[v.verdict]));
  vb.appendChild(verdictBlock(rec));
  body.appendChild(vb);
  body.appendChild(annsBlock(rec, false));
  card.appendChild(body);
  return card;
}
function kv(k, v){ return h('span', {class:'kv'}, h('b', null, k), ' ', h('span', null, String(v))); }
function truncate(s, n){ return s.length > n ? s.slice(0, n-1) + '…' : s; }
function injLabel(f){ return f === 'second_person_assignment' ? 'second-person' : (f === 'no_injection' ? 'no-injection' : 'first-person'); }
function abbr(d){ return {helplessness:'HELP',expectation_violation:'EXP',epistemic_betrayal:'EPI',trust_rupture:'TRUST',self_directed_agency:'AGENCY'}[d] || d; }

/* ------------------------------------------------------------- drawer ---- */
function drawerForVerdict(v){ drawerForRecords(VERDICT_LABEL[v], RECORDS.filter(r => r.verdict.verdict === v)); }
function drawerForRecords(title, recs){
  openDrawerNode(title, recs.length + ' record' + (recs.length===1?'':'s'),
    recs.length ? h('div', null, recs.map(fullCard)) : h('div', {class:'empty'}, 'No records in this subset.'));
}
function openDrawerNode(title, subtitle, node){
  const d = document.getElementById('drawer');
  clear(d);
  d.appendChild(h('div', {class:'drawer-head'},
    h('div', null, h('div', {class:'drawer-title'}, title), subtitle ? h('div', {class:'drawer-count'}, subtitle) : null),
    h('button', {class:'x', title:'close', onclick:closeDrawer}, '✕')));
  d.appendChild(h('div', {class:'drawer-body'}, node));
  document.getElementById('scrim').hidden = false;
  d.classList.add('open');
}
function closeDrawer(){ document.getElementById('drawer').classList.remove('open'); document.getElementById('scrim').hidden = true; }
document.getElementById('scrim').addEventListener('click', closeDrawer);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

/* ---------------------------------------------------------------- boot --- */
applyMode();
</script>
</body>
</html>

"""

if __name__ == "__main__":
    main()
