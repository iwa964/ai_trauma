#!/usr/bin/env python3
"""Gate 1 - Scenario Admissibility Filter: build script (v0.2).

    python gate/build.py                       # reads ../data, writes ../dist/gate1.html
    python gate/build.py <events.json> [out]   # e.g. tests/fixture.json (acceptance run)

Reads   data/events.json (nested), data/probes.json, data/annotations.jsonl
Against gate/schema.json (structural) + the domain rules in the spec (SS8).
Flattens nested events to one record per (event x appraisal); the flattened
record_id is "{event_id}::{appraisal_id}". Writes a single, self-contained,
offline dist/gate1.html.

JSON is the single source of truth; the HTML is only a view. Verdicts are NOT
authored in the seed data (SS3.5) - absent verdict means "pending". On any hard
validation failure the build ABORTS and prints the offending ids; dist/ is left
untouched.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))       # gate1/gate
ROOT = os.path.dirname(HERE)                             # gate1
DATA = os.path.join(ROOT, "data")
DIST = os.path.join(ROOT, "dist")
SCHEMA_PATH = os.path.join(HERE, "schema.json")
PROBES_PATH = os.path.join(DATA, "probes.json")
ANNOTATIONS_PATH = os.path.join(DATA, "annotations.jsonl")
DEFAULT_EVENTS = os.path.join(DATA, "events.json")
DEFAULT_OUT = os.path.join(DIST, "gate1.html")

VERDICT_ENUM = {"fits", "fits_via_proxy_probe", "reject_no_probe", "reject_guardrail", "reject_multi_dimension"}
REJECTS = {"reject_no_probe", "reject_guardrail", "reject_multi_dimension"}
TAGS = ["helpless", "passive-witness", "epistemic-betrayal", "betrayal", "shame-based",
        "self-directed", "expectation-violation", "collective", "guilt-moral-injury", "benign"]
AGENCIES = {"victim", "witness", "learned", "agent", "instrument", "perpetrator"}
BEHAVIORAL_KINDS = {"task"}  # self_report sits outside the distance gradient


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_probes(pdoc):
    """Return a flat probe list [{probe_id, label, kind, distance, ...}] from
    either probes.json shape: the old {"probes": [...]} battery, or the runner's
    turn-scripted {"tasks": [...], "sessions": [...]} instrument. Yields the same
    probe_ids either way, so the gate (coverage matrix, probe_ids validation)
    is agnostic to which file is present."""
    if isinstance(pdoc, dict) and isinstance(pdoc.get("probes"), list):
        return pdoc["probes"]
    out = []
    for t in pdoc.get("tasks", []) or []:
        out.append({"probe_id": t["probe_id"], "label": t.get("label", ""),
                    "kind": t.get("kind", "task"), "distance": t.get("distance"),
                    "predicted_dimensions": []})
    for s in pdoc.get("sessions", []) or []:
        out.append({"probe_id": s["session_id"], "label": s.get("label", ""),
                    "kind": s.get("kind", "instrument"), "distance": None,
                    "predicted_dimensions": []})
    return out


def load_jsonl(path):
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
                raise SystemExit(f"ABORT: {os.path.basename(path)} line {i} is not valid JSON: {e}")
    return rows


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validator (pure stdlib; jsonschema used if installed)
# --------------------------------------------------------------------------- #
def _resolve_ref(root, ref):
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
            return
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
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errors.append((path, f"additional property '{k}' is not allowed (rejects migrated/legacy fields such as target_dimension)"))
        for k, sub in props.items():
            if k in value:
                _validate_node(value[k], sub, root, f"{path}.{k}", errors)
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append((path, f"array shorter than minItems {schema['minItems']}"))
        if "items" in schema:
            for i, item in enumerate(value):
                _validate_node(item, schema["items"], root, f"{path}[{i}]", errors)


def structural_errors(doc, schema):
    """Validate the events document; attribute errors to event_ids where possible."""
    try:
        import jsonschema  # type: ignore
        out = []
        for err in jsonschema.Draft7Validator(schema).iter_errors(doc):
            p = list(err.absolute_path)
            scope = "<doc>"
            if len(p) >= 2 and p[0] == "events" and isinstance(p[1], int):
                try:
                    scope = doc["events"][p[1]].get("event_id", f"events[{p[1]}]")
                except Exception:
                    scope = f"events[{p[1]}]"
            out.append((scope, err.message))
        return out
    except ImportError:
        pass
    errors = []
    if not isinstance(doc, dict):
        return [("<doc>", "top-level value must be an object")]
    for req in schema.get("required", []):
        if req not in doc:
            errors.append(("<doc>", f"missing required property '{req}'"))
    if "meta" in doc:
        me = []
        _validate_node(doc["meta"], schema["properties"]["meta"], schema, "meta", me)
        errors += [("<meta>", f"{p}: {m}") for p, m in me]
    events = doc.get("events")
    if not isinstance(events, list) or not events:
        errors.append(("<doc>", "'events' must be a non-empty array"))
        return errors
    ev_schema = schema["$defs"]["event"]
    for e in events:
        eid = e.get("event_id", "<no event_id>") if isinstance(e, dict) else "<bad event>"
        ee = []
        _validate_node(e, ev_schema, schema, "event", ee)
        errors += [(eid, f"{p}: {m}") for p, m in ee]
    return errors


# --------------------------------------------------------------------------- #
# Flatten
# --------------------------------------------------------------------------- #
EVENT_FIELDS = ("event_id", "label", "source_row", "threshold", "text", "exposure", "onset",
                "duration", "resolution", "time_since", "recurrence", "event_age",
                "severity_intended", "modifiers", "guardrail_risk", "shift_note")
APPRAISAL_FIELDS = ("appraisal_id", "text", "provisional_tag", "agency_position", "valence")


def flatten(doc):
    """Flatten nested events to one record per (event x appraisal).

    Tolerant of malformed input (missing ids, wrong types) so that structural
    validation reports the offending ids with the documented build-abort, rather
    than this raising a KeyError. Only whitelisted event / appraisal fields are
    copied, so no legacy field (e.g. a migrated target_dimension) can leak into a
    record even if validation were bypassed."""
    m = doc.get("meta")
    injection_format = m.get("injection_format", "") if isinstance(m, dict) else ""
    records = []
    for e in doc.get("events", []) or []:
        if not isinstance(e, dict):
            continue
        eid = e.get("event_id", "?")
        ev = {k: e[k] for k in EVENT_FIELDS if k in e}
        for a in e.get("appraisals", []) or []:
            if not isinstance(a, dict):
                continue
            apid = a.get("appraisal_id", "?")
            appraisal = {k: a[k] for k in APPRAISAL_FIELDS if k in a}
            records.append({
                "record_id": f"{eid}::{apid}",
                "event": ev,
                "appraisal": appraisal,
                "verdict": a.get("verdict"),
                "injection_format": injection_format,
            })
    return records


# --------------------------------------------------------------------------- #
# Domain validation (spec SS8)
# --------------------------------------------------------------------------- #
_NUMWORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand",
             "once", "twice", "dozen"}


def new_fact_tokens(appraisal_text, event_text):
    """Heuristic for the no-new-facts rule: numerals / number-words / mid-sentence
    proper nouns present in the appraisal but absent from the event text."""
    ev = event_text.lower()
    flagged = []
    for num in re.findall(r"\d+", appraisal_text):
        if num not in ev:
            flagged.append(num)
    # sentence-initial words are excluded (capitalisation there is grammatical)
    tokens = re.findall(r"[A-Za-z']+|[.!?]", appraisal_text)
    at_start = True
    for tok in tokens:
        if tok in ".!?":
            at_start = True
            continue
        low = tok.lower().strip("'")
        if not at_start:
            if tok[:1].isupper() and low not in ("i",) and low.isalpha() and len(low) > 1 and low not in ev:
                flagged.append(tok)
            if low in _NUMWORDS and low not in ev:
                flagged.append(tok)
        at_start = False
    return sorted(set(flagged))


def domain_validation(doc, records, probes, skip=None):
    skip = skip or set()
    abort, warnings = [], []
    probe_list = normalize_probes(probes)
    probe_ids = {p["probe_id"] for p in probe_list}
    meta = doc.get("meta", {})

    # duplicate appraisal_id (global) -> abort
    seen_ap = {}
    for e in doc["events"]:
        for a in e.get("appraisals", []):
            ap = a.get("appraisal_id")
            if ap in seen_ap:
                abort.append((e.get("event_id", "?"), f"duplicate appraisal_id '{ap}' (also in {seen_ap[ap]})"))
            else:
                seen_ap[ap] = e.get("event_id", "?")

    # meta.provenance model_generated must differ from subject model -> abort
    prov = str(meta.get("provenance", ""))
    if prov.startswith("model_generated"):
        gen, subj = meta.get("generator_model"), meta.get("subject_model")
        if gen is not None and gen == subj:
            abort.append(("<meta>", f"provenance is model_generated and generator_model equals subject_model ('{gen}')"))

    # per-record verdict checks -> abort
    for r in records:
        rid = r["record_id"]
        if rid in skip:
            continue
        v = r.get("verdict")
        if v is None:
            continue
        vv = v.get("verdict")
        if vv not in VERDICT_ENUM:
            abort.append((rid, f"verdict '{vv}' is outside the enum"))
        if vv == "fits_via_proxy_probe" and not v.get("proxy_probe"):
            abort.append((rid, "verdict is fits_via_proxy_probe but proxy_probe is empty"))
        for pid in v.get("probe_ids", []) or []:
            if pid not in probe_ids:
                abort.append((rid, f"probe_ids references unknown probe '{pid}'"))
        pp = v.get("proxy_probe")
        if pp and pp not in probe_ids:
            abort.append((rid, f"proxy_probe references unknown probe '{pp}'"))

    # ---- warnings ----
    # benign control per event (may be legitimate to lack one - flag, do not force)
    for e in doc["events"]:
        if not any(a.get("valence") == "benign" for a in e.get("appraisals", [])):
            warnings.append({"level": "warn", "code": "no_benign_control",
                             "message": f"event '{e.get('event_id')}' has no valence:benign control - may be legitimate for severe events (matched-control power is lost where effects are largest)"})

    # (tag x behavioral probe) cell with zero records -> summary
    behavioral = [p for p in probe_list if p.get("kind") in BEHAVIORAL_KINDS]
    cell = Counter()
    for r in records:
        v = r.get("verdict")
        if not v:
            continue
        for pid in v.get("probe_ids", []) or []:
            cell[(r["appraisal"]["provisional_tag"], pid)] += 1
    blank = sum(1 for t in TAGS for p in behavioral if cell[(t, p["probe_id"])] == 0)
    total = len(TAGS) * len(behavioral)
    if blank:
        warnings.append({"level": "warn", "code": "zero_measurement_cells",
                         "message": f"{blank}/{total} (tag x probe) measurement cells have zero records - expected while verdicts are unadjudicated; see coverage matrix (a)"})

    # empty onset x duration cell -> warn each
    od = Counter((r["event"]["onset"], r["event"]["duration"]) for r in records)
    for o in ("sudden", "insidious"):
        for d in ("bounded", "persisting"):
            if od[(o, d)] == 0:
                warnings.append({"level": "warn", "code": "empty_onset_duration_cell",
                                 "message": f"onset x duration cell '{o} x {d}' is empty - needs one deliberate authoring attempt before concluding it is rare by nature"})

    # guardrail_risk vs measured refusal_rate disagreement -> warn
    order = {"low": 0, "moderate": 1, "high": 2}
    for r in records:
        v = r.get("verdict")
        rr = v.get("refusal_rate") if v else None
        if isinstance(rr, (int, float)):
            gr = r["event"].get("guardrail_risk")
            if gr == "low" and rr >= 0.4:
                warnings.append({"level": "warn", "code": "guardrail_refusal_mismatch",
                                 "message": f"{r['record_id']}: guardrail_risk 'low' but measured refusal_rate {rr:.2f}"})
            elif gr == "high" and rr <= 0.1:
                warnings.append({"level": "warn", "code": "guardrail_refusal_mismatch",
                                 "message": f"{r['record_id']}: guardrail_risk 'high' but measured refusal_rate {rr:.2f}"})

    # appraisal introduces a proper noun / numeral absent from event.text -> warn
    for r in records:
        toks = new_fact_tokens(r["appraisal"]["text"], r["event"]["text"])
        if toks:
            warnings.append({"level": "warn", "code": "possible_new_fact",
                             "message": f"{r['record_id']}: appraisal token(s) not in event.text: {', '.join(toks)} (heuristic - verify no facts were introduced)"})

    return abort, warnings


def ascii_errors(paths):
    """Spec SS1.5 / SS8: data files must be pure ASCII."""
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        b = open(p, "rb").read()
        bad = [i for i, byte in enumerate(b) if byte > 127]
        if bad:
            out.append((os.path.basename(p), f"non-ASCII byte(s) at offset(s) {bad[:5]}{' ...' if len(bad) > 5 else ''}"))
    return out


# --------------------------------------------------------------------------- #
# HTML generation
# --------------------------------------------------------------------------- #
def _embed(obj):
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def build_html(doc, records, probes, annotations, warnings):
    counts = Counter((r["verdict"]["verdict"] if r.get("verdict") else "pending") for r in records)
    meta = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "version": doc.get("meta", {}).get("version", "0.2"),
        "provenance": doc.get("meta", {}).get("provenance", ""),
        "injection_format": doc.get("meta", {}).get("injection_format", ""),
        "record_count": len(records),
        "event_count": len(doc.get("events", [])),
        "events_deferred": doc.get("events_deferred", []),
        "warnings": warnings,
        "counts": dict(counts),
    }
    html = HTML_TEMPLATE
    html = html.replace("__RECORDS_JSON__", _embed(records))
    html = html.replace("__PROBES_JSON__", _embed(normalize_probes(probes)))
    html = html.replace("__ANNOTATIONS_JSON__", _embed(annotations))
    html = html.replace("__META_JSON__", _embed(meta))
    return html


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    events_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVENTS
    out_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    print("Gate 1 build (v0.2)")
    print("-" * 62)
    print(f"events:      {os.path.relpath(events_path, ROOT)}")

    schema = load_json(SCHEMA_PATH)
    doc = load_json(events_path)
    probes = load_json(PROBES_PATH)
    annotations = load_jsonl(ANNOTATIONS_PATH)

    # Validate structure BEFORE flattening: a malformed document must abort with
    # the offending ids, not raise mid-flatten. flatten() is tolerant regardless.
    a_errors = list(ascii_errors([events_path, PROBES_PATH, ANNOTATIONS_PATH]))
    s_errors = structural_errors(doc, schema)
    records = flatten(doc)

    print(f"events:      {len(doc.get('events', []))}  ->  records: {len(records)}")
    print(f"probes:      {len(probes.get('probes', []))}   annotations: {len(annotations)} (append-only)")
    print("-" * 62)

    # domain checks (records are safe to use; flatten tolerated any malformation)
    d_abort, warnings = ([], [])
    if not any(e[0] == "<doc>" for e in s_errors):
        d_abort, warnings = domain_validation(doc, records, probes, skip={rid for rid, _ in s_errors})

    abort = a_errors + s_errors + d_abort
    if abort:
        print("BUILD ABORTED - validation failed.\n")
        by_id = defaultdict(list)
        for rid, msg in abort:
            by_id[rid].append(msg)
        print("Offending ids:")
        for rid in sorted(by_id):
            print(f"  {rid}")
            for msg in by_id[rid]:
                print(f"      - {msg}")
        print(f"\n{len(abort)} error(s). dist/ was not written.")
        sys.exit(1)

    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [warn:{w['code']}] {w['message']}")
    else:
        print("no warnings.")
    print("-" * 62)

    html = build_html(doc, records, probes, annotations, warnings)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(out_path, ROOT)}  ({os.path.getsize(out_path)/1024:.0f} KB, self-contained, offline)")


HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gate 1 - Scenario Admissibility Filter</title>
<style>
:root{
  --bg:#f5f6f8; --panel:#ffffff; --panel2:#fafbfc; --ink:#181b20; --muted:#697280;
  --line:#e5e8ec; --line2:#eef1f4; --accent:#3b5bdb; --accent-ink:#2b3fa8;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --sans: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --fits:#2f9e44; --proxy:#0c8599; --noprobe:#e8890c; --guard:#e03131; --multi:#7048e8; --pending:#9aa4b2;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.08);
  --radius:12px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d1016; --panel:#151a21; --panel2:#111620; --ink:#e6e9ee; --muted:#96a0ae;
    --line:#242b35; --line2:#1c222b; --accent:#7aa2ff; --accent-ink:#a9c2ff;
    --fits:#51cf66; --proxy:#3bc9db; --noprobe:#ffa94d; --guard:#ff6b6b; --multi:#9775fa; --pending:#5b6675;
    --shadow:none;
  }
}
html[data-theme="light"]{
  --bg:#f5f6f8; --panel:#ffffff; --panel2:#fafbfc; --ink:#181b20; --muted:#697280;
  --line:#e5e8ec; --line2:#eef1f4; --accent:#3b5bdb; --accent-ink:#2b3fa8;
  --fits:#2f9e44; --proxy:#0c8599; --noprobe:#e8890c; --guard:#e03131; --multi:#7048e8; --pending:#9aa4b2;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.08);
}
html[data-theme="dark"]{
  --bg:#0d1016; --panel:#151a21; --panel2:#111620; --ink:#e6e9ee; --muted:#96a0ae;
  --line:#242b35; --line2:#1c222b; --accent:#7aa2ff; --accent-ink:#a9c2ff;
  --fits:#51cf66; --proxy:#3bc9db; --noprobe:#ffa94d; --guard:#ff6b6b; --multi:#9775fa; --pending:#5b6675;
  --shadow:none;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.mono{font-family:var(--mono)}
.label{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.wrap{max-width:1200px;margin:0 auto;padding:0 20px}
h1,h2,h3{margin:0;font-weight:650}
button{font-family:inherit;cursor:pointer}
.btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 11px;font-size:12.5px}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn.primary:disabled{opacity:.45;cursor:not-allowed}
.btn.sm{padding:3px 8px;font-size:11.5px}

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

main{padding:22px 0 90px}
.section{margin:26px 0}
.section > .hd{display:flex;align-items:baseline;gap:10px;margin-bottom:12px}
.section > .hd h2{font-size:15px}
.section > .hd .n{font-family:var(--mono);font-size:11px;color:var(--muted)}
.section .desc{font-size:12.5px;color:var(--muted);max-width:74ch;margin:-4px 0 12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.pad{padding:16px}

.mask{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:12px;padding:46px 20px;border:1px dashed var(--line);border-radius:var(--radius);background:var(--panel2)}
.mask .lock{font-size:26px}
.mask h2{font-size:16px}
.mask p{max-width:60ch;color:var(--muted);font-size:13px;margin:0}

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
.v-fits{background:var(--fits)} .v-proxy{background:var(--proxy)} .v-noprobe{background:var(--noprobe)} .v-guard{background:var(--guard)} .v-multi{background:var(--multi)} .v-pending{background:var(--pending)}

.gates{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:860px){.gates{grid-template-columns:1fr}}
.gate .top{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:2px}
.gate .gid{font-family:var(--mono);font-size:11px;color:var(--muted)}
.gate .cnt{font-size:26px;font-weight:700}
.gate .verd{font-family:var(--mono);font-size:11px;padding:2px 7px;border-radius:6px;color:#fff}
.gate h3{font-size:13.5px;margin:6px 0 2px}
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

.grp{margin-bottom:18px}
.grp .gt{margin-bottom:8px;display:flex;gap:8px;align-items:baseline}
.rowbar{display:grid;grid-template-columns:160px 1fr 150px;gap:12px;align-items:center;padding:5px 0;cursor:pointer;border-radius:8px}
.rowbar:hover{background:var(--panel2)}
@media(max-width:640px){.rowbar{grid-template-columns:104px 1fr 96px;gap:8px}}
.rowbar .gl{font-size:12.5px;font-family:var(--mono);color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rowbar .track{display:flex;height:20px;border-radius:6px;overflow:hidden;background:var(--line2);border:1px solid var(--line)}
.rowbar .track .seg{min-width:0}
.rowbar .rt{font-family:var(--mono);font-size:11.5px;text-align:right;color:var(--muted)}
.rowbar .rt b{color:var(--ink)}
.rowbar.empty .track{opacity:.5}

.matrix-wrap{overflow-x:auto}
table.mx{border-collapse:separate;border-spacing:0;font-size:12px}
table.mx .corner{min-width:140px;text-align:left;padding:6px 8px;color:var(--muted);font-family:var(--mono);font-size:10px;vertical-align:bottom}
table.mx .col{padding:6px 4px;text-align:center;font-family:var(--mono);font-size:10px;color:var(--muted);border-bottom:1px solid var(--line);white-space:nowrap}
table.mx .col small{display:block;font-size:8px;letter-spacing:.04em;opacity:.8}
table.mx .rowh{padding:6px 10px 6px 8px;text-align:right;font-family:var(--mono);font-size:10.5px;white-space:nowrap;border-right:1px solid var(--line)}
td.cell{width:42px;height:38px;text-align:center;vertical-align:middle;border:1px solid var(--line2);position:relative;cursor:pointer;font-family:var(--mono);font-weight:600}
td.cell:hover{outline:2px solid var(--accent);outline-offset:-2px;z-index:2}
td.cell.blank{cursor:default;color:var(--muted);border-style:dashed;background:repeating-linear-gradient(45deg,transparent,transparent 5px,var(--line2) 5px,var(--line2) 6px)}
td.cell.blank:hover{outline:none}
td.cell.allrej{color:var(--guard);box-shadow:inset 3px 0 0 var(--guard)}
td.cell.pend{color:var(--muted)}
.mx-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;font-size:11.5px;color:var(--muted)}
.mx-legend .it{display:inline-flex;align-items:center;gap:6px}
.swatch{width:16px;height:16px;border-radius:3px;border:1px solid var(--line);display:inline-block}
.swatch.blank{border-style:dashed;background:repeating-linear-gradient(45deg,transparent,transparent 3px,var(--line2) 3px,var(--line2) 4px)}
.swatch.allrej{box-shadow:inset 3px 0 0 var(--guard)}
.swatch.pend{background:color-mix(in srgb,var(--pending) 40%,transparent)}
.mxblock{margin-top:8px}
.mxblock h3{font-size:12.5px;margin:14px 0 6px}

.agree-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
@media(max-width:860px){.agree-grid{grid-template-columns:1fr}}
table.cm{border-collapse:separate;border-spacing:0;font-size:11px;width:100%}
table.cm th,table.cm td{padding:5px 6px;text-align:center;border:1px solid var(--line2)}
table.cm th{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.02em}
table.cm td.rh{text-align:right;font-family:var(--mono);font-size:9.5px;color:var(--muted);border-right:1px solid var(--line)}
table.cm td.hit{background:color-mix(in srgb,var(--fits) 22%,transparent);font-weight:700}
table.cm td.miss{background:color-mix(in srgb,var(--guard) 16%,transparent)}
table.cm td.zero{color:color-mix(in srgb,var(--muted) 55%,transparent)}
.arate{display:flex;flex-direction:column;gap:6px}
.arate .r{display:grid;grid-template-columns:120px 1fr 42px;gap:8px;align-items:center}
.arate .r .l{font-family:var(--mono);font-size:10.5px}
.arate .r .bar{height:8px;border-radius:5px;background:var(--line2);overflow:hidden}
.arate .r .bar > i{display:block;height:100%;background:var(--fits)}
.arate .r .p{font-family:var(--mono);font-size:11px;text-align:right}
.pairs{margin-top:10px;font-size:12px}
.pairs .pr{display:flex;justify-content:space-between;border-top:1px solid var(--line2);padding:5px 0}
.kbig{font-size:12.5px}
.kbig b{font-size:15px}

.deferred-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
@media(max-width:720px){.deferred-grid{grid-template-columns:1fr}}
.defcard{border:1px solid var(--line);border-left:3px solid var(--noprobe);border-radius:8px;padding:11px 13px;background:var(--panel2)}
.defcard .sr{font-weight:650;font-size:13px}
.defcard .rs{font-family:var(--mono);font-size:10.5px;color:var(--noprobe);margin:3px 0 6px}
.defcard .nt{font-size:12px;color:var(--muted)}

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

.rec{border:1px solid var(--line);border-radius:var(--radius);background:var(--panel);margin-bottom:12px;overflow:hidden}
.rec .rhd{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:10px 14px;background:var(--panel2);border-bottom:1px solid var(--line2)}
.rec .rid{font-family:var(--mono);font-size:12px;font-weight:600}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--ink)}
.pill.verd{color:#fff;border:none}
.pill.tag{background:color-mix(in srgb,var(--accent) 14%,var(--panel));border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
.pill.pend{background:color-mix(in srgb,var(--pending) 22%,var(--panel));border-color:color-mix(in srgb,var(--pending) 45%,var(--line));color:var(--muted)}
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
.anns{margin-top:8px;border-top:1px solid var(--line2);padding-top:9px}
.ann{font-family:var(--mono);font-size:11px;display:flex;flex-wrap:wrap;gap:8px;padding:4px 0;color:var(--muted)}
.ann b{color:var(--ink)}
.ann.mine{color:var(--accent-ink)}

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
.pending-panel{margin-top:18px;border-color:color-mix(in srgb,var(--accent) 30%,var(--line))}
.pending-panel .hd{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.pending-panel .cnt{font-family:var(--mono);font-size:12px}
.pending-panel textarea{width:100%;min-height:84px;font-family:var(--mono);font-size:11px;margin-top:10px;border:1px solid var(--line);border-radius:8px;padding:9px;background:var(--panel2);color:var(--ink);resize:vertical}
.hint{font-size:11.5px;color:var(--muted);margin-top:6px}

#scrim[hidden]{display:none}
#scrim{position:fixed;inset:0;background:rgba(8,11,16,.45);z-index:40}
#drawer{position:fixed;top:0;right:0;height:100%;width:min(580px,94vw);background:var(--panel);border-left:1px solid var(--line);z-index:50;transform:translateX(100%);transition:transform .18s ease;display:flex;flex-direction:column}
#drawer.open{transform:none;box-shadow:-8px 0 30px rgba(8,11,16,.2)}
.drawer-head{display:flex;align-items:flex-start;gap:10px;padding:16px 18px;border-bottom:1px solid var(--line)}
.drawer-title{font-size:15px;font-weight:660}
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
      <div class="t">Gate 1 - Scenario Admissibility Filter</div>
      <div class="s">v<b id="ds-ver"></b> &middot; <span id="ds-meta"></span></div>
    </div>
    <button id="warnBtn" class="warnchip" hidden></button>
    <button id="themeBtn" class="btn sm" title="Toggle light / dark">theme</button>
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
  JSON is the single source of truth &middot; annotations are append-only &middot; verdicts are produced by running the gate, not authored with the material &middot; self-contained, opens offline
</div>

<script type="application/json" id="data-records">__RECORDS_JSON__</script>
<script type="application/json" id="data-probes">__PROBES_JSON__</script>
<script type="application/json" id="data-annotations">__ANNOTATIONS_JSON__</script>
<script type="application/json" id="data-meta">__META_JSON__</script>

<script>
'use strict';
const RECORDS = JSON.parse(document.getElementById('data-records').textContent);
const PROBES  = JSON.parse(document.getElementById('data-probes').textContent);
const EMBEDDED_ANN = JSON.parse(document.getElementById('data-annotations').textContent);
const META    = JSON.parse(document.getElementById('data-meta').textContent);
const PROBE_BY_ID = Object.fromEntries(PROBES.map(p => [p.probe_id, p]));

const TAGS = ['helpless','passive-witness','epistemic-betrayal','betrayal','shame-based','self-directed','expectation-violation','collective','guilt-moral-injury','benign'];
const TAG_SHORT = {'helpless':'HELP','passive-witness':'P.WIT','epistemic-betrayal':'EPIST','betrayal':'BETR','shame-based':'SHAME','self-directed':'SELF','expectation-violation':'EXPECT','collective':'COLL','guilt-moral-injury':'GUILT','benign':'BENIGN'};
const EXPOSURES = ['direct','witness','learned','indirect','perpetrator'];
const AGENCIES = ['victim','witness','learned','agent','instrument','perpetrator'];
const VERDICTS = ['fits','fits_via_proxy_probe','reject_no_probe','reject_guardrail','reject_multi_dimension'];
const STATES = VERDICTS.concat(['pending']);
const VERDICT_LABEL = {'fits':'Fits','fits_via_proxy_probe':'Fits - proxy','reject_no_probe':'Reject - no probe','reject_guardrail':'Reject - guardrail','reject_multi_dimension':'Reject - multi-injury','pending':'Pending'};
const VCLASS = {'fits':'v-fits','fits_via_proxy_probe':'v-proxy','reject_no_probe':'v-noprobe','reject_guardrail':'v-guard','reject_multi_dimension':'v-multi','pending':'v-pending'};
const VVAR = {'fits':'--fits','fits_via_proxy_probe':'--proxy','reject_no_probe':'--noprobe','reject_guardrail':'--guard','reject_multi_dimension':'--multi','pending':'--pending'};
const USABLE = new Set(['fits','fits_via_proxy_probe']);
const REJECTS = new Set(['reject_no_probe','reject_guardrail','reject_multi_dimension']);
const BLIND_OPTIONS = TAGS.concat(['unclear']);

const LS_KEY = 'gate1v2:localAnns:' + (META.version || 'v');
let localAnns = [];
try { localAnns = JSON.parse(localStorage.getItem(LS_KEY) || '[]'); } catch(e){ localAnns = []; }
function saveLocal(){ try { localStorage.setItem(LS_KEY, JSON.stringify(localAnns)); } catch(e){} }
let blind = true;
let rater = null;
let wsFilter = 'unrated';

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
  for (let i = 2; i < arguments.length; i++) appendKid(e, arguments[i]);
  return e;
}
function appendKid(e, kid){
  if (kid == null || kid === false) return;
  if (Array.isArray(kid)){ kid.forEach(k => appendKid(e, k)); return; }
  e.appendChild(kid.nodeType ? kid : document.createTextNode(String(kid)));
}
function clear(n){ while (n.firstChild) n.removeChild(n.firstChild); return n; }
function pct(n, d){ return d ? Math.round(100 * n / d) : 0; }
function vof(r){ return r.verdict ? r.verdict.verdict : 'pending'; }
function usable(r){ return USABLE.has(vof(r)); }
function isReject(r){ return REJECTS.has(vof(r)); }
function tagShort(t){ return TAG_SHORT[t] || t; }

function annKey(a){ return a.record_id + '|' + a.rater_id + '|' + a.submitted_at; }
function allAnns(){
  const seen = new Set(), out = [];
  for (const a of EMBEDDED_ANN.concat(localAnns)){ const k = annKey(a); if (!seen.has(k)){ seen.add(k); out.push(a); } }
  return out;
}
function annsFor(rid){ return allAnns().filter(a => a.record_id === rid); }
function hasSubmitted(rid, who){ return allAnns().some(a => a.record_id === rid && a.rater_id === who); }
function recordById(rid){ return RECORDS.find(r => r.record_id === rid); }
function raterIds(){ return Array.from(new Set(allAnns().map(a => a.rater_id))).sort(); }

document.getElementById('ds-ver').textContent = META.version || '?';
document.getElementById('ds-meta').textContent = META.record_count + ' records / ' + META.event_count + ' events - ' + META.provenance + ' - ' + META.injection_format + ' - built ' + META.generated_at;

(function initWarn(){
  const btn = document.getElementById('warnBtn');
  const ws = META.warnings || [];
  if (!ws.length) return;
  btn.hidden = false;
  btn.textContent = String(ws.length) + ' build warning' + (ws.length===1?'':'s');
  btn.addEventListener('click', () => openDrawerNode('Build warnings', ws.length + ' non-fatal warning(s) - build completed',
    h('div', null, ws.map(w => h('div', {class:'blk'}, h('div', {class:'label'}, w.code), h('div', {class:'txt'}, w.message))))));
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
    if (!confirm('Revealing analytics shows preset labels (provisional_tag, severity_intended, verdict, why, probe_ids).\n\nA rater should not view these before submitting their own blind judgements. Continue to the analyst view?')) return;
  }
  blind = v; applyMode();
}

function applyMode(){
  document.body.dataset.blind = blind ? 'on' : 'off';
  const bb = document.getElementById('blindBtn');
  bb.textContent = 'Blind mode: ' + (blind ? 'ON' : 'OFF');
  bb.className = 'toggle ' + (blind ? 'on' : 'off');
  const ws = document.getElementById('workspace');
  const an = document.getElementById('analytics');
  clear(ws); clear(an);
  if (blind){ renderWorkspace(ws); an.appendChild(maskCard()); }
  else { ws.appendChild(analystBanner()); renderAnalytics(an); }
  window.scrollTo({top:0});
}
function maskCard(){
  return h('section', {class:'section'},
    h('div', {class:'mask'},
      h('div', {class:'lock'}, 'BLIND'),
      h('h2', null, 'Analytics hidden while blind mode is on'),
      h('p', null, 'The dashboard is keyed on preset labels - provisional_tag, severity, verdict. Showing it would let a rater see a record\'s preset before judging it. Rate first; reveal after. The agreement rate this produces is the manipulation check for the whole tag scheme.'),
      h('button', {class:'btn primary', onclick:() => setBlind(false)}, 'Reveal analytics (turns blind mode off)')));
}
function analystBanner(){
  return h('div', {class:'card pad', style:'display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px'},
    h('div', {class:'label'}, 'Analyst view'),
    h('div', {style:'font-size:12.5px;color:var(--muted)'}, 'Blind mode is off - preset labels are visible. Return to blind mode to collect rater judgements.'),
    h('button', {class:'btn', style:'margin-left:auto', onclick:() => setBlind(true)}, 'Return to blind rating'));
}

/* ---- blind workspace ---- */
function renderWorkspace(root){
  if (!rater){ root.appendChild(raterGate()); return; }
  const mine = RECORDS.filter(r => hasSubmitted(r.record_id, rater));
  root.appendChild(h('div', {class:'ws-bar'},
    h('div', {class:'who'}, 'Rater: ', h('b', null, rater)),
    h('button', {class:'btn sm', onclick:() => { rater = null; applyMode(); }}, 'change'),
    h('div', {class:'prog'}, 'rated ', h('b', {id:'ws-prog'}, String(mine.length)), ' / ' + RECORDS.length),
    h('div', {class:'filterbtns'}, fbtn('all','all'), fbtn('unrated','unrated'), fbtn('rated','rated'))));
  let list = RECORDS;
  if (wsFilter === 'unrated') list = RECORDS.filter(r => !hasSubmitted(r.record_id, rater));
  else if (wsFilter === 'rated') list = mine;
  const holder = h('div', null);
  if (!list.length) holder.appendChild(h('div', {class:'empty'}, 'Nothing to show in this filter.'));
  for (const r of list) holder.appendChild(blindCard(r));
  root.appendChild(holder);
  root.appendChild(pendingSlot());
}
function fbtn(key, label){ return h('button', {class:'btn sm' + (wsFilter===key?' active':''), onclick:() => { wsFilter = key; applyMode(); }}, label); }
function raterGate(){
  const input = h('input', {placeholder:'your rater_id', maxlength:'40'});
  const start = () => { const v = input.value.trim(); if (v){ rater = v; wsFilter = 'unrated'; applyMode(); } };
  input.addEventListener('keydown', e => { if (e.key === 'Enter') start(); });
  const existing = raterIds();
  return h('div', {class:'ws-intro'},
    h('div', {class:'card pad'},
      h('div', {class:'label'}, 'Step 1 - identify the rater'),
      h('h2', {style:'font-size:16px;margin:6px 0'}, 'Blind rating'),
      h('p', {style:'color:var(--muted);font-size:13px;margin:0 0 4px'}, 'You will see only the event text and the appraisal text - no labels. Submit your own judgement (tag, severity, whether it seems to carry more than one injury). Preset values, and other raters\' annotations, are revealed only after you submit.'),
      h('div', {class:'raterpick'}, input, h('button', {class:'btn primary', onclick:start}, 'Start rating')),
      existing.length ? h('div', {class:'raterpick'}, h('span', {class:'label'}, 'or continue as'),
        existing.map(id => h('button', {class:'chip', onclick:() => { rater = id; wsFilter = 'unrated'; applyMode(); }}, id))) : null),
    h('div', {class:'card pad'},
      h('div', {class:'label'}, 'Why blind'),
      h('p', {style:'color:var(--muted);font-size:12.5px;margin:6px 0 0'}, 'The preset-vs-blind agreement rate is the manipulation check for the whole tag scheme, obtainable before any model is run at the cost of reading time alone. If two tags are reliably swapped, they are one tag. It only means anything if raters never see the preset tag first - which is why blind mode is the default and the analytics stay locked behind it.')));
}
function blindCard(rec){
  const submitted = hasSubmitted(rec.record_id, rater);
  const card = h('div', {class:'rec', 'data-rid':rec.record_id});
  card.appendChild(h('div', {class:'rhd'},
    h('span', {class:'rid'}, rec.record_id),
    submitted ? h('span', {class:'pill', style:'color:var(--fits)'}, 'you rated this') : h('span', {class:'label'}, 'awaiting your judgement')));
  const body = h('div', {class:'body'});
  body.appendChild(block('Event', rec.event.text));
  body.appendChild(block('Appraisal', rec.appraisal.text));
  if (submitted){ body.appendChild(revealBlock(rec)); body.appendChild(annsBlock(rec, true)); }
  else body.appendChild(ratingForm(rec, card));
  card.appendChild(body);
  return card;
}
function ratingForm(rec, card){
  const state = { tag:null, sev:null, multi:false };
  const wrap = h('div', {class:'form'});
  wrap.appendChild(h('div', {class:'q'}, 'Which single injury does this most carry?'));
  const tagOpts = h('div', {class:'opts'});
  BLIND_OPTIONS.forEach(opt => {
    const o = h('div', {class:'opt', onclick:() => { state.tag = opt; tagOpts.querySelectorAll('.opt').forEach(x=>x.classList.remove('sel')); o.classList.add('sel'); refresh(); }}, opt);
    tagOpts.appendChild(o);
  });
  wrap.appendChild(tagOpts);
  wrap.appendChild(h('div', {class:'q'}, 'How severe does it read? (1 = mild, 5 = severe)'));
  const sevOpts = h('div', {class:'opts'});
  [1,2,3,4,5].forEach(n => { const o = h('div', {class:'opt sev', onclick:() => { state.sev = n; sevOpts.querySelectorAll('.opt').forEach(x=>x.classList.remove('sel')); o.classList.add('sel'); refresh(); }}, String(n)); sevOpts.appendChild(o); });
  wrap.appendChild(sevOpts);
  const chk = h('input', {type:'checkbox', onchange:e => { state.multi = e.target.checked; }});
  wrap.appendChild(h('label', {class:'checkline'}, chk, 'It seems to carry more than one injury'));
  const btn = h('button', {class:'btn primary', disabled:true, onclick:() => submitRating(rec, state, card)}, 'Submit judgement');
  const note = h('span', {class:'hint'}, 'pick a tag and a severity to submit');
  wrap.appendChild(h('div', {class:'submit'}, btn, note));
  function refresh(){ const ok = state.tag && state.sev; btn.disabled = !ok; note.textContent = ok ? 'preset values reveal after you submit' : 'pick a tag and a severity to submit'; }
  return wrap;
}
function submitRating(rec, state, card){
  const now = new Date();
  const submitted_at = now.toISOString().replace(/\.\d+Z$/, 'Z');
  const revealed_at = new Date(now.getTime() + 1000).toISOString().replace(/\.\d+Z$/, 'Z');
  localAnns.push({ record_id:rec.record_id, rater_id:rater, blind_tag:state.tag, blind_severity:state.sev, blind_multi_injury:!!state.multi, submitted_at:submitted_at, revealed_at:revealed_at });
  saveLocal();
  if (card){
    card.replaceWith(blindCard(rec));
    const prog = document.getElementById('ws-prog');
    if (prog) prog.textContent = String(RECORDS.filter(r => hasSubmitted(r.record_id, rater)).length);
    const pend = document.getElementById('ws-pending');
    if (pend) pend.replaceWith(pendingSlot());
  } else applyMode();
}
function revealBlock(rec){
  const a = rec.appraisal, e = rec.event;
  const box = h('div', {class:'blk', style:'border-top:1px solid var(--line2);padding-top:11px'});
  box.appendChild(h('div', {class:'label'}, 'Preset values - revealed after your submission'));
  box.appendChild(h('div', {class:'kvs'},
    kv('provisional_tag', a.provisional_tag), kv('severity_intended', e.severity_intended),
    kv('agency_position', a.agency_position), kv('valence', a.valence), kv('verdict', VERDICT_LABEL[vof(rec)])));
  box.appendChild(verdictBlock(rec));
  return box;
}
function pendingSlot(){ return localAnns.length ? pendingPanel() : h('div', {id:'ws-pending'}); }
function pendingPanel(){
  const wrap = h('section', {id:'ws-pending', class:'card pad pending-panel'});
  const jsonl = localAnns.map(a => JSON.stringify(a)).join('\n');
  const ta = h('textarea', {readonly:true}, jsonl);
  wrap.appendChild(h('div', {class:'hd'},
    h('div', {class:'label'}, 'Pending annotations (this browser)'),
    h('div', {class:'cnt'}, localAnns.length + ' unsaved'),
    h('button', {class:'btn sm', onclick:() => { if (navigator.clipboard) navigator.clipboard.writeText(jsonl); ta.select(); }}, 'Copy'),
    h('button', {class:'btn sm', onclick:downloadPending}, 'Download .jsonl'),
    h('button', {class:'btn sm', onclick:() => { if (confirm('Clear pending annotations from this browser? Do this only after appending them to data/annotations.jsonl and rebuilding.')){ localAnns = []; saveLocal(); applyMode(); } }}, 'Clear')));
  wrap.appendChild(ta);
  wrap.appendChild(h('div', {class:'hint'}, 'Append these lines to ', h('span', {class:'mono'}, 'data/annotations.jsonl'), ' (append-only) and re-run ', h('span', {class:'mono'}, 'python gate/build.py'), '. The JSON is the source of truth; the browser only stages new lines.'));
  return wrap;
}
function downloadPending(){
  const blob = new Blob([localAnns.map(a => JSON.stringify(a)).join('\n') + '\n'], {type:'application/x-ndjson'});
  const url = URL.createObjectURL(blob);
  const a = h('a', {href:url, download:'annotations_new.jsonl'});
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---- analytics ---- */
function renderAnalytics(root){
  root.appendChild(headerStats());
  root.appendChild(gatesSection());
  root.appendChild(barsSection());
  root.appendChild(matrixSection());
  root.appendChild(deferredSection());
  root.appendChild(agreementSection());
}
function counts(recs){ const c = {}; STATES.forEach(s => c[s] = 0); recs.forEach(r => c[vof(r)]++); return c; }
function headerStats(){
  const c = counts(RECORDS);
  const u = c.fits + c.fits_via_proxy_probe;
  const rej = c.reject_no_probe + c.reject_guardrail + c.reject_multi_dimension;
  const stack = h('div', {class:'stack'});
  STATES.forEach(s => { if (!c[s]) return; stack.appendChild(h('div', {class:'seg ' + VCLASS[s], style:'width:' + (100*c[s]/RECORDS.length) + '%', title: VERDICT_LABEL[s] + ' - ' + c[s], onclick:() => drawerForState(s)})); });
  const legend = h('div', {class:'legend'});
  STATES.forEach(s => legend.appendChild(h('button', {class:'lg', onclick:() => drawerForState(s)}, h('span', {class:'dot ' + VCLASS[s]}), VERDICT_LABEL[s], h('span', {class:'c'}, c[s]))));
  return section('Admissibility overview', '6.1',
    'Share of material eligible for the benchmark. Verdicts are produced by running the gate; records not yet adjudicated show as pending. Click any segment or chip to list those records.',
    h('div', {class:'card pad'}, h('div', {class:'stats'},
      h('div', {class:'big'}, h('div', {class:'num'}, pct(u, RECORDS.length) + '%'), h('div', {class:'cap'}, u + ' usable / ' + rej + ' rejected / ' + c.pending + ' pending, of ' + RECORDS.length)),
      h('div', null, stack, legend))));
}
function gatesSection(){
  const gates = [
    {v:'reject_no_probe', id:'Gate A', q:'Is there a probe that can detect it?', ex:'Injury that surfaces only as physiological arousal (startle, sleep disruption) has no textual carrier - the model can report but not exhibit it.'},
    {v:'reject_guardrail', id:'Gate B', q:'Will the model accept the injection?', ex:'Material that trips a categorical refusal yields no data; refusal rate is recorded per event as a dependent variable, not discarded.'},
    {v:'reject_multi_dimension', id:'Gate C', q:'Does it carry a single leading injury?', ex:'Material with a second independent injury cannot be assigned to a cell, and blind labelling will necessarily fail.'}
  ];
  const grid = h('div', {class:'gates'});
  gates.forEach(g => {
    const recs = RECORDS.filter(r => vof(r) === g.v);
    const card = h('div', {class:'card pad gate'});
    card.appendChild(h('div', {class:'top'}, h('span', {class:'gid'}, g.id), h('span', {class:'verd ' + VCLASS[g.v]}, g.v)));
    card.appendChild(h('div', {class:'cnt'}, recs.length));
    card.appendChild(h('h3', null, g.q));
    card.appendChild(h('div', {style:'font-size:12px;color:var(--muted)'}, g.ex));
    const egs = h('div', {class:'eglist'});
    recs.slice(0,3).forEach(r => egs.appendChild(h('div', {class:'eg', onclick:() => drawerForRecords(r.record_id, [r])},
      h('div', {class:'a'}, truncate(r.appraisal.text, 84)), h('div', {class:'e'}, r.record_id))));
    card.appendChild(egs);
    if (recs.length > 3) card.appendChild(h('button', {class:'seeall', onclick:() => drawerForState(g.v)}, 'see all ' + recs.length + ' ->'));
    else if (recs.length === 0) card.appendChild(h('div', {class:'hint'}, 'none rejected here yet'));
    grid.appendChild(card);
  });
  const proxyRecs = RECORDS.filter(r => vof(r) === 'fits_via_proxy_probe');
  const strictFits = RECORDS.filter(r => vof(r) === 'fits').length;
  const withProxy = pct(strictFits + proxyRecs.length, RECORDS.length);
  const without = pct(strictFits, RECORDS.length);
  const proxyBox = h('div', {class:'card pad proxy-box'},
    h('div', {class:'label'}, 'Passing state - fits via proxy probe'),
    h('p', {style:'font-size:12.5px;color:var(--muted);margin:6px 0 8px'}, 'Material that fails Gate A directly but has a pure-text structural analogue (narrative breakage, false-correction concession, cue dissociation). Tracked separately - never folded into fits.'),
    h('div', {class:'rate'}, 'Usable rate falls from ', h('b', null, withProxy + '%'), ' ', h('span',{class:'arrow'},'->'), ' ', h('b', null, without + '%'), ' without these ' + proxyRecs.length + ' proxy record(s).'),
    proxyRecs.length ? h('button', {class:'seeall', onclick:() => drawerForRecords('Fits via proxy probe', proxyRecs)}, 'see all ' + proxyRecs.length + ' ->') : null);
  return section('The three gates', '6.2', 'Failing any one gate means the material cannot enter the benchmark - no rewording rescues it. Passing all three = usable.', h('div', null, grid, proxyBox));
}
function barsSection(){
  const groupings = [
    {t:'provisional_tag', groups: TAGS.map(g => ({label:g, test:r => r.appraisal.provisional_tag === g}))},
    {t:'exposure', groups: EXPOSURES.map(g => ({label:g, test:r => r.event.exposure === g}))},
    {t:'agency_position', groups: AGENCIES.map(g => ({label:g, test:r => r.appraisal.agency_position === g}))},
    {t:'threshold (criterion A vs sub-threshold)', groups: ['criterion_a','subthreshold'].map(g => ({label:g, test:r => r.event.threshold === g}))},
    {t:'guardrail_risk vs measured refusal_rate', groups: ['low','moderate','high'].map(g => ({label:g, test:r => r.event.guardrail_risk === g, refusal:true}))},
    {t:'injection_format', groups: Array.from(new Set(RECORDS.map(r => r.injection_format))).map(g => ({label:g, test:r => r.injection_format === g}))}
  ];
  const holder = h('div', {class:'card pad'});
  groupings.forEach((grp, i) => {
    const g = h('div', {class:'grp'});
    g.appendChild(h('div', {class:'gt'}, h('span', {class:'label'}, grp.t)));
    grp.groups.forEach(gr => g.appendChild(groupBar(gr.label, RECORDS.filter(gr.test), gr.refusal)));
    holder.appendChild(g);
    if (i < groupings.length - 1) holder.appendChild(h('div', {style:'border-top:1px solid var(--line2);margin:14px 0'}));
  });
  return section('Composition by group', '6.3', 'Each bar is one group; segments are its verdict mix (pending shown in grey). Click a bar to list that subset. For guardrail_risk, the measured mean refusal rate is shown where recorded.', holder);
}
function groupBar(label, recs, showRefusal){
  const c = counts(recs);
  const u = c.fits + c.fits_via_proxy_probe;
  const track = h('div', {class:'track'});
  STATES.forEach(s => { if (c[s]) track.appendChild(h('div', {class:'seg ' + VCLASS[s], style:'flex:' + c[s], title: VERDICT_LABEL[s] + ' - ' + c[s]})); });
  let right;
  if (showRefusal){
    const rr = recs.map(r => r.verdict && r.verdict.refusal_rate).filter(x => typeof x === 'number');
    const mean = rr.length ? (rr.reduce((a,b)=>a+b,0)/rr.length) : null;
    right = h('div', {class:'rt'}, mean==null ? 'refusal n/a' : h('b', null, mean.toFixed(2)), ' - n=' + recs.length);
  } else {
    right = h('div', {class:'rt'}, h('b', null, pct(u, recs.length) + '%'), ' - n=' + recs.length);
  }
  return h('div', {class:'rowbar' + (recs.length?'':' empty'), onclick:() => recs.length && drawerForRecords(label, recs)},
    h('div', {class:'gl', title:label}, label), track, right);
}

/* ---- coverage matrices (6.4) ---- */
function cellStateClass(recs){
  if (!recs.length) return {cls:'blank', txt:'.'};
  if (recs.every(isReject)) return {cls:'allrej', txt:'x' + recs.length};
  const adj = recs.filter(r => vof(r) !== 'pending');
  if (!adj.length) return {cls:'pend', txt:String(recs.length), style:'background:color-mix(in srgb,var(--pending) 22%,transparent)'};
  const u = adj.filter(usable).length;
  const alpha = 0.14 + 0.6 * (u/adj.length);
  return {cls:'', txt:String(recs.length), style:'background:rgba(47,158,68,' + alpha.toFixed(3) + ')'};
}
function matrixCell(recs, title){
  const st = cellStateClass(recs);
  return h('td', {class:'cell ' + st.cls, style:st.style, title:title + '\n' + recs.length + ' record(s)',
    onclick:() => recs.length && drawerForRecords(title, recs)}, st.txt);
}
function matrixSection(){
  // (a) provisional_tag x probe
  const cols = PROBES;
  const tableA = h('table', {class:'mx'});
  const hrA = h('tr'); hrA.appendChild(h('th', {class:'corner'}, 'tag \\ probe (probe_ids)'));
  cols.forEach(p => hrA.appendChild(h('th', {class:'col', title:p.label + ' - ' + p.kind + (p.distance?(' - distance ' + p.distance):'')}, (p.distance||'-'), h('small', null, p.probe_id.slice(0,7)))));
  tableA.appendChild(h('thead', null, hrA));
  const tbA = h('tbody');
  TAGS.forEach(tag => {
    const tr = h('tr'); tr.appendChild(h('td', {class:'rowh'}, tag));
    cols.forEach(p => {
      const recs = RECORDS.filter(r => r.appraisal.provisional_tag === tag && r.verdict && (r.verdict.probe_ids||[]).includes(p.probe_id));
      tr.appendChild(matrixCell(recs, tag + ' x ' + p.label));
    });
    tbA.appendChild(tr);
  });
  tableA.appendChild(tbA);

  // (b1) onset x duration
  const onsets = ['sudden','insidious'], durations = ['bounded','persisting'];
  const tableB = h('table', {class:'mx'});
  const hrB = h('tr'); hrB.appendChild(h('th', {class:'corner'}, 'onset \\ duration'));
  durations.forEach(d => hrB.appendChild(h('th', {class:'col'}, d)));
  tableB.appendChild(h('thead', null, hrB));
  const tbB = h('tbody');
  onsets.forEach(o => {
    const tr = h('tr'); tr.appendChild(h('td', {class:'rowh'}, o));
    durations.forEach(d => { const recs = RECORDS.filter(r => r.event.onset===o && r.event.duration===d); tr.appendChild(matrixCell(recs, o + ' x ' + d)); });
    tbB.appendChild(tr);
  });
  tableB.appendChild(tbB);

  // (b2) provisional_tag x exposure
  const tableC = h('table', {class:'mx'});
  const hrC = h('tr'); hrC.appendChild(h('th', {class:'corner'}, 'tag \\ exposure'));
  EXPOSURES.forEach(x => hrC.appendChild(h('th', {class:'col'}, x.slice(0,7))));
  tableC.appendChild(h('thead', null, hrC));
  const tbC = h('tbody');
  TAGS.forEach(tag => {
    const tr = h('tr'); tr.appendChild(h('td', {class:'rowh'}, tag));
    EXPOSURES.forEach(x => { const recs = RECORDS.filter(r => r.appraisal.provisional_tag===tag && r.event.exposure===x); tr.appendChild(matrixCell(recs, tag + ' x ' + x)); });
    tbC.appendChild(tr);
  });
  tableC.appendChild(tbC);

  const legend = h('div', {class:'mx-legend'},
    h('span', {class:'it'}, h('span', {class:'swatch', style:'background:rgba(47,158,68,.5)'}), 'sampled + adjudicated (shade = usable rate)'),
    h('span', {class:'it'}, h('span', {class:'swatch pend'}), 'sampled, pending'),
    h('span', {class:'it'}, h('span', {class:'swatch blank'}), 'not sampled (blank)'),
    h('span', {class:'it'}, h('span', {class:'swatch allrej'}), 'all rejected (x)'));

  return section('Coverage matrices', '6.4',
    'Two matrices because the holes are in two different places. (a) tag x probe is non-uniform - it shows which tags have thin measurement coverage (reject_no_probe lists no probes, proxies list only proxies). (b) the event design space shows authoring holes. There is no event x task matrix: every event runs every task, so it would be uniformly full and show nothing.',
    h('div', {class:'card pad'},
      h('div', {class:'mxblock'}, h('h3', null, '(a) provisional_tag x probe - measurement coverage'), h('div', {class:'matrix-wrap'}, tableA)),
      h('div', {class:'mxblock'}, h('h3', null, '(b) event design space - onset x duration'), h('div', {class:'matrix-wrap'}, tableB)),
      h('div', {class:'mxblock'}, h('h3', null, '(b) event design space - provisional_tag x exposure'), h('div', {class:'matrix-wrap'}, tableC)),
      legend));
}

function deferredSection(){
  const def = META.events_deferred || [];
  if (!def.length) return h('div');
  const grid = h('div', {class:'deferred-grid'});
  def.forEach(d => grid.appendChild(h('div', {class:'defcard'},
    h('div', {class:'sr'}, d.source_row), h('div', {class:'rs'}, d.reason), h('div', {class:'nt'}, d.note))));
  return section('Deferred events', '6.6', 'Rows consciously not authored yet - mostly on guardrail or purity grounds. They belong in the coverage picture, not hidden in the file.', h('div', {class:'card pad'}, grid));
}

function agreementSection(){
  const anns = allAnns().map(a => ({a:a, rec:recordById(a.record_id)})).filter(x => x.rec);
  const cols = TAGS.concat(['unclear']);
  const cm = {}; TAGS.forEach(t => { cm[t] = {}; cols.forEach(c => cm[t][c] = 0); });
  anns.forEach(x => { const preset = x.rec.appraisal.provisional_tag; const b = x.a.blind_tag; if (cm[preset] && b in cm[preset]) cm[preset][b]++; });
  const table = h('table', {class:'cm'});
  const head = h('tr'); head.appendChild(h('th', null, 'preset \\ blind'));
  cols.forEach(c => head.appendChild(h('th', {title:c}, c==='unclear'?'?':tagShort(c))));
  table.appendChild(head);
  TAGS.forEach(t => {
    const tr = h('tr'); tr.appendChild(h('td', {class:'rh', title:t}, tagShort(t)));
    cols.forEach(c => { const n = cm[t][c]; const cls = n===0 ? 'zero' : (c===t ? 'hit' : (c==='unclear' ? '' : 'miss')); tr.appendChild(h('td', {class:cls}, n||'')); });
    table.appendChild(tr);
  });
  const arate = h('div', {class:'arate'});
  let totAgree = 0, totN = 0;
  TAGS.forEach(t => {
    const tot = anns.filter(x => x.rec.appraisal.provisional_tag === t).length;
    const ag = cm[t][t]; totAgree += ag; totN += tot;
    arate.appendChild(h('div', {class:'r'}, h('div', {class:'l', title:t}, tagShort(t)), h('div', {class:'bar'}, h('i', {style:'width:' + pct(ag,tot) + '%'})), h('div', {class:'p'}, tot ? pct(ag,tot)+'%' : '-')));
  });
  const pairCount = {};
  TAGS.forEach((d1,i) => TAGS.forEach((d2,j) => { if (j>i){ pairCount[d1+'|'+d2] = cm[d1][d2] + cm[d2][d1]; } }));
  const topPairs = Object.entries(pairCount).filter(kv => kv[1]>0).sort((a,b)=>b[1]-a[1]).slice(0,4);
  const overall = pct(totAgree, totN);
  const left = h('div', null,
    h('div', {class:'label', style:'margin-bottom:8px'}, 'Confusion matrix - preset (row) vs blind (col)'),
    h('div', {style:'overflow-x:auto'}, table),
    topPairs.length ? h('div', {class:'pairs'}, h('div', {class:'label'}, 'Systematically confused tag pairs'),
      topPairs.map(kv => { const p = kv[0].split('|'); return h('div', {class:'pr'}, h('span', null, p[0] + '  <->  ' + p[1]), h('span', {class:'mono'}, kv[1] + 'x')); })) : null);
  const right = h('div', null,
    h('div', {class:'kbig', style:'margin-bottom:12px'}, 'Overall preset-blind agreement: ', h('b', null, totN ? overall + '%' : '-'), ' ', h('span', {class:'hint'}, '(' + totAgree + '/' + totN + ' labelled annotations)')),
    h('div', {class:'label', style:'margin-bottom:6px'}, 'Per-tag agreement'), arate);
  return section('Agreement - the tag scheme manipulation check', '7',
    'Confusion matrix of preset vs blind tag, per-tag agreement, and the tag pairs raters keep swapping. Obtainable before any model is run. If two tags are reliably swapped, they are one tag.',
    h('div', {class:'card pad'}, h('div', {class:'agree-grid'}, left, right)));
}

/* ---- shared ---- */
function section(title, n, desc, body){
  return h('section', {class:'section'}, h('div', {class:'hd'}, h('h2', null, title), h('span', {class:'n'}, 'section ' + n)), desc ? h('div', {class:'desc'}, desc) : null, body);
}
function block(title, text){ return h('div', {class:'blk'}, h('div', {class:'bt label'}, title), h('div', {class:'txt'}, text)); }
function eventKnobs(e){
  return h('div', {class:'kvs'},
    kv('threshold', e.threshold), kv('exposure', e.exposure), kv('onset', e.onset), kv('duration', e.duration),
    kv('resolution', e.resolution), kv('time_since', e.time_since), kv('recurrence', e.recurrence===0?'0 (constitutive)':e.recurrence),
    kv('event_age', e.event_age), kv('severity_intended', e.severity_intended), kv('guardrail_risk', e.guardrail_risk),
    e.modifiers && e.modifiers.length ? kv('modifiers', e.modifiers.join(', ')) : null, kv('shift_note', e.shift_note ? 'yes' : 'null'));
}
function verdictBlock(rec){
  const v = rec.verdict;
  if (!v) return h('div', {class:'why'}, 'Pending - no verdict yet. Produced by running the gate.');
  const probes = h('div', {class:'probes'});
  (v.probe_ids||[]).forEach(pid => { const p = PROBE_BY_ID[pid]; probes.appendChild(h('span', {class:'pb', title:(p?p.label:pid)}, pid)); });
  if (!(v.probe_ids||[]).length) probes.appendChild(h('span', {class:'hint'}, 'no probe listed'));
  return h('div', null, v.why ? h('div', {class:'why'}, v.why) : null, probes,
    h('div', {class:'kvs', style:'margin-top:8px'}, v.proxy_probe ? kv('proxy_probe', v.proxy_probe) : null, kv('refusal_rate', v.refusal_rate==null?'null':v.refusal_rate)));
}
function annsBlock(rec, isBlindContext){
  const list = annsFor(rec.record_id);
  const box = h('div', {class:'anns'});
  box.appendChild(h('div', {class:'label'}, 'Annotations (' + list.length + ')'));
  if (!list.length){ box.appendChild(h('div', {class:'hint'}, 'none yet')); return box; }
  list.forEach(a => { const mine = isBlindContext && a.rater_id === rater;
    box.appendChild(h('div', {class:'ann' + (mine?' mine':'')}, h('b', null, a.rater_id), 'tag=' + a.blind_tag, 'sev=' + a.blind_severity, 'multi=' + a.blind_multi_injury, h('span', {style:'margin-left:auto'}, a.submitted_at))); });
  return box;
}
function fullCard(rec){
  const e = rec.event, a = rec.appraisal, v = vof(rec);
  const card = h('div', {class:'rec'});
  card.appendChild(h('div', {class:'rhd'},
    h('span', {class:'rid'}, rec.record_id),
    v==='pending' ? h('span', {class:'pill pend'}, 'pending') : h('span', {class:'pill verd', style:'background:var(' + VVAR[v] + ')'}, VERDICT_LABEL[v]),
    h('span', {class:'pill tag'}, a.provisional_tag), h('span', {class:'pill'}, a.agency_position), h('span', {class:'pill'}, a.valence),
    h('span', {class:'pill'}, e.exposure), h('span', {class:'pill'}, rec.injection_format)));
  const body = h('div', {class:'body'});
  const evb = block('Event - ' + e.label + ' (' + e.event_id + ')', e.text); evb.appendChild(eventKnobs(e));
  if (e.shift_note) evb.appendChild(h('div', {class:'why', style:'margin-top:7px'}, 'shift_note: ' + e.shift_note));
  body.appendChild(evb);
  body.appendChild(block('Appraisal', a.text));
  const vb = h('div', {class:'blk'}, h('div', {class:'bt label'}, 'Verdict - ' + VERDICT_LABEL[v])); vb.appendChild(verdictBlock(rec));
  body.appendChild(vb);
  body.appendChild(annsBlock(rec, false));
  card.appendChild(body);
  return card;
}
function kv(k, v){ return h('span', {class:'kv'}, h('b', null, k), ' ', h('span', null, String(v))); }
function truncate(s, n){ return s.length > n ? s.slice(0, n-1) + '...' : s; }

function drawerForState(s){ drawerForRecords(VERDICT_LABEL[s], RECORDS.filter(r => vof(r) === s)); }
function drawerForRecords(title, recs){
  openDrawerNode(title, recs.length + ' record' + (recs.length===1?'':'s'), recs.length ? h('div', null, recs.map(fullCard)) : h('div', {class:'empty'}, 'No records in this subset.'));
}
function openDrawerNode(title, subtitle, node){
  const d = document.getElementById('drawer'); clear(d);
  d.appendChild(h('div', {class:'drawer-head'}, h('div', null, h('div', {class:'drawer-title'}, title), subtitle ? h('div', {class:'drawer-count'}, subtitle) : null), h('button', {class:'x', title:'close', onclick:closeDrawer}, 'x')));
  d.appendChild(h('div', {class:'drawer-body'}, node));
  document.getElementById('scrim').hidden = false; d.classList.add('open');
}
function closeDrawer(){ document.getElementById('drawer').classList.remove('open'); document.getElementById('scrim').hidden = true; }
document.getElementById('scrim').addEventListener('click', closeDrawer);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

applyMode();
</script>
</body>
</html>

"""

if __name__ == "__main__":
    main()
