"""Small, frozen final-stage replay of inspected glaucoma errors. Offline by default."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "OphthalmicAgent"), str(ROOT / "OphthalmicAgent/scripts")]
import run_fairvision_ablation as base
from Confirmation import contract

VERSION = "glaucoma_final_replay_v2_20261006"
ARMS = ("legacy_prompt_control", "v2")
DEFAULT_BUNDLE = ROOT / "OphthalmicAgent/ReplayV2/bundle.json"
DEFAULT_RUN = ROOT / "OphthalmicAgent/outputs/glaucoma_v2_replay"
EVIDENCE_FIELDS = (
    "patient_narrative", "retfound_glaucoma_probability_percent",
    "oct_specialist_report", "slo_specialist_report", "vertical_cup_to_disc_ratio",
    "demographic_reliability_trust_score",
)
GUARDRAILS = """Evidence handling for this replay:
- An ungradable or non-diagnostic image/report is missing evidence, not evidence of absence.
  Distinguish findings explicitly observed from findings that could not be assessed.
- CDR is an approximate segmentation output. Mask quality is not verified here.
  Do not assume an extreme or boundary value is a verified anatomical finding;
  unresolved tool/report conflict warrants review, not an automatic positive or negative label.
- Do not resolve uncertain scan location by inventing anatomy or new measurements.
Use the original full evidence for the forced diagnosis. Record unresolved evidence
limitations in the reasoning and escalation decision; do not vote across scenarios."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def code_hashes():
    paths = [Path(__file__), Path(contract.__file__), Path(base.__file__), Path(base.audit.__file__)]
    return {str(p.relative_to(ROOT)): base.sha(p) for p in paths}


def legacy_system(path):
    systems = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Dict):
            values = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
            role = values.get("role")
            if isinstance(role, ast.Constant) and role.value == "system":
                systems.append(ast.literal_eval(values["content"]))
    base.require(len(systems) == 1 and isinstance(systems[0], str), "Expected one literal orchestrator prompt")
    return systems[0]


def measurement_checks(evidence):
    raw = evidence["vertical_cup_to_disc_ratio"]
    missing = raw is None or str(raw).strip().lower() in ("", "not available", "none", "null", "n/a")
    status = "missing" if missing else "unverified"
    if not missing:
        base.require(not isinstance(raw, bool), "Boolean CDR is invalid")
        value = float(raw)
        status = ("invalid" if not math.isfinite(value) or not 0 <= value <= 1 else
                  "boundary_unverified" if value in (0, 1) else "unverified")
    return dict(cdr_status=status, segmentation_mask_quality="not_verified",
                original_images_reassessed=False, scan_location_verified=False)


def requests_for(case, settings, systems):
    e = case["evidence"]
    user = ("### MULTI-AGENT CASE INPUTS\n"
            f"- Patient Narrative: {e['patient_narrative']}\n"
            f"- RetFound (OCT) Score: {e['retfound_glaucoma_probability_percent']}%\n"
            f"- OCT Specialist Report: {e['oct_specialist_report']}\n"
            f"- SLO Specialist Report: {e['slo_specialist_report']}\n"
            f"- Cup to Disc ratio: {e['vertical_cup_to_disc_ratio']}\n"
            f"- Trust Score: {e['demographic_reliability_trust_score']}\n"
            f"- Counterfactual Evidence-Ablation Trace: {json.dumps(case['trace'], sort_keys=True)}")
    result = {}
    for arm in ARMS:
        content = user
        if arm == "v2":
            content += "\n- Measurement provenance: " + json.dumps(measurement_checks(e), sort_keys=True)
        request = dict(model=settings["deployment"], temperature=settings["temperature"],
                       max_completion_tokens=settings["max_completion_tokens"],
                       messages=[dict(role="system", content=systems[arm]), dict(role="user", content=content)])
        if arm == "v2":
            request["response_format"] = contract.final_schema()
        result[arm] = request
    return result


def make_bundle(cases, predictions, traces, system, sources):
    from CounterfactualAgent.counterfactual_agent import CounterfactualAgent
    settings = dict(deployment="gpt-5.1", api_version="2024-12-01-preview", temperature=.3,
                    max_completion_tokens=2000, sdk_max_retries=0, timeout_seconds=180)
    # Retain the original reasoning instructions; replace only the output-format section.
    marker = "Output EXACTLY in the following format:"
    base.require(system.count(marker) == 1, "Legacy output section changed")
    v2 = contract.final_messages([dict(role="system", content=system.split(marker)[0])])[0]["content"]
    systems = {ARMS[0]: system, "v2": v2 + "\n" + GUARDRAILS}
    selected, evaluation = [], []
    for i, case in enumerate(cases):
        truth, old, anchor = case["truth"], predictions["RetinAgent"][i], predictions["RETFound"][i]
        if old == truth and anchor == truth:
            continue
        group = "historical_error" if old != truth else "historical_correction_control"
        records = traces.get(case["case_id"], [])
        evaluation.append(dict(case_id=case["case_id"], truth=truth, historical_prediction=old,
                               retfound_prediction=anchor, group=group, eligible=bool(records)))
        if not records:
            continue
        # Earliest file position, fixed before new outputs and independent of correctness.
        saved = min(records, key=lambda r: r["source_line"])
        record = saved["record"]
        evidence = {k: record["evidence"][k] for k in EVIDENCE_FIELDS}
        row = dict(case_id=case["case_id"], evidence=evidence,
                   trace=CounterfactualAgent.concise_trace(record), source_line=saved["source_line"],
                   source_fingerprint=record["fingerprint"], available_trace_records=len(records))
        row["evidence_sha256"] = base.digest(dict(evidence=evidence, trace=row["trace"]))
        row["requests"] = requests_for(row, settings, systems)
        selected.append(row)
    selected.sort(key=lambda c: base.digest([VERSION, c["case_id"]]))
    calls = len(selected) * len(ARMS)
    base.require(0 < calls <= 250, f"Replay needs {calls} calls; exceeds the 250-call ceiling")
    bundle = dict(version=VERSION, scope="exploratory error-selected final-stage replay; not full-pipeline or held-out",
                  selection="all historical errors plus all historical corrections; first saved trace by file position",
                  settings=settings, systems=systems, planned_api_calls=calls, api_budget=calls,
                  cases=selected, evaluation=evaluation, source_sha256=sources, runtime_code_sha256=code_hashes())
    return {**bundle, "fingerprint": base.digest(bundle)}


def prepare(args):
    import audit_fairvision_glaucoma_traces as review
    cases, predictions, _, sources, _ = review.uncertainty.load_paper(args.paper_root, 250)
    baseline = review.uncertainty.indexed(base.read_csv(args.paper_root / "retfound_predictions_recovered.csv"), "Filename")
    traces = review.load_traces(args.traces, {c["case_id"] for c in cases}, baseline)
    prompt_source = ROOT / "OphthalmicAgent/Orchestrator/fairvision_glaucoma.py"
    sources += [args.traces, prompt_source, Path(review.__file__), Path(review.uncertainty.__file__)]
    hashes = {os.path.relpath(p.resolve(), ROOT): base.sha(p) for p in sources}
    bundle = make_bundle(cases, predictions, traces, legacy_system(prompt_source), hashes)
    if args.bundle.exists():
        base.require(json.loads(args.bundle.read_text()) == bundle, "Bundle already frozen; refusing to overwrite")
    else:
        base.write_json(args.bundle, bundle)
    print(summary(bundle))


def load_bundle(path):
    bundle = json.loads(path.read_text())
    base.require(base.digest({k: v for k, v in bundle.items() if k != "fingerprint"}) == bundle["fingerprint"],
                 "Bundle checksum mismatch")
    base.require(bundle["version"] == VERSION and bundle["runtime_code_sha256"] == code_hashes(),
                 "Replay code changed since preparation; do not mix versions")
    base.require(0 < bundle["api_budget"] <= 250 and bundle["planned_api_calls"] == 2 * len(bundle["cases"]),
                 "Invalid frozen API budget")
    ids = [c["case_id"] for c in bundle["cases"]]
    base.require(len(ids) == len(set(ids)), "Duplicate replay case")
    for case in bundle["cases"]:
        base.require(case["requests"] == requests_for(case, bundle["settings"], bundle["systems"]),
                     "Frozen request differs from allowed evidence")
        base.require(case["evidence_sha256"] == base.digest(dict(evidence=case["evidence"], trace=case["trace"])),
                     "Evidence fingerprint mismatch")
    return bundle


def summary(bundle):
    selected = Counter(r["group"] for r in bundle["evaluation"] if r["eligible"])
    missing = [r["case_id"] for r in bundle["evaluation"] if not r["eligible"]]
    return (f"{VERSION}: {bundle['scope']}\n"
            f"Eligible: {dict(selected)}; missing saved evidence: {len(missing)}\n"
            f"Two final-stage requests/case; maximum {bundle['api_budget']} attempts, including failures.\n"
            "No specialist, counterfactual, imaging-model or training calls. No automatic retries.\n"
            "Missing cases: " + ", ".join(Path(k).stem for k in missing))


def parse_response(arm, raw):
    base.require(arm in ARMS and isinstance(raw, str), "Invalid arm or response type")
    if arm == "v2":
        return contract.parse_final(raw)
    blocks = re.findall(r"\[LABELS\](.*?)\[/LABELS\]", raw, re.DOTALL)
    base.require(len(blocks) == 1, "Legacy response must have exactly one LABELS block; no negative fallback")
    label = re.fullmatch(r"\s*GLAUCOMA_DETECTED:\s*([01])\s*", blocks[0])
    base.require(label is not None, "Legacy label is malformed or ambiguous; no negative fallback")
    return dict(diagnosis=int(label.group(1)), reasoning=raw, escalation_required=None)


def ensure_run_root(root, bundle):
    base.require(not root.is_symlink(), "Replay root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "run_identity.json"
    identity = dict(version=VERSION, bundle_fingerprint=bundle["fingerprint"], api_budget=bundle["api_budget"])
    if marker.exists():
        base.require(json.loads(marker.read_text()) == identity, "Run root belongs to a different frozen replay")
    else:
        base.require(not any(p.name != ".run.lock" for p in root.iterdir()),
                     "Refusing to write into a nonempty directory without a V2 run identity")
        base.write_json(marker, identity)


class Ledger:
    """Reserve durably before sending; an interrupted attempt is never automatically replayed."""
    def __init__(self, root, bundle):
        ensure_run_root(root, bundle)
        self.db = sqlite3.connect(root / "ledger.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            case_id TEXT NOT NULL, arm TEXT NOT NULL, request_hash TEXT NOT NULL,
            request_json TEXT NOT NULL, started_utc TEXT NOT NULL, status TEXT NOT NULL,
            finished_utc TEXT, raw TEXT, response_json TEXT, parsed_json TEXT, error TEXT,
            PRIMARY KEY (case_id, arm))""")
        self.db.commit()
        try:
            self.pin("bundle", bundle["fingerprint"])
            self.pin("budget", str(bundle["api_budget"]))
        except BaseException:
            self.db.close()
            raise
        self.limit = bundle["api_budget"]

    def pin(self, name, value):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?)", (name, value))
            actual = self.db.execute("SELECT value FROM metadata WHERE name=?", (name,)).fetchone()[0]
            base.require(actual == value, f"Run identity changed: {name}")

    def rows(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM attempts ORDER BY started_utc, case_id, arm")]

    def reserve(self, case_id, arm, request):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT request_hash FROM attempts WHERE case_id=? AND arm=?", (case_id, arm)).fetchone()
            if old:
                base.require(old[0] == base.digest(request), "Attempted request changed")
                self.db.commit()
                return False
            count = self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            base.require(count < self.limit, "Persistent API budget exhausted")
            self.db.execute("INSERT INTO attempts (case_id,arm,request_hash,request_json,started_utc,status) VALUES (?,?,?,?,?,?)",
                            (case_id, arm, base.digest(request), base.canonical(request), utc(), "reserved"))
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            raise

    def save_response(self, key, raw, response):
        with self.db:
            self.db.execute("UPDATE attempts SET raw=?,response_json=?,status='received',finished_utc=? WHERE case_id=? AND arm=?",
                            (raw, base.canonical(response), utc(), *key))

    def finish(self, key, status, parsed=None, error=None):
        with self.db:
            self.db.execute("UPDATE attempts SET status=?,parsed_json=?,error=?,finished_utc=? WHERE case_id=? AND arm=?",
                            (status, base.canonical(parsed) if parsed is not None else None, error, utc(), *key))

    def close(self):
        self.db.close()


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another replay process holds this run root") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(bundle, root, client_factory, max_cases=None):
    base.require(max_cases is None or max_cases > 0, "max-cases must be positive")
    ensure_run_root(root, bundle)
    with run_lock(root):
        ledger = Ledger(root, bundle)
        client = None
        try:
            base.require(not ledger.db.execute("SELECT 1 FROM attempts WHERE status='model_drift'").fetchone(),
                         "Returned model drift was detected; this run is stopped")
            for case in bundle["cases"][:max_cases]:
                arms = list(ARMS)
                if int(base.digest([VERSION, case["case_id"], "order"])[0], 16) % 2:
                    arms.reverse()
                for arm in arms:
                    request = case["requests"][arm]
                    key = (case["case_id"], arm)
                    if not ledger.reserve(*key, request):
                        continue
                    try:
                        if client is None:
                            client = client_factory(ledger)
                        response = client.chat.completions.create(**request)
                    except Exception as exc:
                        ledger.finish(key, "api_error", error=type(exc).__name__)
                        # Stop, without retries or burning the rest of the budget on a configuration failure.
                        raise RuntimeError("API attempt failed and was recorded; no retry was made") from exc
                    data = response.model_dump(mode="json")
                    data["provider_request_id"] = getattr(response, "_request_id", None)
                    choice = response.choices[0] if response.choices else None
                    raw = choice.message.content if choice else None
                    ledger.save_response(key, raw, data)
                    try:
                        base.require(isinstance(response.model, str) and bool(response.model), "Missing returned model identity")
                        ledger.pin("returned_model", response.model)
                    except ValueError:
                        ledger.finish(key, "model_drift", error="Returned model identity changed or missing")
                        raise
                    try:
                        base.require(choice is not None and choice.finish_reason == "stop", "Response did not finish normally")
                        base.require(isinstance(raw, str) and bool(raw.strip()), "Empty/refused response")
                        parsed = parse_response(arm, raw)
                    except (ValueError, TypeError) as exc:
                        ledger.finish(key, "invalid", error=str(exc))
                        raise ValueError("Invalid response saved; inspect it before continuing. No retry or negative fallback.") from exc
                    else:
                        ledger.finish(key, "valid", parsed=parsed)
                    print(f"attempt {len(ledger.rows())}/{ledger.limit}: {Path(key[0]).stem} {arm}", flush=True)
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()
            ledger.close()


def azure_factory(bundle):
    def make(ledger):
        from openai import AzureOpenAI
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        base.require(endpoint and key, "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        ledger.pin("endpoint_sha256", base.digest(endpoint))
        settings = bundle["settings"]
        return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=settings["api_version"],
                           max_retries=0, timeout=settings["timeout_seconds"])
    return make


def collect(bundle, root):
    ensure_run_root(root, bundle)
    attempts = []
    if (root / "ledger.sqlite3").exists():
        ledger = Ledger(root, bundle)
        try:
            attempts = ledger.rows()
        finally:
            ledger.close()
    mapping = {(r["case_id"], r["arm"]): r for r in attempts}
    allowed = {c["case_id"]: c for c in bundle["cases"]}
    base.require(all(k in allowed and arm in ARMS for k, arm in mapping), "Unknown case or arm in ledger")
    output, parsed = [], {}
    for row in bundle["evaluation"]:
        result = dict(row)
        if row["eligible"]:
            case = allowed[row["case_id"]]
            result.update(source_line=case["source_line"], evidence_sha256=case["evidence_sha256"],
                          cdr_status=measurement_checks(case["evidence"])["cdr_status"],
                          saved_full_evidence_label=case["trace"]["full_evidence_diagnosis"])
        else:
            result.update(source_line="", evidence_sha256="", cdr_status="missing_evidence", saved_full_evidence_label="")
        for arm in ARMS:
            attempt = mapping.get((row["case_id"], arm))
            status = "missing_evidence" if not row["eligible"] else "not_attempted"
            value = None
            if attempt:
                request = allowed[row["case_id"]]["requests"][arm]
                base.require(attempt["request_hash"] == base.digest(request) and
                             json.loads(attempt["request_json"]) == request, "Request receipt mismatch")
                status = attempt["status"]
                if status == "valid":
                    value = parse_response(arm, attempt["raw"])
                    base.require(value == json.loads(attempt["parsed_json"]), "Saved parse mismatch")
                    parsed[(row["case_id"], arm)] = value
            result[arm + "_status"] = status
            result[arm + "_prediction"] = value["diagnosis"] if value else ""
            result[arm + "_escalation_required"] = value["escalation_required"] if value else ""
            result[arm + "_reasoning"] = value["reasoning"] if value else ""
            result[arm + "_escalation_reason"] = value.get("escalation_reason", "") if value else ""
        output.append(result)
    groups = {}
    for group in ("historical_error", "historical_correction_control"):
        rows = [r for r in output if r["group"] == group]
        matched = [r for r in rows if all((r["case_id"], arm) in parsed for arm in ARMS)]
        both, only_v2, only_control, neither = 0, 0, 0, 0
        for row in matched:
            control, new = (parsed[(row["case_id"], arm)]["diagnosis"] == row["truth"] for arm in ARMS)
            both += control and new
            only_v2 += new and not control
            only_control += control and not new
            neither += not new and not control
        groups[group] = dict(expected=len(rows), eligible=sum(r["eligible"] for r in rows),
            paired_valid=len(matched), both_correct=both, v2_only_correct=only_v2,
            control_only_correct=only_control, both_wrong=neither,
            arms={arm: dict(valid=sum((r["case_id"], arm) in parsed for r in rows),
                correct=sum(parsed[(r["case_id"], arm)]["diagnosis"] == r["truth"] for r in rows if (r["case_id"], arm) in parsed)) for arm in ARMS})
    report = dict(version=VERSION, bundle_fingerprint=bundle["fingerprint"], scope=bundle["scope"],
                  attempts_reserved=len(attempts), api_budget=bundle["api_budget"],
                  complete=all(r["v2_status"] == r[ARMS[0]+"_status"] == "valid" for r in output if r["eligible"]),
                  groups=groups,
                  caution="Error-selected replay. No full-cohort F1, superiority claim, or replacement of historical predictions.")
    root.mkdir(parents=True, exist_ok=True)
    base.audit.write_csv(root / "case_results.csv", output)
    with (root / "api_receipts.jsonl").open("w") as handle:
        for row in attempts:
            receipt = dict(row)
            for field in ("request_json", "response_json", "parsed_json"):
                receipt[field] = json.loads(receipt[field]) if receipt[field] else None
            handle.write(base.canonical(receipt) + "\n")
    base.write_json(root / "summary.json", report)
    text = [summary(bundle), "", f"Reserved attempts: {len(attempts)}; complete eligible pairs: {report['complete']}", ""]
    for group, value in groups.items():
        text += [group + ": " + json.dumps(value, sort_keys=True)]
    text += ["", report["caution"], "Malformed, failed, interrupted and missing cases are not assigned a negative prediction."]
    (root / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "preflight", "run", "collect"), default="preflight")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--paper-root", type=Path, default=ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_case_recovery")
    parser.add_argument("--traces", type=Path, default=ROOT / "OphthalmicAgent/outputs/glaucoma_counterfactual_250/counterfactual_traces.jsonl")
    parser.add_argument("--allow-api", action="store_true", help="Explicit opt-in required for any paid request")
    parser.add_argument("--max-cases", type=int, help="Run only this many cases from the frozen order; resume uses the same prefix")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
        return
    bundle = load_bundle(args.bundle)
    if args.stage == "preflight":
        print(summary(bundle))
        print("Bundle and frozen requests verified. No API client created; no network calls.")
    elif args.stage == "run":
        base.require(args.allow_api, "Paid run requires --allow-api; default is offline")
        base.require(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"), "Missing Azure credentials")
        execute(bundle, args.run_root, azure_factory(bundle), args.max_cases)
        collect(bundle, args.run_root)
    else:
        collect(bundle, args.run_root)


if __name__ == "__main__":
    main()
