"""Four-attempt evidence-qualification pilot using existing V3 OCT reports only."""
from __future__ import annotations

import argparse
import copy
from contextlib import closing, ExitStack
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "OphthalmicAgent"), str(ROOT / "OphthalmicAgent/scripts")]
import run_glaucoma_v3_replay as v3
import audit_glaucoma_v2_replay as v2_audit
from ReplayV4 import evidence_contract as contract

replay, base = v3.replay, v3.base
VERSION = "glaucoma_evidence_qualification_v4_20261007"
STAGES = ("review", "final")
API_CAP = 4
DEFAULT_DIR = ROOT / "OphthalmicAgent/outputs/glaucoma_v4_replay"
SCOPE = ("Two-case, error-selected downstream development pilot; cached V3 OCT and other sources. "
         "Qualification review replaces counterfactual diagnosis generation. Not a full pipeline or held-out test.")


def code_hashes():
    paths = [Path(__file__), Path(contract.__file__), Path(v2_audit.__file__), Path(v2_audit.errors.__file__)]
    return {**v3.code_hashes(), **{str(p.relative_to(ROOT)): base.sha(p) for p in paths}}


def read_ledger(root):
    path = root / "ledger.sqlite3"
    if not path.exists():
        return [], {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return ([dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY started_utc, case_id, arm")],
                dict(db.execute("SELECT name,value FROM metadata")))


def verify_envelope(row, model):
    response = json.loads(row["response_json"])
    choices = response.get("choices") or []
    base.require(len(choices) == 1 and choices[0]["finish_reason"] == "stop" and
                 not choices[0]["message"].get("refusal") and not choices[0]["message"].get("tool_calls") and
                 choices[0]["message"].get("content") == row["raw"] and response.get("model") == model,
                 "Invalid, refused, incomplete or changed-model receipt")


def read_sources(v3_dir, v2_root, v2_bundle):
    old = replay.load_bundle(v2_bundle)
    receipts = [json.loads(x) for x in (v2_root / "api_receipts.jsonl").read_text().splitlines() if x.strip()]
    old_rows, _ = v2_audit.validate_exports(old, base.read_csv(v2_root / "failed_case_results.csv"), receipts)
    # On the cluster, also check the live V2 ledger, not merely a potentially stale export.
    live, _ = read_ledger(v2_root)
    if (v2_root / "ledger.sqlite3").exists():
        decoded = copy.deepcopy(live)
        for r in decoded:
            for field in ("request_json", "response_json", "parsed_json"):
                r[field] = json.loads(r[field]) if r[field] else None
        base.require(decoded == receipts, "V2 ledger/export mismatch; do not ignore additional attempts")
    bundle = v3.load_bundle(v3_dir)
    amendment = v3.load_amendment(bundle, v3_dir)
    rows, meta = v3.read_attempts(v3_dir)
    mapping, parsed = v3.validated_attempts(bundle, v3_dir, rows)
    base.require(meta.get("bundle") == bundle["fingerprint"] and meta.get("version") == v3.VERSION and
                 meta.get("budget") == str(v3.run_identity(bundle, amendment)["api_budget"]), "V3 ledger identity mismatch")
    base.require(bundle["source_bundle_fingerprint"] == old["fingerprint"], "V2/V3 source mismatch")
    base.require(all(r["status"] == "valid" for r in mapping.values()), "Unresolved V3 attempt; inspect first")
    base.require(meta.get("returned_model") and meta.get("endpoint_sha256"), "Missing V3 model/endpoint pins")
    for row in mapping.values():
        verify_envelope(row, meta["returned_model"])
    completed = [c for c in bundle["cases"] if "final" in parsed[c["case_id"]]]
    base.require([c["case_id"] for c in completed] == [c["case_id"] for c in bundle["cases"][:2]] and
                 len(mapping) == 6, "V4 is restricted to the two completed V3 pilot cases; no cohort expansion")
    base.require(len(receipts) + len(rows) + API_CAP <= 250, "Known prior attempts plus V4 exceed 250")
    snapshot = dict(v2_receipts_sha256=base.sha(v2_root / "api_receipts.jsonl"),
                    v2_results_sha256=base.sha(v2_root / "failed_case_results.csv"),
                    v2_bundle_fingerprint=old["fingerprint"], v3_bundle_fingerprint=bundle["fingerprint"],
                    v3_attempts_sha256=base.digest(rows), v3_metadata=meta,
                    v2_attempts=len(receipts), v3_attempts=len(rows))
    return bundle, parsed, old, old_rows, snapshot


def payload(case):
    e = case["evidence"]
    units = contract.source_units(e)
    provenance = replay.measurement_checks(e)
    provenance.pop("original_images_reassessed")
    provenance.update(oct_images_reassessed_in_saved_v3=True, slo_images_reassessed=False,
                      original_historical_request_images_verified=False)
    return dict(sources=units, protected_limitation_ids=contract.protected_ids(units),
                measurement_provenance=provenance,
                source_dependence=dict(oct=["RETFound", "OCT specialist"], slo=["CDR", "SLO specialist"]),
                oct_acquisition=case["oct_acquisition"])


def request_for(stage, case, bundle, parsed):
    data = payload(case)
    if stage == "final":
        data["qualification_review"] = parsed["review"]
        data["other_evidence"] = {k: case["evidence"][k] for k in replay.EVIDENCE_FIELDS
                                  if k not in ("oct_specialist_report", "slo_specialist_report")}
    else:
        base.require(stage == "review", "Unknown stage")
    settings = bundle["settings"]
    return dict(model=settings["deployment"], temperature=0 if stage == "review" else settings["temperature"],
                max_completion_tokens=2000, response_format=contract.schema(stage),
                messages=[dict(role="system", content=contract.REVIEW_SYSTEM if stage == "review" else contract.FINAL_SYSTEM),
                          dict(role="user", content=base.canonical(data))])


def parse(stage, raw, case, parsed):
    return contract.parse(stage, raw, contract.source_units(case["evidence"]), parsed.get("review"),
                          replay.measurement_checks(case["evidence"])["cdr_status"])


def prepare(args):
    root = args.experiment_dir.resolve()
    for source in (args.v3_dir.resolve(), args.v2_root.resolve(), args.v2_bundle.parent.resolve()):
        base.require(not root.is_relative_to(source) and not source.is_relative_to(root), "Use a separate V4 directory")
    base.require(not root.exists(), "V4 directory already exists; use preflight/resume, never reprepare to retry")
    old, parsed, v2_bundle, old_rows, snapshot = read_sources(args.v3_dir, args.v2_root, args.v2_bundle)
    cases = []
    for c in old["cases"][:2]:
        evidence = {**c["evidence"], "oct_specialist_report": parsed[c["case_id"]]["oct"]["report"]}
        cases.append(dict(case_id=c["case_id"], evidence=evidence, oct_acquisition=c["image"]["metadata"]))
    evaluation = copy.deepcopy(old["evaluation"])
    for row in evaluation:
        final = parsed[row["case_id"]].get("final") if row["eligible"] else None
        row["v3_prediction"] = final["diagnosis"] if final else None
        row["v4_selected"] = row["case_id"] in {c["case_id"] for c in cases}
    bundle = dict(version=VERSION, scope=SCOPE, api_budget=API_CAP, cases=cases, evaluation=evaluation,
                  settings=old["settings"], source_snapshot=snapshot,
                  source_paths=dict(v3_dir=str(args.v3_dir.resolve()), v2_root=str(args.v2_root.resolve()),
                                    v2_bundle=str(args.v2_bundle.resolve())), runtime_code_sha256=code_hashes())
    for case in cases:
        case["review_request_sha256"] = base.digest(request_for("review", case, bundle, {}))
    bundle["fingerprint"] = base.digest(bundle)
    root.mkdir(parents=True)
    base.write_json(root / "bundle.json", bundle)
    offline_audit(root, v2_bundle, old_rows, old, parsed)
    return bundle


def load_bundle(root):
    bundle = json.loads((root / "bundle.json").read_text())
    base.require(bundle["fingerprint"] == base.digest({k: v for k, v in bundle.items() if k != "fingerprint"}),
                 "V4 bundle checksum changed")
    base.require(bundle["version"] == VERSION and bundle["runtime_code_sha256"] == code_hashes(), "V4 runtime changed")
    base.require(len(bundle["cases"]) == 2 and bundle["api_budget"] == API_CAP, "Only the four-attempt pilot is allowed")
    for case in bundle["cases"]:
        base.require(case["review_request_sha256"] == base.digest(request_for("review", case, bundle, {})), "Request changed")
    return bundle


def check_sources(bundle):
    paths = {k: Path(v) for k, v in bundle["source_paths"].items()}
    *_, snapshot = read_sources(**paths)
    base.require(snapshot == bundle["source_snapshot"], "Upstream run changed; stop, do not mix runs or budgets")


def identity(bundle):
    return dict(version=VERSION, bundle_fingerprint=bundle["fingerprint"], api_budget=API_CAP)


def read_run(bundle, root):
    rows, meta = read_ledger(root / "run")
    if (root / "run/ledger.sqlite3").exists():
        base.require(json.loads((root / "run/run_identity.json").read_text()) == identity(bundle), "Run marker changed")
        expected = dict(bundle=bundle["fingerprint"], version=VERSION, budget=str(API_CAP))
        expected.update({key: bundle["source_snapshot"]["v3_metadata"][key]
                         for key in ("returned_model", "endpoint_sha256")})
        base.require(all(meta.get(k) == v for k, v in expected.items()), "V4 ledger identity changed")
    return rows


class Ledger(replay.Ledger):
    def __init__(self, root, bundle):
        base.require(not root.is_symlink(), "Run directory cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "run_identity.json"
        if marker.exists():
            base.require(json.loads(marker.read_text()) == identity(bundle), "Run belongs to another experiment")
        else:
            base.require(not any(p.name != ".run.lock" for p in root.iterdir()), "Unidentified run directory")
            base.write_json(marker, identity(bundle))
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
            self.pin("version", VERSION)
            self.pin("budget", str(API_CAP))
            for key in ("returned_model", "endpoint_sha256"):
                self.pin(key, bundle["source_snapshot"]["v3_metadata"][key])
        except BaseException:
            self.db.close()
            raise
        self.limit = API_CAP


def validate_attempts(bundle, rows):
    mapping = {(r["case_id"], r["arm"]): r for r in rows}
    allowed = {(c["case_id"], s) for c in bundle["cases"] for s in STAGES}
    base.require(len(mapping) == len(rows) <= API_CAP and set(mapping) <= allowed, "Invalid V4 ledger")
    results = {}
    for case in bundle["cases"]:
        parsed = {}
        for stage in STAGES:
            row = mapping.get((case["case_id"], stage))
            if row is None:
                continue
            base.require(stage == "review" or "review" in parsed, "Missing valid source review")
            request = request_for(stage, case, bundle, parsed)
            base.require(row["request_hash"] == base.digest(request) and json.loads(row["request_json"]) == request,
                         "Saved dependent request differs from frozen inputs")
            if row["status"] == "valid":
                verify_envelope(row, bundle["source_snapshot"]["v3_metadata"]["returned_model"])
                parsed[stage] = parse(stage, row["raw"], case, parsed)
                base.require(parsed[stage] == json.loads(row["parsed_json"]), "Saved parse differs")
            else:
                base.require(row["parsed_json"] is None, "Failed attempt cannot have a prediction")
        results[case["case_id"]] = parsed
    return results


def execute(bundle, root, factory, max_cases):
    base.require(type(max_cases) is int and 1 <= max_cases <= 2, "--max-cases must be 1 or 2; this is a pilot only")
    with ExitStack() as stack:
        # Cooperate with the existing runners; no simultaneous upstream continuation.
        for path in (Path(bundle["source_paths"]["v2_root"]), Path(bundle["source_paths"]["v3_dir"]) / "run", root / "run"):
            stack.enter_context(replay.run_lock(path))
        check_sources(bundle)
        ledger, client = Ledger(root / "run", bundle), None
        try:
            rows = ledger.rows()
            saved = validate_attempts(bundle, rows)
            base.require(all(r["status"] == "valid" for r in rows), "Failed/interrupted attempt blocks continuation; no retries")
            for case in bundle["cases"][:max_cases]:
                parsed = saved[case["case_id"]]
                for stage in STAGES:
                    if stage in parsed:
                        continue
                    check_sources(bundle)
                    request = request_for(stage, case, bundle, parsed)
                    key = case["case_id"], stage
                    base.require(ledger.reserve(*key, request), "Unexpected prior reservation")
                    try:
                        if client is None:
                            client = factory(ledger)
                        response = client.chat.completions.create(**request)
                    except Exception as exc:
                        ledger.finish(key, "api_error", error=type(exc).__name__)
                        raise RuntimeError("API attempt retained; no retry or downstream call") from exc
                    data = response.model_dump(mode="json")
                    data["provider_request_id"] = getattr(response, "_request_id", None)
                    choice = response.choices[0] if response.choices else None
                    raw = choice.message.content if choice else None
                    ledger.save_response(key, raw, data)
                    try:
                        base.require(isinstance(response.model, str) and response.model, "Missing returned model")
                        ledger.pin("returned_model", response.model)
                        verify_envelope(dict(response_json=base.canonical(data), raw=raw),
                                        bundle["source_snapshot"]["v3_metadata"]["returned_model"])
                        parsed[stage] = parse(stage, raw, case, parsed)
                    except (ValueError, TypeError, KeyError) as exc:
                        ledger.finish(key, "invalid", error=str(exc))
                        raise ValueError("Invalid response retained. Stop and inspect; no retry, repair or negative fallback") from exc
                    ledger.finish(key, "valid", parsed=parsed[stage])
                    print(f"attempt {len(ledger.rows())}/{API_CAP}: {Path(case['case_id']).stem} {stage}", flush=True)
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()
            ledger.close()


def collect(bundle, root):
    with replay.run_lock(root / "run"):
        rows = read_run(bundle, root)
        parsed = validate_attempts(bundle, rows)
        results = []
        for old in bundle["evaluation"]:
            current = parsed.get(old["case_id"], {}).get("final")
            results.append({**old, "v4_status": "valid" if current else "unassessed",
                            "v4_prediction": current["diagnosis"] if current else None,
                            "escalation_required": current["escalation_required"] if current else None})
        assessed = [r for r in results if r["v4_status"] == "valid"]
        prior = bundle["source_snapshot"]["v2_attempts"] + bundle["source_snapshot"]["v3_attempts"]
        report = dict(version=VERSION, scope=SCOPE, api_budget=API_CAP, attempts_reserved=len(rows),
                      known_prior_attempts=prior, known_total_attempts=prior + len(rows),
                      valid=len(assessed), repaired=sum(r["v4_prediction"] == r["truth"] for r in assessed),
                      unassessed=len(results) - len(assessed),
                      failures=[dict(case_id=r["case_id"], stage=r["arm"], status=r["status"], error=r["error"])
                                for r in rows if r["status"] != "valid"],
                      caution="Citation/retention checks are not semantic or clinical validation. No full-cohort F1, significance, or correct-case regression assessment. Saved V3 is noncontemporaneous.")
        base.audit.write_csv(root / "run/case_results.csv", results)
        base.write_json(root / "run/summary.json", report)
        (root / "run/api_receipts.jsonl").write_text("".join(base.canonical(r) + "\n" for r in rows))
        (root / "run/report.txt").write_text(json.dumps(report, indent=2) + "\n")
        for case in bundle["cases"]:
            data = payload(case)
            data["responses"] = parsed[case["case_id"]]
            base.write_json(root / "run" / (Path(case["case_id"]).stem + "_source_review.json"), data)
    print(json.dumps(report, indent=2))
    return report


def offline_audit(root, old, rows, v3_bundle, v3_parsed):
    """Export every error packet and lexical review candidates, not semantic verdicts."""
    out = root / "offline_audit"
    (out / "cases").mkdir(parents=True)
    cases = {c["case_id"]: c for c in old["cases"]}
    candidates = []
    for row in rows:
        if row["group"] != "historical_error":
            continue
        case = cases.get(row["case_id"])
        packet = dict(evaluation=row, saved_evidence=case["evidence"] if case else None)
        base.write_json(out / "cases" / (Path(row["case_id"]).stem + ".json"), packet)
        if not case:
            candidates.append(dict(case_id=row["case_id"], source="missing", source_qualification="", final_candidate=""))
            continue
        final_lines = [s.strip() for s in row.get("v2_reasoning", "").splitlines()
                       if replay.re.search(r"reassur|robust|independent|strongly supports|rules? out", s, replay.re.I)]
        for source, units in contract.source_units(case["evidence"]).items():
            qualifications = [u["text"] for u in units if contract.QUALIFIER.search(u["text"])]
            candidates.append(dict(case_id=row["case_id"], source=source,
                                   source_qualification="\n".join(qualifications), final_candidate="\n".join(final_lines)))
    base.audit.write_csv(out / "qualification_candidates.csv", candidates)
    for c in v3_bundle["cases"][:2]:
        base.write_json(out / "cases" / (Path(c["case_id"]).stem + "_v3.json"),
                        dict(saved_evidence=c["evidence"], v3_responses=v3_parsed[c["case_id"]]))
    base.write_json(out / "summary.json", dict(historical_errors=sum(r["group"] == "historical_error" for r in rows),
        valid_v2_errors=sum(r["group"] == "historical_error" and r["v2_status"] == "valid" for r in rows),
        candidate_rows=len(candidates), confirmed_semantic_error_rate=None,
        note="All error packets exported. Lexical candidates are not adjudicated errors; no clinical labels inferred."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "preflight", "run", "inspect", "collect"), default="preflight")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--v3-dir", type=Path, default=v3.DEFAULT_DIR)
    parser.add_argument("--v2-root", type=Path, default=replay.DEFAULT_RUN)
    parser.add_argument("--v2-bundle", type=Path, default=replay.DEFAULT_BUNDLE)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    bundle = prepare(args) if args.stage == "prepare" else load_bundle(args.experiment_dir)
    snapshot = bundle["source_snapshot"]
    print(f"{VERSION}: {SCOPE}\nTwo cases, two calls/case, hard cap {API_CAP} attempts including failures.\n"
          f"Known prior attempts: {snapshot['v2_attempts']} V2 + {snapshot['v3_attempts']} V3. "
          "No new OCT/SLO/classifier/CDR calls. No automatic retries. Do not resume V3 alongside V4.")
    if args.stage in ("prepare", "preflight"):
        check_sources(bundle)
        rows = read_run(bundle, args.experiment_dir)
        validate_attempts(bundle, rows)
        base.require(all(r["status"] == "valid" for r in rows), "Unresolved attempt; inspect before proceeding")
        print("Frozen inputs, receipts, requests and prior budget verified offline. No client or network calls.")
    elif args.stage == "run":
        base.require(args.allow_api, "Paid run requires --allow-api")
        try:
            execute(bundle, args.experiment_dir, replay.azure_factory(bundle), args.max_cases)
        finally:
            if (args.experiment_dir / "run/ledger.sqlite3").exists():
                collect(bundle, args.experiment_dir)
    elif args.stage == "inspect":
        rows = read_run(bundle, args.experiment_dir)
        validate_attempts(bundle, rows)
        print(json.dumps([v3.response_details(r) for r in rows], indent=2))
    else:
        collect(bundle, args.experiment_dir)


if __name__ == "__main__":
    main()
