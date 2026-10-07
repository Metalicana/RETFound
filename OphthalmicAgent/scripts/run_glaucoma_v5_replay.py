"""Six-attempt native-SLO pilot with the existing V4 downstream contract. Offline by default."""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "OphthalmicAgent"), str(ROOT / "OphthalmicAgent/scripts")]
import run_glaucoma_v4_replay as v4
from ReplayV5 import slo_contract as slo

v3, replay, base = v4.v3, v4.replay, v4.base
VERSION = "glaucoma_native_slo_v5_20261007"
STAGES = ("slo", "review", "final")
API_CAP = 6
DEFAULT_DIR = ROOT / "OphthalmicAgent/outputs/glaucoma_v5_replay"
# These pin the two inspected source JPEGs, not case-specific diagnostic rules.
NATIVE_SHA256 = {
    "data_07014": "f909e048efbb0a3578cf1ac7b9c3f05c3f5d91ea9f33c6adbef5264dca681c8d",
    "data_07137": "8ba19db51b5728285b8044cdad1c1abccfcd95473701c35d681d6fa1c3436e2b",
}
SCOPE = ("Two-case historical-error-selected development pilot: native SLO plus structured reporting, "
         "then unchanged V4 qualification review and final decision. Combined input/reporting change; "
         "not an isolated resolution ablation, full pipeline or held-out evaluation.")


def code_hashes():
    return {**v4.code_hashes(), **{str(p.relative_to(ROOT)): base.sha(p)
                                 for p in (Path(__file__), Path(slo.__file__))}}


def read_sources(v4_dir, v3_dir, v2_root, v2_bundle):
    prior = v4.load_bundle(v4_dir)
    rows = v4.read_run(prior, v4_dir)
    parsed = v4.validate_attempts(prior, rows)
    base.require(len(rows) == 4 and all(r["status"] == "valid" for r in rows) and
                 all("final" in p for p in parsed.values()), "Require the completed four-attempt V4 pilot")
    old, oct_parsed, _, _, snapshot = v4.read_sources(v3_dir, v2_root, v2_bundle)
    base.require(snapshot == prior["source_snapshot"], "V2/V3 changed since V4; do not mix runs or budgets")
    expected = [dict(case_id=c["case_id"],
                     evidence={**c["evidence"], "oct_specialist_report": oct_parsed[c["case_id"]]["oct"]["report"]},
                     oct_acquisition=c["image"]["metadata"]) for c in old["cases"][:2]]
    for actual, wanted in zip(prior["cases"], expected):
        base.require(all(actual[k] == value for k, value in wanted.items()), "V4 evidence differs from V3")
    base.require([Path(c["case_id"]).stem for c in prior["cases"]] == list(NATIVE_SHA256),
                 "Only the existing two pilot cases are allowed")
    total = snapshot["v2_attempts"] + snapshot["v3_attempts"] + len(rows)
    base.require(total + API_CAP <= 250, "Known prior attempts plus V5 exceed 250")
    snapshot = {**snapshot, "v4_bundle_fingerprint": prior["fingerprint"],
                "v4_attempts_sha256": base.digest(rows), "v4_attempts": len(rows), "known_prior_attempts": total}
    return prior, parsed, snapshot


def image_bytes(case, root):
    image = case["native_slo"]
    expected = f"images/{Path(case['case_id']).stem}.jpg"
    base.require(image["path"] == expected, "Unexpected frozen image path")
    path = root / expected
    base.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                 "Frozen SLO image is missing or indirect")
    data = path.read_bytes()
    base.require(hashlib.sha256(data).hexdigest() == image["sha256"] == NATIVE_SHA256[Path(case["case_id"]).stem],
                 "Native SLO image changed or does not match the inspected case")
    base.require(slo.image_metadata(data) == image["metadata"], "SLO image metadata changed")
    return data


def updated_case(case, parsed):
    base.require("slo" in parsed, "A validated new SLO report is required")
    return {**case, "evidence": {**case["evidence"], "slo_specialist_report": slo.report(parsed["slo"])}}


def request_for(stage, case, bundle, root, parsed):
    if stage == "slo":
        return slo.request(image_bytes(case, root), bundle["settings"]["deployment"])
    base.require(stage in ("review", "final"), "Unknown V5 stage")
    request = v4.request_for(stage, updated_case(case, parsed), bundle, parsed)
    data = json.loads(request["messages"][1]["content"])
    # Only input/provenance changes. V4 instructions, schema, settings and parser are reused verbatim.
    data["measurement_provenance"].update(slo_images_reassessed=True,
        slo_image=case["native_slo"]["metadata"], cdr_recomputed=False,
        cdr_input="Cached 200x200 NPZ derivative; current SLO report uses the native JPEG of the same acquisition",
        same_acquisition_not_identical_slo_pixels=True)
    request["messages"][1]["content"] = base.canonical(data)
    return request


def parse(stage, raw, case, parsed):
    return slo.parse(raw) if stage == "slo" else v4.parse(stage, raw, updated_case(case, parsed), parsed)


def prepare(args):
    root = args.experiment_dir.resolve()
    paths = {key: getattr(args, key).resolve() for key in ("v4_dir", "v3_dir", "v2_root", "v2_bundle")}
    for source in (*[p for k, p in paths.items() if k != "v2_bundle"], paths["v2_bundle"].parent, args.image_dir.resolve()):
        base.require(not root.is_relative_to(source) and not source.is_relative_to(root), "Use a separate V5 directory")
    base.require(not root.exists(), "V5 already exists; use preflight/resume, never reprepare to retry")
    prior, parsed, snapshot = read_sources(**paths)
    cases, images = [], {}
    for old in prior["cases"]:
        key = Path(old["case_id"]).stem
        source = args.image_dir / ("slo_fundus_" + key.removeprefix("data_") + ".jpg")
        base.require(source.is_file() and not source.is_symlink(), f"Missing native SLO: {source}")
        data = source.read_bytes()
        base.require(hashlib.sha256(data).hexdigest() == NATIVE_SHA256[key], "Native JPEG differs from inspected source")
        metadata = slo.image_metadata(data)
        case = {k: copy.deepcopy(old[k]) for k in ("case_id", "evidence", "oct_acquisition")}
        case["native_slo"] = dict(path=f"images/{key}.jpg", sha256=NATIVE_SHA256[key], metadata=metadata,
                                  source_path=str(source.resolve()))
        cases.append(case)
        images[case["native_slo"]["path"]] = data
    evaluation = copy.deepcopy(prior["evaluation"])
    for row in evaluation:
        final = parsed.get(row["case_id"], {}).get("final")
        row.update(v4_prediction=final["diagnosis"] if final else None, v5_selected=final is not None)
    bundle = dict(version=VERSION, scope=SCOPE, api_budget=API_CAP, cases=cases, evaluation=evaluation,
                  settings=prior["settings"], source_paths={k: str(p) for k, p in paths.items()},
                  source_snapshot=snapshot, runtime_code_sha256=code_hashes())
    for case in cases:
        case["slo_request_sha256"] = base.digest(slo.request(images[case["native_slo"]["path"]], bundle["settings"]["deployment"]))
    bundle["fingerprint"] = base.digest(bundle)
    (root / "images").mkdir(parents=True)
    for path, data in images.items():
        (root / path).write_bytes(data)
    base.write_json(root / "bundle.json", bundle)
    return load_bundle(root)


def load_bundle(root):
    bundle = json.loads((root / "bundle.json").read_text())
    base.require(bundle["fingerprint"] == base.digest({k: v for k, v in bundle.items() if k != "fingerprint"}),
                 "V5 bundle checksum changed")
    base.require(bundle["version"] == VERSION and bundle["runtime_code_sha256"] == code_hashes(), "V5 runtime changed")
    base.require(bundle["api_budget"] == API_CAP and len(bundle["cases"]) == 2 and
                 [Path(c["case_id"]).stem for c in bundle["cases"]] == list(NATIVE_SHA256), "Only the six-attempt pilot is allowed")
    base.require(bundle["source_snapshot"]["known_prior_attempts"] + API_CAP <= 250, "Known budget exceeds 250")
    for case in bundle["cases"]:
        base.require(case["slo_request_sha256"] == base.digest(request_for("slo", case, bundle, root, {})),
                     "Frozen SLO request changed")
    return bundle


def check_sources(bundle):
    _, _, snapshot = read_sources(**{k: Path(v) for k, v in bundle["source_paths"].items()})
    base.require(snapshot == bundle["source_snapshot"], "Upstream run changed; stop rather than mixing experiments")
    for case in bundle["cases"]:
        image = case["native_slo"]
        base.require(base.sha(Path(image["source_path"])) == image["sha256"], "Original native SLO changed")


def identity(bundle):
    return dict(version=VERSION, bundle_fingerprint=bundle["fingerprint"], api_budget=API_CAP)


def expected_metadata(bundle):
    pins = bundle["source_snapshot"]["v3_metadata"]
    return dict(bundle=bundle["fingerprint"], version=VERSION, budget=str(API_CAP),
                returned_model=pins["returned_model"], endpoint_sha256=pins["endpoint_sha256"])


def read_run(bundle, root):
    rows, meta = v4.read_ledger(root / "run")
    if (root / "run/ledger.sqlite3").exists():
        base.require(json.loads((root / "run/run_identity.json").read_text()) == identity(bundle), "Run marker changed")
        base.require(all(meta.get(k) == v for k, v in expected_metadata(bundle).items()), "V5 ledger identity changed")
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
            for key, value in expected_metadata(bundle).items():
                self.pin(key, value)
        except BaseException:
            self.db.close()
            raise
        self.limit = API_CAP


def validate_attempts(bundle, root, rows):
    mapping = {(r["case_id"], r["arm"]): r for r in rows}
    allowed = {(c["case_id"], s) for c in bundle["cases"] for s in STAGES}
    base.require(len(mapping) == len(rows) <= API_CAP and set(mapping) <= allowed, "Invalid V5 ledger")
    results = {}
    for case in bundle["cases"]:
        parsed = {}
        for i, stage in enumerate(STAGES):
            row = mapping.get((case["case_id"], stage))
            if row is None:
                continue
            base.require(all(s in parsed for s in STAGES[:i]), "Missing valid upstream stage")
            request = request_for(stage, case, bundle, root, parsed)
            base.require(row["request_hash"] == base.digest(request) and json.loads(row["request_json"]) == request,
                         "Saved request differs from frozen inputs")
            if row["status"] == "valid":
                v4.verify_envelope(row, expected_metadata(bundle)["returned_model"])
                parsed[stage] = parse(stage, row["raw"], case, parsed)
                base.require(parsed[stage] == json.loads(row["parsed_json"]), "Saved parse differs")
            else:
                base.require(row["parsed_json"] is None, "Failed attempt cannot have a prediction")
        results[case["case_id"]] = parsed
    return results


def execute(bundle, root, factory, max_cases):
    base.require(type(max_cases) is int and 1 <= max_cases <= 2, "--max-cases must be 1 or 2; no cohort expansion")
    with ExitStack() as stack:
        paths = bundle["source_paths"]
        for path in (Path(paths["v2_root"]), Path(paths["v3_dir"]) / "run", Path(paths["v4_dir"]) / "run", root / "run"):
            stack.enter_context(replay.run_lock(path))
        check_sources(bundle)
        read_run(bundle, root)
        ledger, client = Ledger(root / "run", bundle), None
        try:
            rows = ledger.rows()
            saved = validate_attempts(bundle, root, rows)
            base.require(all(r["status"] == "valid" for r in rows), "Failed/interrupted attempt blocks continuation; no retries")
            for case in bundle["cases"][:max_cases]:
                parsed = saved[case["case_id"]]
                for stage in STAGES:
                    if stage in parsed:
                        continue
                    check_sources(bundle)
                    request = request_for(stage, case, bundle, root, parsed)
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
                        v4.verify_envelope(dict(response_json=base.canonical(data), raw=raw), expected_metadata(bundle)["returned_model"])
                        parsed[stage] = parse(stage, raw, case, parsed)
                    except (ValueError, TypeError, KeyError) as exc:
                        ledger.finish(key, "invalid", error=str(exc))
                        raise ValueError("Invalid response retained; stop and inspect, no retry or negative fallback") from exc
                    ledger.finish(key, "valid", parsed=parsed[stage])
                    print(f"attempt {len(ledger.rows())}/{API_CAP}: {Path(case['case_id']).stem} {stage}", flush=True)
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()
            ledger.close()


def collect(bundle, root):
    with replay.run_lock(root / "run"):
        rows = read_run(bundle, root)
        parsed = validate_attempts(bundle, root, rows)
        results = []
        for old in bundle["evaluation"]:
            final = parsed.get(old["case_id"], {}).get("final")
            results.append({**old, "v5_status": "valid" if final else "unassessed",
                            "v5_prediction": final["diagnosis"] if final else None,
                            "escalation_required": final["escalation_required"] if final else None})
        assessed = [r for r in results if r["v5_status"] == "valid"]
        prior = bundle["source_snapshot"]["known_prior_attempts"]
        report = dict(version=VERSION, scope=SCOPE, api_budget=API_CAP, attempts_reserved=len(rows),
            known_prior_attempts=prior, known_total_attempts=prior + len(rows), expected=len(results),
            selected=len(bundle["cases"]), valid=len(assessed),
            repaired=sum(r["v5_prediction"] == r["truth"] for r in assessed),
            still_wrong=sum(r["v5_prediction"] != r["truth"] for r in assessed), unassessed=len(results)-len(assessed),
            additional_repairs_vs_saved_v4=sum(r["v5_prediction"] == r["truth"] != r["v4_prediction"] for r in assessed),
            lost_repairs_vs_saved_v4=sum(r["v4_prediction"] == r["truth"] != r["v5_prediction"] for r in assessed),
            failures=[dict(case_id=r["case_id"], stage=r["arm"], status=r["status"], error=r["error"])
                      for r in rows if r["status"] != "valid"],
            caution="Descriptive two-case development results only. V4 is a saved noncontemporaneous comparator. No full-cohort F1, significance, clinical validation or correct-case regression assessment. Known receipts are not a global account counter.")
        base.audit.write_csv(root / "run/case_results.csv", results)
        base.write_json(root / "run/summary.json", report)
        (root / "run/api_receipts.jsonl").write_text("".join(base.canonical(r) + "\n" for r in rows))
        for case in bundle["cases"]:
            p = parsed[case["case_id"]]
            base.write_json(root / "run" / (Path(case["case_id"]).stem + "_source_review.json"),
                dict(native_slo=case["native_slo"], evidence=updated_case(case, p)["evidence"] if "slo" in p else None,
                     responses=p))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "preflight", "run", "inspect", "collect"), default="preflight")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--v4-dir", type=Path, default=v4.DEFAULT_DIR)
    parser.add_argument("--v3-dir", type=Path, default=v3.DEFAULT_DIR)
    parser.add_argument("--v2-root", type=Path, default=replay.DEFAULT_RUN)
    parser.add_argument("--v2-bundle", type=Path, default=replay.DEFAULT_BUNDLE)
    parser.add_argument("--image-dir", type=Path, default=v3.DEFAULT_DATA)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    bundle = prepare(args) if args.stage == "prepare" else load_bundle(args.experiment_dir)
    prior = bundle["source_snapshot"]["known_prior_attempts"]
    print(f"{VERSION}: {SCOPE}\nThree requests/case (SLO -> review -> final); hard cap {API_CAP}, including failures.\n"
          f"Known prior attempts {prior}; at most {prior + API_CAP} including V5. Not a global account counter.\n"
          "No new OCT, CDR, classifier, Bio-Profiler, counterfactual or training calls. No automatic retries.\n"
          "V1-V4 are not overwritten. Do not resume upstream runs alongside V5.")
    if args.stage in ("prepare", "preflight"):
        check_sources(bundle)
        rows = read_run(bundle, args.experiment_dir)
        validate_attempts(bundle, args.experiment_dir, rows)
        base.require(all(r["status"] == "valid" for r in rows), "Unresolved attempt; inspect before proceeding")
        print("Frozen JPEGs, source receipts, requests and prior budget verified offline. No API client or network calls.")
    elif args.stage == "run":
        base.require(args.allow_api, "Paid run requires --allow-api")
        try:
            execute(bundle, args.experiment_dir, replay.azure_factory(bundle), args.max_cases)
        finally:
            if (args.experiment_dir / "run/ledger.sqlite3").exists():
                collect(bundle, args.experiment_dir)
    elif args.stage == "inspect":
        rows = read_run(bundle, args.experiment_dir)
        validate_attempts(bundle, args.experiment_dir, rows)
        print(json.dumps([v3.response_details(r) for r in rows], indent=2))
    else:
        collect(bundle, args.experiment_dir)


if __name__ == "__main__":
    main()
