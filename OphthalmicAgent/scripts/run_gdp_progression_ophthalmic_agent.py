"""Run the staged OphthalmicAgent GDP adaptation without retraining any model.

prepare: validate all six clean native runs and baseline inputs, no API calls.
smoke: execute one complete patient through the real pipeline, resumable by run.
run: complete all 200 patients; collect refuses partial or invalid cohorts.
prompts: export the exact original/adapted system prompts side by side.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent"))

from Progression import evidence, reporting
from Progression.prompts import ENDPOINTS, SYSTEM_PROMPTS, VERSION
from Progression.workflow import CachedCaller, digest, response_format, run_case, write_json


# Only this reviewed JSON-mode implementation can upgrade a failed, zero-case run.
LEGACY_OUTPUT_IMPLEMENTATION = {
    "OphthalmicAgent/Progression/workflow.py": "34d9398adeff06d80b98f9a81d92b6397f9d20b25f888e79d8df1d5b3b07f3f2",
    "OphthalmicAgent/scripts/run_gdp_progression_ophthalmic_agent.py": "e8d723dd9e2525c21e6ba0258de0d2ab0ed1d1146af34a6265cda687e6f45d56",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("prepare", "smoke", "run", "collect", "status", "prompts"), default="prepare")
    parser.add_argument("--clean-root", type=Path, default=ROOT / "equi-agent/outputs/gdp_progression_clean_v1")
    parser.add_argument("--primary-run", type=Path, default=ROOT / "equi-agent/outputs/gdp_native_oof_v1")
    parser.add_argument("--manifests-root", type=Path, default=ROOT / "equi-agent/outputs/manifests")
    parser.add_argument("--llm-root", type=Path, default=ROOT / "equi-agent/outputs/baselines/gdp_progression_llm_v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "OphthalmicAgent/outputs/gdp_progression_staged_v1")
    parser.add_argument("--include-oct", action="store_true", help="Additional-input experiment; requires GDP Bscan files. Not input-matched to native helper.")
    parser.add_argument("--deployment", default="gpt-5.1")
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--upgrade-output-contract", action="store_true",
                        help="Upgrade a failed v1 run with zero completed cases to strict final-response JSON. Keeps unchanged upstream stage caches and archives original provenance.")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    return args


def azure_configuration(args):
    from llm_api_config import config_for_model, public_config
    return public_config(config_for_model("gpt-5.1", args.deployment))


def make_client(args):
    from openai import AzureOpenAI
    from llm_api_config import require_shared_api_key
    config = azure_configuration(args)
    return AzureOpenAI(azure_endpoint=config["endpoint"], api_version=config["api_version"],
                       api_key=require_shared_api_key(), timeout=180, max_retries=0)


def upgrade_failed_output_contract(args, saved, config, fingerprint):
    previous = {k: v for k, v in saved.items() if k != "fingerprint"}
    if saved.get("fingerprint") != digest(previous):
        raise ValueError("Invalid saved configuration fingerprint; cannot upgrade")
    expected_previous = {k: v for k, v in config.items() if k != "response_formats"}
    expected_previous["implementation"] = {**config["implementation"], **LEGACY_OUTPUT_IMPLEMENTATION}
    if previous != expected_previous:
        raise ValueError("Output-contract upgrade requires the original v1 code and identical prompts, evidence and settings")
    if (any((args.out_dir / "cases").glob("*.json"))
            or any(args.out_dir.glob("predictions_*.csv"))
            or any((args.out_dir / "stage_cache").glob("*/orchestrator.json"))
            or (args.out_dir / "results").exists()):
        raise ValueError("Cannot upgrade a run with completed final predictions; use a new --out-dir")
    summary_path = args.out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("completed_cases", 0) != 0 or summary.get("complete_live_cohort"):
            raise ValueError("Cannot upgrade a completed or partially completed cohort")
    backup = args.out_dir / "resolved_config.before_output_contract.json"
    if backup.exists() and json.loads(backup.read_text()) != saved:
        raise ValueError("An incompatible output-contract backup already exists")
    write_json(backup, saved)
    write_json(args.out_dir / "output_contract_upgrade.json", {
        "previous_fingerprint": saved["fingerprint"], "new_fingerprint": fingerprint,
        "change": "orchestrator response_format: json_object -> strict json_schema",
        "clinical_prompts_changed": False, "previous_config": str(backup),
        "retained_upstream_cache_files": sorted(str(p.relative_to(args.out_dir))
                    for p in (args.out_dir / "stage_cache").glob("*/*.json")),
    })
    print("Upgraded final output contract; retained upstream stage caches and archived original configuration", flush=True)


def freeze_run(args, cases, answers, sources):
    files = list((ROOT / "OphthalmicAgent/Progression").glob("*.py")) + [Path(__file__),
        ROOT / "OphthalmicAgent/EquityAgent/compute_demographic_reliability_score.py",
        ROOT / "equi-agent/scripts/run_gdp_progression_clean_suite.py",
        ROOT / "equi-agent/scripts/estimate_gdp_native_oof.py",
        ROOT / "equi-agent/scripts/run_gdp_progression_llm_baseline.py",
        ROOT / "equi-agent/scripts/llm_api_config.py",
        ROOT / "equi-agent/src/metrics/classification.py"]
    config = {"prompt_version": VERSION, "system_prompts": SYSTEM_PROMPTS,
              "api": azure_configuration(args), "include_oct": args.include_oct,
              "targets": list(ENDPOINTS), "case_ids": [c["case_id"] for c in cases],
              "calls_per_case": 6 if args.include_oct else 5,
              "evidence_sha256": digest(cases), "sources": sources,
              "implementation": {str(p.relative_to(ROOT)): evidence.native.sha256(p) for p in files},
              "input_protocol": "RNFLT + baseline TDS + descriptive/reliability-only demographics" + (" + OCT" if args.include_oct else ""),
              "label_policy": "direct_orchestrator_label; no probability clamp, rethreshold, confidence bypass or helper lock",
              "test_used_for_fitting_selection_or_priors": False,
              "response_formats": {stage: response_format(stage) for stage in SYSTEM_PROMPTS}}
    fingerprint = digest(config)
    path = args.out_dir / "resolved_config.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != {**config, "fingerprint": fingerprint}:
            if getattr(args, "upgrade_output_contract", False):
                upgrade_failed_output_contract(args, saved, config, fingerprint)
            else:
                raise ValueError("Run inputs, prompts, settings or implementation changed; use a new --out-dir. "
                                 "For the failed v1 smoke with zero completed cases, use --upgrade-output-contract.")
    elif any((args.out_dir / name).exists() for name in ("stage_cache", "cases", "summary.json")):
        raise ValueError("Refusing unprovenanced existing outputs; use an empty --out-dir")
    write_json(path, {**config, "fingerprint": fingerprint})
    write_json(args.out_dir / "prepared_cases.json", cases)
    write_json(args.out_dir / "evaluation_labels.json", answers)
    write_json(args.out_dir / "prompt_snapshot.json", SYSTEM_PROMPTS)
    return fingerprint


def execute(args):
    if args.stage == "prompts":
        from Progression.prompt_review import export
        print(export(ROOT / "OphthalmicAgent/docs/gdp_progression_prompt_comparison.html"))
        return
    if args.stage == "status":
        path = args.out_dir / "summary.json"
        print(path.read_text() if path.exists() else "No summary yet; inspect run.log")
        return
    if args.stage == "collect":
        # Revalidate inputs too, rather than trusting stale result rows.
        print("Validating saved evidence and native receipts before collection", flush=True)
    else:
        print("Validating six native runs and all 200 baseline cases; no API calls yet", flush=True)
    cases, answers, sources = evidence.prepare(args)
    fingerprint = freeze_run(args, cases, answers, sources)
    case_ids = [c["case_id"] for c in cases]
    completed = reporting.completed_cases(args, case_ids, fingerprint)
    if args.stage == "prepare":
        write_json(args.out_dir / "preflight.json", {"validated_cases": len(cases), "targets": list(ENDPOINTS),
                   "api_calls": 0, "calls_per_case": 6 if args.include_oct else 5, "fingerprint": fingerprint})
        print(f"Preflight passed: {len(cases)} cases, six endpoints, no API calls. Prompts: {args.out_dir / 'prompt_snapshot.json'}")
        return
    errors = []
    caller = CachedCaller(args.out_dir, args.deployment, lambda: make_client(args), max_attempts=args.max_attempts)
    try:
        if args.stage in {"smoke", "run"}:
            requested = cases[:1] if args.stage == "smoke" else cases
            for index, case in enumerate(requested, 1):
                case_id = case["case_id"]
                if case_id in completed:
                    print(f"skip {index}/{len(requested)} case={case_id}", flush=True)
                    continue
                try:
                    result = {**run_case(case, caller), "run_fingerprint": fingerprint}
                    write_json(args.out_dir / "cases" / f"{case_id}.json", result)
                    completed[case_id] = result
                    print(f"done {index}/{len(requested)} case={case_id}", flush=True)
                except Exception as error:
                    errors.append({"case_id": case_id, "error": str(error)})
                    # Stop on the first exhausted stage; do not spend on a broken 200-case run.
                    raise
    finally:
        caller.close()
        reporting.export_predictions(args, answers, completed)
        write_json(args.out_dir / "summary.json", {
            "pipeline": "OphthalmicAgent staged progression", "prompt_version": VERSION,
            "run_fingerprint": fingerprint, "deployment": args.deployment, "dry_run": False,
            "stage": args.stage, "expected_cases": len(cases), "completed_cases": len(completed),
            "missing_cases": sorted(set(case_ids) - set(completed)), "new_errors": errors,
            "complete_live_cohort": set(completed) == set(case_ids),
            "targets": list(ENDPOINTS), "include_oct": args.include_oct})
    if args.stage in {"run", "collect"}:
        reporting.collect(args, answers, completed)
    else:
        print("One-case staged smoke completed. No performance table on this partial cohort. --stage run resumes cached stages.", flush=True)


def main():
    args = parse_args()
    if args.stage in {"prompts", "status"}:
        execute(args)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another staged progression process is using this output directory")
        execute(args)


if __name__ == "__main__":
    main()
