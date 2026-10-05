"""Prepare, verify and run the committed RetinAgent confirmation experiment."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "OphthalmicAgent"), str(ROOT / "OphthalmicAgent/scripts"), str(ROOT / "equi-agent/scripts")]
import run_fairvision_ablation as base
from Confirmation import design


def external_cases(args):
    results, sources = [], {}
    manifest = ROOT / "OphthalmicAgent/data_drishti/manifest.csv"
    official = {r["case_id"]: r for r in base.read_csv(manifest) if r["split"] == "test"}
    sources[str(manifest)] = base.sha(manifest)
    for dataset, rel in (
        ("drishti", "drishti/drishti_retfound_cfp_predictions.csv"),
        ("refuge2", "refuge/refuge_test_retfound_cfp_predictions.csv")):
        path = ROOT / "OphthalmicAgent/results" / rel
        sources[str(path)] = base.sha(path)
        rows = [r for r in base.read_csv(path) if r.get("split", "test").lower() == "test"]
        keys = set()
        for r in rows:
            key = r.get("case_id") or Path(r["Filename"]).stem
            base.require(key not in keys, f"Duplicate external case {dataset}/{key}")
            keys.add(key)
            truth = base.binary(r["Ground_Truth"])
            if dataset == "drishti":
                base.require(key in official and base.binary(official[key]["label"]) == truth, "Drishti cohort/label mismatch")
                image = official[key]["cfp_path"]
            else:
                image = r["Filename"]
                base.require("Test" in Path(image).parts, "REFUGE2 must use its full test cohort")
            results.append(dict(dataset=dataset, case_id=key, truth=truth,
                cfp_path=str((args.external_root / image).resolve()), probability=base.probability(r["Probability_GL"])))
        base.require(len(keys) == design.read_protocol()["external_counts"][dataset], f"Incomplete {dataset}")
        if dataset == "drishti":
            base.require(keys == set(official), "Different Drishti test IDs")
    return results, sources


def prepare(args):
    commit, lock = design.verify_freeze()
    p = design.read_protocol()
    root = args.run_root
    if (root / "confirmation.json").exists():
        read_run(args)
        print("Existing frozen preparation verified; no new calls or changed cohort.")
        return
    base.require(not (root / "config.json").exists(), "Partial preparation exists; inspect it before using a fresh run root")
    external, external_sources = external_cases(args)
    selection = json.loads((design.DESIGN / "selection_receipt.json").read_text())
    # Detect known additional CECSL exposure before any paid call. Never reselect
    # a cohort silently on a machine containing extra historical results.
    excluded = {t: set(ids) for t, ids in selection["exclusions"].items()}
    fresh = base.load_locked(design.DESIGN / "cohorts.csv")
    for path in (ROOT / "OphthalmicAgent/results/fairvision").rglob("*agentic*.csv"):
        for row in base.read_csv(path):
            task = (row.get("Task_Folder") or row.get("Disease") or
                    next((t for t in base.TASKS if f"_{t}_" in path.name), "")).lower()
            key = Path(row.get("Filename") or row.get("filename", "")).stem
            base.require(task in excluded and key not in fresh[task], f"Prior live exposure overlaps frozen cohort: {path}/{key}")
    old_args = SimpleNamespace(run_root=root, locked_csv=design.DESIGN / "cohorts.csv",
        models=list(base.MODELS), predictions_root=args.predictions_root, data_root=args.data_root,
        oct_weights=args.oct_weights, deployment=p["deployment"], api_version=p["api_version"])
    with contextlib.redirect_stdout(io.StringIO()):
        base.prepare(old_args, report=False)
    config, cases, _ = base.load_prepared(root)
    config.update(frozen_generation=p["generation"], confirmation_source_lock=lock,
                  confirmation_protocol=base.digest(p), confirmation_commit=commit)
    config["fingerprint"] = base.digest({k: v for k, v in config.items() if k != "fingerprint"})
    base.write_json(root / "config.json", config)
    base.write_json(root / "external_cases.json", external)
    base.write_json(root / "confirmation.json", dict(commit=commit, source_lock=lock,
        protocol=p, external_sha256=base.digest(external), external_sources=external_sources,
        selection_sha256=base.sha(design.DESIGN / "cohorts.csv")))
    print("Frozen preparation complete: 750 fresh cases, four FairVision arms, 4500 demographic profiles, 451 external cases.")
    print("Expected 20103 successful API calls before retries; 0 API calls made during preparation.")


def read_run(args):
    _, lock = design.verify_freeze()
    root = args.run_root
    config, cases, offline = base.load_prepared(root)
    receipt = json.loads((root / "confirmation.json").read_text())
    base.require(receipt["source_lock"] == lock == config["confirmation_source_lock"], "Frozen source lock changed")
    base.require(receipt["protocol"] == design.read_protocol(), "Protocol changed")
    base.require(config["frozen_generation"] == receipt["protocol"]["generation"], "Generation settings changed")
    base.require(config["deployment"] == receipt["protocol"]["deployment"] and
                 config["api_version"] == receipt["protocol"]["api_version"], "Deployment/API settings changed")
    base.require(base.sha(design.DESIGN / "cohorts.csv") == receipt["selection_sha256"], "Cohort changed")
    validation = json.loads((root / "validation_cases.json").read_text())
    base.require(base.digest(validation) == config["validation_sha256"], "Validation cohort changed")
    external = json.loads((root / "external_cases.json").read_text())
    base.require(base.digest(external) == receipt["external_sha256"], "External evidence changed")
    for path, expected in {**config["source_hashes"], **receipt["external_sources"]}.items():
        base.require(base.sha(Path(path)) == expected, f"Prepared source changed: {path}")
    return config, cases, validation, external


def preflight(args):
    import numpy as np
    from build_manifests import normalize_binary_label, require_data_libs
    require_data_libs()
    config, cases, validation, external = read_run(args)
    print(f"Checking {len(cases)+len(validation)} FairVision images, {len(external)} external images and checkpoint; no API calls", flush=True)
    images = base.require_images(Path(config["data_root"]), cases + validation)
    weights = Path(config["oct_weights"])
    base.require(weights.is_file(), f"Missing OCT checkpoint {weights}")
    hashes = {}
    for row in cases + validation:
        path = images[row["filename"]]
        key = row["task"] + "/" + Path(row["filename"]).stem
        hashes[key] = base.sha(path)
        field = dict(glaucoma="glaucoma", amd="amd_condition", dr="dr_subtype")[row["task"]]
        with np.load(path, allow_pickle=False) as data:
            base.require({"oct_bscans", "slo_fundus", field} <= set(data.files), f"Missing image/label fields in {path}")
            truth = base.binary(normalize_binary_label(data[field].item(), row["task"]))
            base.require(truth == row.get("truth", row.get("y_true")), f"NPZ/reference label mismatch: {path}")
    for row in external:
        path = Path(row["cfp_path"])
        base.require(path.is_file(), f"Missing external CFP: {path}")
        hashes[row["dataset"] + "/" + row["case_id"]] = base.sha(path)
    receipt = dict(config_fingerprint=config["fingerprint"], weights_sha256=base.sha(weights), images=hashes,
                   external_sha256=base.digest(external), source_lock=config["confirmation_source_lock"])
    design.locked_json(args.run_root / "input_receipt.json", receipt)
    print(f"Preflight passed: {len(images)} FairVision images, {len(external)} external CFPs, NPZ reference labels verified. No API calls.")
    return config, cases, validation, external, base.digest(receipt)


def run(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "OphthalmicAgent/.env")
    config, cases, validation, external, receipt = preflight(args)
    from Confirmation import workflow
    from importlib.metadata import version
    environment = dict(python=sys.version, platform=platform.platform(), packages={name: version(name) for name in
        ("openai", "numpy", "pandas", "scipy", "torch", "torchvision", "timm", "transformers", "Pillow", "python-dotenv")})
    design.locked_json(args.run_root / "environment.json", environment)
    # CDR preprocessing is local inference only. Complete it before paid calls.
    workflow.external_tools(args.run_root, external, receipt, args.device)
    client = workflow.make_client(args.run_root, config, args.max_api_calls)
    workflow.fairvision(args.run_root, config, cases, validation, receipt, client, args.device)
    workflow.external(args.run_root, config, external, receipt, client)
    collect(args)


def collect(args):
    config, cases, _, external = read_run(args)
    from Confirmation.reporting import collect_complete
    collect_complete(args.run_root, config, cases, external)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("select", "freeze", "prepare", "preflight", "run", "collect"))
    parser.add_argument("--run-root", type=Path, default=ROOT / "OphthalmicAgent/outputs/fairvision_confirmation_v1")
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "equi-agent/outputs/predictions")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Datasets/FairVision")
    parser.add_argument("--external-root", type=Path, default=ROOT / "OphthalmicAgent")
    parser.add_argument("--oct-weights", type=Path, default=ROOT / "OphthalmicAgent/weights/oct_model_8_slices_not_center.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-api-calls", type=int, default=design.read_protocol()["default_max_api_attempts"])
    args = parser.parse_args()
    if args.stage == "select":
        design.select_cohorts()
        return
    if args.stage == "freeze":
        design.freeze()
        return
    for field in ("run_root", "data_root", "predictions_root", "external_root", "oct_weights"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.run_root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.run_root / "confirmation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A confirmation process is already active") from exc
        {"prepare": prepare, "preflight": preflight, "run": run, "collect": collect}[args.stage](args)


if __name__ == "__main__":
    main()
