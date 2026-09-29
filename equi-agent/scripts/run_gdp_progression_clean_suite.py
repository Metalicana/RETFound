"""Run six independent native OOF/final fits, then the existing six-target agent.

Standalone LLM predictions may be collected, but this script never reruns them.
It never supplies historical test-derived priors or unaudited auxiliary models.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import estimate_gdp_native_oof as native


ROOT = native.ROOT
SCRIPTS = ROOT / "equi-agent/scripts"
TARGETS = native.TARGETS
LLMS = {"gpt51": "GPT-5.1", "gpt54": "GPT-5.4", "gpt56_luna": "GPT-5.6-luna",
        "claude_haiku45": "Claude Haiku 4.5"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "agent", "collect", "all", "status"), default="train")
    parser.add_argument("--run-root", type=Path, default=ROOT / "equi-agent/outputs/gdp_progression_clean_v1")
    parser.add_argument("--primary-run", type=Path, default=ROOT / "equi-agent/outputs/gdp_native_oof_v1")
    parser.add_argument("--native-root", type=Path, default=Path.home() / "Harvard-GDP")
    parser.add_argument("--original-args", type=Path, default=ROOT / "equi-agent/outputs/gdp_native_progression_td_pointwise_no_p_cut_efficientnet_modality2_auc0.8170/args_train.txt")
    parser.add_argument("--manifests-root", type=Path, default=ROOT / "equi-agent/outputs/manifests")
    parser.add_argument("--llm-root", type=Path, default=ROOT / "equi-agent/outputs/baselines/gdp_progression_llm_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deployment", default="gpt-5.1")
    return parser.parse_args()


def target_manifest(args, target):
    return args.manifests_root / f"gdp_progression_forecasting_{target}.csv"


def target_run(args, target):
    return args.primary_run if target == native.TARGET else args.run_root / "native" / target


def check_manifests(args):
    cohorts = {}
    identity = None
    for target in TARGETS:
        dev, test = native.validate_cohort(native.read_csv(target_manifest(args, target)), target=target)
        current = {native.case_id(r): (r["patient_id"], r.get("eye_id", ""), r["split"]) for r in dev + test}
        if identity is not None and current != identity:
            raise ValueError(f"Endpoint IDs, patient grouping or split membership differ: {target}")
        identity = current
        cohorts[target] = (dev, test)
    return cohorts


def progress(args, stage):
    native.write_json(args.run_root / "status.json", {"stage": stage})
    print(f"\n=== {stage} ===", flush=True)


def run_command(command):
    print(" ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def native_command(args, target, prepare=False):
    return [sys.executable, SCRIPTS / "estimate_gdp_native_oof.py",
            "--target", target, "--native-root", args.native_root,
            "--original-args", args.original_args, "--manifest", target_manifest(args, target),
            "--out-dir", target_run(args, target), "--device", args.device,
            "--prepare-only" if prepare else "--fit-final"]


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_bundle(args, target, cohort):
    directory = target_run(args, target)
    config = read_json(directory / "resolved_config.json")
    signature = native.fingerprint(config)
    dev, test = cohort
    if (config["manifest_sha256"] != native.sha256(target_manifest(args, target))
            or config["checkpoint_rule"] != "last_epoch_fixed_in_advance"
            or config["threshold"] != 0.5 or config["recipe"]["num_epochs"] != 60
            or config["folds"] != 5
            or config.get("progression_target", native.TARGET) != target
            or set(config["test_ids_excluded_from_oof"]) != {native.case_id(r) for r in test}):
        raise ValueError(f"Native protocol/cohort mismatch: {target}")
    oof_summary = read_json(directory / "oof_summary.json")
    final_summary = read_json(directory / "final_summary.json")
    if (not oof_summary.get("complete_development_oof") or oof_summary.get("cases") != 300
            or oof_summary.get("test_used_for_fitting_selection_or_priors") is not False
            or not final_summary.get("complete_test_cohort") or final_summary.get("cases") != 200
            or oof_summary["run_fingerprint"] != signature or final_summary["run_fingerprint"] != signature):
        raise ValueError(f"Incomplete or incompatible native run: {target}")
    assignments = config["fold_assignments"]
    if set(assignments) != {native.case_id(r) for r in dev}:
        raise ValueError(f"Incomplete development fold assignments: {target}")
    all_oof = []
    for fold in range(1, config["folds"] + 1):
        heldout = [r for r in dev if assignments[native.case_id(r)] == fold]
        train = [r for r in dev if assignments[native.case_id(r)] != fold]
        if {r["patient_id"] for r in heldout} & {r["patient_id"] for r in train}:
            raise ValueError(f"OOF patient overlap: {target}/{fold}")
        path = directory / f"fold_{fold}/predictions.csv"
        receipt = read_json(path.with_name("complete.json"))
        if (receipt["run_fingerprint"] != signature or receipt["epochs"] != 60
                or receipt["seed"] != config["recipe"]["random_seed"] + fold
                or receipt["checkpoint_rule"] != "last_epoch_fixed_in_advance"
                or receipt["predictions_sha256"] != native.sha256(path)
                or receipt["train_ids"] != [native.case_id(r) for r in train]
                or receipt["heldout_ids"] != [native.case_id(r) for r in heldout]):
            raise ValueError(f"Invalid native fold receipt: {target}/{fold}")
        rows = native.read_csv(path)
        native.validate_predictions(rows, heldout, "oof", fold, signature, target)
        all_oof.extend(rows)
    prefix = f"gdp_progression_forecasting_{target}"
    prior = directory / "priors" / f"exp8_{prefix}_{native.MODEL}" / f"{prefix}_{native.MODEL}_aggregate.csv"
    prior_rows = native.read_csv(prior)
    if len(prior_rows) != 1 or prior_rows[0].get("run_fingerprint") != signature:
        raise ValueError(f"Invalid prior provenance: {target}")
    aggregate = prior_rows[0]
    if aggregate.get("split") != "oof" or aggregate.get("prior_source") != "development_oof" or aggregate.get("progression_target") != target:
        raise ValueError(f"Not target-specific OOF priors: {target}")
    expected = native.metrics(all_oof, "oof")
    for metric in ("n", "f1", "auroc", "balanced_accuracy", "ece", "fpr", "fnr", "sensitivity", "specificity"):
        if not math.isclose(float(aggregate[metric]), float(expected[metric]), abs_tol=1e-10):
            raise ValueError(f"Prior differs from OOF predictions: {target}/{metric}")
    prediction = directory / "predictions" / f"{prefix}_{native.MODEL}.csv"
    rows = native.read_csv(prediction)
    native.validate_predictions(rows, test, "test", "final", signature, target)
    receipt = read_json(directory / "final/complete.json")
    if (receipt["run_fingerprint"] != signature or receipt["epochs"] != 60
            or receipt["seed"] != config["final_seed"]
            or receipt["checkpoint_rule"] != "last_epoch_fixed_in_advance"
            or receipt["predictions_sha256"] != native.sha256(directory / "final/predictions.csv")
            or native.sha256(prediction) != native.sha256(directory / "final/predictions.csv")
            or receipt["checkpoint_sha256"] != native.sha256(directory / "final/model.pt")
            or receipt["train_ids"] != [native.case_id(r) for r in dev]
            or receipt["heldout_ids"] != [native.case_id(r) for r in test]):
        raise ValueError(f"Invalid final-fit receipt: {target}")
    return prediction, prior


def stage_inputs(args, cohorts):
    provenance = {}
    for target in TARGETS:
        prediction, prior = checked_bundle(args, target, cohorts[target])
        destinations = (args.run_root / "predictions" / prediction.name,
                        args.run_root / "priors" / prior.parent.name / prior.name)
        provenance[target] = {}
        for source, destination in zip((prediction, prior), destinations):
            digest = native.sha256(source)
            if destination.exists() and native.sha256(destination) != digest:
                raise ValueError(f"Staged inputs changed; use a new run root: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            provenance[target][str(destination)] = {"source": str(source), "sha256": digest}
    native.write_json(args.run_root / "input_sources.json", provenance)


def agent_command(args, dry_run=False):
    command = [sys.executable, SCRIPTS / "run_equi_agent_gdp_progression_multitarget_live.py",
               "--predictions-root", args.run_root / "predictions", "--metrics-root", args.run_root / "priors",
               "--out-dir", args.run_root / ("agent_join_check" if dry_run else "agent"),
               "--models", native.MODEL, "--targets", *TARGETS, "--case-key-columns", "image_id,task",
               "--expected-cases", "200", "--require-oof-priors", "--expected-prior-cases", "300",
               "--deployment", args.deployment, "--provider", "azure", "--api-version", "2024-12-01-preview",
               "--reference-strategy", "weighted", "--max-consecutive-errors", "3"]
    if dry_run:
        command.extend(["--dry-run", "--max-cases", "1"])
    return command


def align_predictions(path, expected):
    rows = native.read_csv(path)
    index = {native.case_id(r): r for r in rows}
    if len(rows) != len(index) or set(index) != {native.case_id(r) for r in expected}:
        raise ValueError(f"Not the exact locked cohort: {path}")
    aligned = []
    for meta in expected:
        row = index[native.case_id(meta)]
        probability = float(row["y_prob"])
        if (native.label(row) != native.label(meta) or row.get("split") != "test"
                or float(row["y_pred"]) not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError(f"Invalid labels/predictions: {path}/{native.case_id(meta)}")
        aligned.append({**meta, "y_prob": probability, "y_pred": int(float(row["y_pred"]))})
    return aligned


def table_metrics(rows):
    sys.path.insert(0, str(ROOT / "equi-agent"))
    from src.metrics.classification import binary_classification_metrics
    values = binary_classification_metrics([native.label(r) for r in rows], [r["y_prob"] for r in rows],
                                           y_pred=[r["y_pred"] for r in rows])
    groups = []
    eligible_f1 = []
    # Prespecified available single-attribute demographics; no unstable fallback.
    for attribute in ("race", "sex_gender", "age_group"):
        bins = defaultdict(list)
        for row in rows:
            value = row.get(attribute, "").strip().lower()
            if value not in {"", "unknown", "missing", "nan", "none", "-1"}:
                bins[value].append(row)
        attribute_groups = []
        for value, members in sorted(bins.items()):
            positives = sum(native.label(r) for r in members)
            negatives = len(members) - positives
            tp = sum(native.label(r) == 1 and r["y_pred"] == 1 for r in members)
            fp = sum(native.label(r) == 0 and r["y_pred"] == 1 for r in members)
            fn = positives - tp
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            attribute_groups.append({"attribute": attribute, "group": value, "n": len(members),
                                     "n_positive": positives, "n_negative": negatives, "f1": f1,
                                     "eligible": positives >= 20 and negatives >= 20})
        eligible = [r["f1"] for r in attribute_groups if r["eligible"]]
        if len(eligible) >= 2:
            eligible_f1.extend(eligible)
        groups.extend(attribute_groups)
    values["worst_group_f1"] = min(eligible_f1) if eligible_f1 else None
    return values, groups


def collect(args, cohorts):
    summary = read_json(args.run_root / "agent/summary.json")
    if summary.get("dry_run") is not False or not summary.get("complete_live_cohort"):
        raise ValueError("Agent is incomplete or dry-run; no main results exported")
    results, groups, skipped = [], [], []
    latex = []
    markdown = ["# Clean GDP Progression Results", "",
                "Same 200 cases per endpoint; binary positive-class F1; forced predictions include escalations.",
                "Worst-group F1 uses race, sex/gender and age groups, with >=20 positives and >=20 negatives per group and >=2 eligible groups per attribute; otherwise N/A.",
                "Existing standalone LLM rows are reused only after full-cohort/label checks, not retrained or supplied as agent evidence.", ""]
    for target in TARGETS:
        expected = cohorts[target][1]
        prefix = f"gdp_progression_forecasting_{target}"
        methods = [("Native helper", args.run_root / "predictions" / f"{prefix}_{native.MODEL}.csv"),
                   ("Ours", args.run_root / "agent" / f"predictions_{target}.csv")]
        for slug, name in LLMS.items():
            directory = args.llm_root / slug
            try:
                llm_summary = read_json(directory / "summary.json")
                if llm_summary.get("dry_run") is not False or not llm_summary.get("complete_locked_cohort"):
                    raise ValueError("Incomplete/dry-run standalone LLM")
                path = directory / f"predictions_{target}.csv"
                align_predictions(path, expected)
                methods.append((name, path))
            except (OSError, ValueError, KeyError) as error:
                skipped.append({"target": target, "method": name, "reason": str(error)})
        markdown.extend([f"## {target}", "", f"N=200; positives={sum(native.label(r) for r in expected)}.", "",
                         "| Method | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy |",
                         "|---|---:|---:|---:|---:|---:|"])
        escaped_target = target.replace("_", r"\_")
        latex.extend([r"\begin{table}[t]", r"\centering",
                      r"\caption{GDP progression: \texttt{" + escaped_target
                      + r"}. All methods are scored on the same 200 cases. F1 is positive-class F1. "
                      + r"Worst-group F1 requires at least 20 positives and 20 negatives per group and two eligible groups per demographic attribute; otherwise N/A.}",
                      r"\label{tab:gdp-clean-" + target.replace("_", "-") + "}",
                      r"\begin{tabular}{lccccc}", r"\toprule",
                      r"Model & F1 & Worst-group F1 & Sensitivity & Specificity & Balanced Acc. \\", r"\midrule"])
        for name, path in methods:
            values, subgroups = table_metrics(align_predictions(path, expected))
            result = {"target": target, "method": name, "source": str(path), **values}
            results.append(result)
            groups.extend({"target": target, "method": name, **r} for r in subgroups)
            text = ["N/A" if values[k] is None else f"{values[k]:.4f}" for k in (
                "f1", "worst_group_f1", "sensitivity", "specificity", "balanced_accuracy")]
            markdown.append("| " + name + " | " + " | ".join(text) + " |")
            latex.append(name + " & " + " & ".join(text) + r" \\")
        markdown.append("")
        latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    if skipped:
        markdown.extend(["## Unavailable Standalone Baselines", "",
                         *[f"- {r['target']} / {r['method']}: {r['reason']}" for r in skipped]])
    destination = args.run_root / "results"
    native.write_csv(destination / "results.csv", results)
    if groups:
        native.write_csv(destination / "subgroups.csv", groups)
    native.write_json(destination / "unavailable_llm_baselines.json", skipped)
    (destination / "results.md").write_text("\n".join(markdown) + "\n")
    (destination / "results.tex").write_text("\n".join(latex) + "\n")
    print("\n".join(markdown), flush=True)


def execute(args):
    cohorts = check_manifests(args)
    if args.stage in {"agent", "all"}:
        # Client construction verifies SDK/credentials; it does not send a query.
        from run_equi_agent_gdp_progression_live import make_client
        _, client = make_client("azure", "2024-12-01-preview")
        client.close()
    if args.stage in {"prepare", "train", "all"}:
        for target in TARGETS:
            progress(args, f"prepare/{target}")
            run_command(native_command(args, target, prepare=True))
        if args.stage != "prepare":
            for target in (native.TARGET, *[t for t in TARGETS if t != native.TARGET]):
                progress(args, f"native_oof_and_final/{target}")
                run_command(native_command(args, target))
    if args.stage in {"train", "agent", "collect", "all"}:
        progress(args, "validate_and_stage_native_outputs")
        stage_inputs(args, cohorts)
    if args.stage in {"agent", "all"}:
        progress(args, "agent_join_check_no_api")
        run_command(agent_command(args, dry_run=True))
        progress(args, "agent_all_six_targets")
        run_command(agent_command(args))
    if args.stage in {"agent", "collect", "all"}:
        progress(args, "collect")
        collect(args, cohorts)
    progress(args, f"complete/{args.stage}")


def main():
    args = parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        path = args.run_root / "status.json"
        print(path.read_text() if path.exists() else "No recorded stage; check run.log and run.pid")
        return
    with (args.run_root / "suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another clean suite process holds this run directory")
        try:
            execute(args)
        except Exception as error:
            previous = read_json(args.run_root / "status.json") if (args.run_root / "status.json").exists() else {}
            native.write_json(args.run_root / "status.json", {**previous, "failed": True, "error": str(error)})
            raise


if __name__ == "__main__":
    main()
