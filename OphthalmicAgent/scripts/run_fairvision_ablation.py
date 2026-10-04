"""Matched FairVision comparison/ablation: prepare offline, then smoke/run on CECSL.

prepare/collect do not import API or imaging clients. Historical result rows are
never substituted for the paired full/no-priors experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "equi-agent/scripts"))
sys.path.insert(0, str(ROOT / "OphthalmicAgent"))
import audit_manuscript_result_gaps as audit
import build_all_model_validation_priors as prior_metrics
import rank_demographic_reliability_models as ranking

TASKS = ("glaucoma", "amd", "dr")
MODELS = tuple(sorted(ranking.FOUNDATION_MODELS))
VARIANTS = ("simple_multimodal_ensemble", "reliability_weighted_fusion",
            "agents_without_reliability", "retinagent_full")
VERSION = "fairvision_paired_ablation_v1"
COEFFICIENTS = dict(fnr_weight=.35, fpr_weight=.25, ece_weight=.15,
                    auroc_weight=.15, f1_weight=.10, k=50.)
DISEASE_FOLDERS = {"glaucoma": "Glaucoma", "amd": "AMD", "dr": "DR"}
# Only the original, unused bundle may adopt this path-only correction.
LEGACY_PATH_CODE = {
    "OphthalmicAgent/scripts/run_fairvision_ablation.py": "70c071cb3c4d30f248348b289da79524657fb1bea54cdf3af3d3b4aa2fbfe786",
    "OphthalmicAgent/Ablation/fairvision_live.py": "a5147fa1bbdbb6dfa1a7a974a3ab7486bbff10ab9d144005ae4973cb24b7c717",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def binary(value):
    result = audit.binary(value)
    require(result is not None, f"Nonbinary label: {value!r}; no automatic repair")
    return result


def probability(value):
    p = float(value)
    require(math.isfinite(p) and 0 <= p <= 1, f"Invalid probability {value!r}")
    return p


def image_candidates(root, row):
    filename = Path(row["filename"])
    require(not filename.is_absolute() and ".." not in filename.parts and len(filename.parts) >= 3,
            f"Expected task/split-qualified relative image path: {filename}")
    disease = DISEASE_FOLDERS[row["task"]]
    require(filename.parts[-3] == disease, f"Task/path mismatch: {row['task']}/{filename}")
    folder = filename.parts[-2]
    require(folder in ("Test", "Validation"), f"Unexpected image split: {filename}")
    require(filename.suffix == ".npz", f"Expected NPZ image: {filename}")
    expected_split = "val" if folder == "Validation" else "test"
    require(row.get("split", expected_split) == expected_split, f"Split/path mismatch: {filename}")
    candidates = [root / filename]
    for base in (root, root / "data", root / "HarvardFairVision30k"):
        candidates.extend((base / disease / folder / filename.name,
                           base / folder / disease / filename.name,
                           base / folder / filename.name))
    return list(dict.fromkeys(candidates))


def resolve_image_path(root, row):
    candidates = image_candidates(Path(root), row)
    # Symlink aliases of one image are harmless; distinct matches need an explicit root.
    matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    require(len(matches) <= 1, f"Ambiguous image {row['filename']}: {matches}; select a more specific --data-root")
    return matches[0] if matches else candidates[0].resolve()


def inspect_images(root, rows):
    resolved, issues = {}, []
    for row in rows:
        try:
            path = resolve_image_path(root, row)
            if path.is_file():
                resolved[row["filename"]] = path
            else:
                issues.append(f"Missing {row['filename']}")
        except ValueError as exc:
            issues.append(str(exc))
    return resolved, issues


def require_images(root, rows):
    resolved, issues = inspect_images(root, rows)
    require(not issues and len(resolved) == len(rows),
            f"Image preflight failed under {root}: {len(resolved)}/{len(rows)} resolved. "
            f"First issues: {issues[:5]}. Use --stage paths to inspect supported dataset roots; no API calls.")
    return resolved


def paths(args):
    config, cases, _ = load_prepared(args.run_root)
    validation = json.loads((args.run_root / "validation_cases.json").read_text())
    require(digest(validation) == config["validation_sha256"], "Validation input changed")
    rows = cases + validation
    roots = dict.fromkeys((args.data_root, Path(config["data_root"]), ROOT / "Datasets/FairVision",
                          ROOT / "Datasets/FairVision/HarvardFairVision30k", ROOT / "OphthalmicAgent/data"))
    for root in roots:
        resolved, issues = inspect_images(root, rows)
        print(f"data_root={root}: {len(resolved)}/{len(rows)} images resolved")
        if issues:
            print("  " + "; ".join(issues[:2]))
        else:
            for task in TASKS:
                for folder in ("Test", "Validation"):
                    selected = [r for r in rows if r["task"] == task and Path(r["filename"]).parts[-2] == folder]
                    if selected:
                        print(f"  {task}/{folder}: {len(selected)}; example={resolved[selected[0]['filename']]}")
    print("Read-only path audit complete. No API calls or configuration changes.")


def demographic(row):
    age = float(row.get("age", row.get("Age", "nan")))
    require(math.isfinite(age) and age >= 0, "Missing/invalid age")
    return dict(age_group="younger" if age < 50 else "middle-aged" if age < 70 else "older",
                race=ranking.norm_race(row.get("race", row.get("Race", ""))),
                sex_gender=ranking.norm_gender(row.get("sex_gender", row.get("Gender", ""))))


def load_locked(path):
    locked = {t: {} for t in TASKS}
    for raw in read_csv(path):
        task = raw["Task_Folder"].strip().lower()
        require(task in TASKS, f"Unknown task {task}")
        case = Path(raw["filename"]).stem
        require(case and case not in locked[task], f"Duplicate case {task}/{case}")
        locked[task][case] = dict(case_id=case, task=task, filename=raw["filename"],
            truth=binary(raw["Ground_Truth"]), metadata={k: raw[k] for k in ("Age", "Gender", "Race", "Ethnicity")},
            **demographic(raw))
    for task, rows in locked.items():
        require(len(rows) == 250, f"Need 250 locked {task} cases, got {len(rows)}")
        require({r["truth"] for r in rows.values()} == {0, 1}, f"One-class {task} cohort")
    return locked


def source_path(root, task, model, split):
    stem = {"retfound_oct": "oct_retfound", "mirage_slo": "slo_mirage"}.get(model)
    return root / (f"fairvision_{stem}_{split}.csv" if stem else f"fairvision_{task}_{model}_{split}.csv")


def load_probabilities(path, task, model, split):
    result = {}
    for raw in read_csv(path):
        if raw["task"].strip().lower() != task:
            continue
        require(raw["split"] == split and raw["model_name"] == model, f"Unexpected model/split in {path}")
        key = Path(raw["image_id"]).stem
        require(key and key not in result, f"Duplicate prediction {path}/{key}")
        result[key] = dict(image_id=key, task=task, model_name=model, split=split,
            y_true=binary(raw["y_true"]), y_prob=probability(raw["y_prob"]), **demographic(raw))
    require(result, f"No {task} predictions in {path}")
    return result


def select_threshold(rows):
    # Fixed grid and deterministic tie break; only validation rows may enter.
    require(rows and all(r["split"] == "val" for r in rows), "Threshold selection requires validation rows")
    scores = []
    for i in range(1, 100):
        threshold = i / 100
        value = audit.metrics([dict(truth=r["y_true"], prediction=int(r["y_prob"] >= threshold)) for r in rows])["f1_macro"]
        scores.append((value, -abs(threshold - .5), threshold))
    return max(scores)[2]


def fit_reliability(rows, task, model, fixed_threshold=None):
    require(all(r["split"] == "val" for r in rows), "Reliability requires validation rows")
    threshold = select_threshold(rows) if fixed_threshold is None else fixed_threshold
    scored = [{**r, "y_pred": r["y_pred"] if fixed_threshold is not None and "y_pred" in r
               else int(r["y_prob"] >= threshold)} for r in rows]
    groups = {("global", "global"): scored}
    for attribute in ("age_group", "race", "sex_gender"):
        for r in scored:
            groups.setdefault((attribute, r[attribute]), []).append(r)
    priors = {}
    for (attr, name), members in groups.items():
        priors[(task, model, attr, name)] = prior_metrics.metrics(members, 20, 20)
    support = ranking.support_tables(rows, task)
    return priors, support, threshold


def trust_for(row, task, model, fitted):
    lookup, support, _ = fitted
    global_row = lookup[(task, model, "global", "global")]
    risk = ranking.risk_score(global_row, None, .35, .25, .15, .15, .10)
    # Unrepresented covariates use global risk; never estimate support on test data.
    combo = (row["age_group"], row["race"], row["sex_gender"])
    if all((task, model, attr, val) in lookup for attr, val in zip(("age_group", "race", "sex_gender"), combo)):
        risk = ranking.score_model_for_combo(lookup, support, task, model, *combo,
                                            SimpleNamespace(**COEFFICIENTS))["score"]
    return probability(1 - risk)


def fused_row(case, records, models, fitted, task, weighted):
    weights = [trust_for(records[m][case], task, m, fitted[m]) if weighted else 1. for m in models]
    require(sum(weights) > 0, "Zero total fusion weight")
    first = records[models[0]][case]
    return {**first, "y_prob": sum(w * records[m][case]["y_prob"] for w, m in zip(weights, models)) / sum(weights)}


def prepare(args):
    locked = load_locked(args.locked_csv)
    sources = {str(args.locked_csv): sha(args.locked_csv)}
    prepared, validation, offline, policies = [], [], [], {}
    for task in TASKS:
        data = {s: {} for s in ("val", "test")}
        for model in args.models:
            for split in data:
                path = source_path(args.predictions_root, task, model, split)
                sources[str(path)] = sha(path)
                data[split][model] = load_probabilities(path, task, model, split)
        reference = data["val"][args.models[0]]
        folder = Path(next(iter(locked[task].values()))["filename"]).parent.parent / "Validation"
        validation.extend({**r, "filename": str(folder / (c + ".npz"))}
                          for c, r in sorted(data["val"]["retfound_oct"].items()))
        require(len(reference) == 1000, f"Need full 1000-case validation cohort for {task}")
        fitted = {}
        for model in args.models:
            require(set(data["val"][model]) == set(reference), f"Validation ID mismatch: {task}/{model}")
            require(not set(data["test"][model]).intersection(reference), f"Validation/test overlap: {task}/{model}")
            require(set(locked[task]) <= set(data["test"][model]), f"Missing locked test cases: {task}/{model}")
            for case, r in data["val"][model].items():
                require(all(r[k] == reference[case][k] for k in ("y_true", "age_group", "race", "sex_gender")),
                        f"Validation label/metadata mismatch: {task}/{model}/{case}")
            for case, r in locked[task].items():
                check = data["test"][model][case]
                require(check["y_true"] == r["truth"], f"Test label mismatch: {task}/{model}/{case}")
                require(all(check[k] == r[k] for k in ("age_group", "race", "sex_gender")),
                        f"Test demographic mismatch: {task}/{model}/{case}")
            fitted[model] = fit_reliability(list(data["val"][model].values()), task, model)
        policies[task] = {"model_thresholds": {m: fitted[m][2] for m in args.models}}
        for variant, weighted in ((VARIANTS[0], False), (VARIANTS[1], True)):
            val = [fused_row(c, data["val"], args.models, fitted, task, weighted) for c in sorted(reference)]
            threshold = select_threshold(val)
            policies[task][variant] = threshold
            for case, meta in sorted(locked[task].items()):
                p = fused_row(case, data["test"], args.models, fitted, task, weighted)["y_prob"]
                offline.append(dict(task=task, case_id=case, variant=variant, prediction=int(p >= threshold),
                                    probability=p, threshold=threshold))
        for case, meta in sorted(locked[task].items()):
            prepared.append(meta)
    config = dict(version=VERSION, models=args.models, coefficients=COEFFICIENTS,
        deployment=args.deployment, api_version=args.api_version, source_hashes=sources,
        data_root=str(args.data_root), oct_weights=str(args.oct_weights), policies=policies,
        f1="macro, labels=[0,1]", worst_group="min macro-F1 over race, sex and age; no support cutoff; not ethnicity",
        reliability_support="validation only; age boundaries 50/70; F1 component is positive-class F1",
        comparisons="Fusion rows use saved FM probes; agent pair uses the OphthalmicAgent RETFound checkpoint plus image reports",
        prepared_sha256=digest(prepared), offline_sha256=digest(offline),
        validation_sha256=digest(validation),
        agent_priors="Recomputed on validation images using the actual OphthalmicAgent checkpoint at its native 0.5 threshold",
        source_code={str(p.relative_to(ROOT)): sha(p) for p in code_paths()})
    config["fingerprint"] = digest(config)
    out = args.run_root
    if (out / "config.json").exists():
        require(json.loads((out / "config.json").read_text()) == config, "Run configuration changed; use a new run root")
    write_json(out / "prepared_cases.json", prepared)
    write_json(out / "offline_predictions.json", offline)
    write_json(out / "validation_cases.json", validation)
    write_json(out / "config.json", config)
    collect(out)
    print("Prepared 750 cases. Offline rows complete. No API calls. Agent rows require a fresh paired run.")


def code_paths():
    paths = [Path(__file__), ROOT / "OphthalmicAgent/Ablation/fairvision_live.py",
             Path(audit.__file__), Path(prior_metrics.__file__), Path(ranking.__file__),
             Path(ranking.risk_score.__code__.co_filename), ROOT / "OphthalmicAgent/data/loader.py"]
    for t in TASKS:
        paths.extend(ROOT / f"OphthalmicAgent/{part}_{t}.py" for part in (
            "BioProfilerAgent/bio_profiler", "VisionAgent/vision_oct", "VisionAgent/vision_slo", "Orchestrator/fairvision"))
    paths.append(ROOT / "OphthalmicAgent/VisionAgent/linear_probing_oct3.py")
    paths.extend(ROOT / f"OphthalmicAgent/{p}" for p in (
        "VisionAgent/models_vit.py", "CounterfactualAgent/counterfactual_agent.py"))
    return paths


def load_prepared(root):
    config = json.loads((root / "config.json").read_text())
    require(digest({k: v for k, v in config.items() if k != "fingerprint"}) == config["fingerprint"], "Bad configuration fingerprint")
    cases = json.loads((root / "prepared_cases.json").read_text())
    predictions = json.loads((root / "offline_predictions.json").read_text())
    require(digest(cases) == config["prepared_sha256"] and digest(predictions) == config["offline_sha256"], "Prepared input changed")
    return config, cases, predictions


def configure(args):
    """Relocate a prepared, unused input bundle from the Mac to CECSL."""
    import fcntl

    with (args.run_root / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Cannot configure while this run is active") from exc
        configure_unused(args)


def configure_unused(args):
    root = args.run_root
    require(not any((root / name).exists() for name in (
        "live_receipt.json", "agent", "api", "shared", "anchor_validation", "anchor_trust.json")),
            "Cannot reconfigure a run after inference has started")
    config, cases, _ = load_prepared(root)
    original = dict(config)
    changed = {path: sha(ROOT / path) for path, value in config["source_code"].items() if sha(ROOT / path) != value}
    if changed:
        require(getattr(args, "upgrade_path_layout", False),
                "Input bundle and code differ. For the original unused bundle's path fix, use --upgrade-path-layout")
        require(set(config["source_code"]) == {str(p.relative_to(ROOT)) for p in code_paths()}
                and all(path in LEGACY_PATH_CODE and config["source_code"][path] == LEGACY_PATH_CODE[path] for path in changed),
                "Not a recognized path-only upgrade; other code or prompts changed. Use matching code or a fresh preparation")
        config["source_code"] = {**config["source_code"], **changed}
        config["path_layout_version"] = "fairvision_paths_v1"
    validation = json.loads((root / "validation_cases.json").read_text())
    require(digest(validation) == config["validation_sha256"], "Validation input changed")
    resolved = require_images(args.data_root, cases + validation)
    config.update(data_root=str(args.data_root), oct_weights=str(args.oct_weights))
    config["fingerprint"] = digest({k: v for k, v in config.items() if k != "fingerprint"})
    if changed:
        archive = root / "config_before_path_layout_update.json"
        require(not archive.exists(), f"Refusing to overwrite previous configuration archive: {archive}")
        write_json(archive, original)
    write_json(root / "config.json", config)
    print(f"Configured data_root={args.data_root}; images={len(resolved)}/{len(cases)+len(validation)}; "
          f"oct_weights={args.oct_weights}. No API calls.")


def score_rows(cases, predictions):
    require(len(predictions) == len(cases), "Partial cohort must not be scored")
    mapping = {}
    for r in predictions:
        require(r["case_id"] not in mapping, "Duplicate case predictions")
        mapping[r["case_id"]] = binary(r["prediction"])
    require(set(mapping) == {r["case_id"] for r in cases}, "Prediction IDs differ")
    rows = [{**c, "prediction": mapping[c["case_id"]]} for c in cases]
    groups = []
    for attribute in ("age_group", "race", "sex_gender"):
        for value in sorted({r[attribute] for r in rows}):
            if value not in ("", "unknown", "missing", "nan", "-1"):
                groups.append(dict(attribute=attribute, subgroup=value,
                                   **audit.metrics([r for r in rows if r[attribute] == value])))
    return {**audit.metrics(rows), "worst_group_macro_f1": min(r["f1_macro"] for r in groups)}, groups


def collect(root):
    config, cases, offline = load_prepared(root)
    summary, groups, table = [], [], []
    for variant in VARIANTS:
        cells, worst = [], []
        for task in TASKS:
            cohort = [c for c in cases if c["task"] == task]
            if variant in VARIANTS[:2]:
                predictions = [r for r in offline if r["task"] == task and r["variant"] == variant]
            else:
                predictions = []
                for c in cohort:
                    path = root / "agent" / task / variant / f"{c['case_id']}.json"
                    if path.exists():
                        r = json.loads(path.read_text())
                        require(r["fingerprint"] == config["fingerprint"] and r["task"] == task
                                and r["variant"] == variant and r["case_id"] == c["case_id"], "Mismatched agent result")
                        binary(r["prediction"])
                        other_arm = next(a for a in VARIANTS[2:] if a != variant)
                        other_path = root / "agent" / task / other_arm / f"{c['case_id']}.json"
                        if other_path.exists():
                            other = json.loads(other_path.read_text())
                            require(other["shared_evidence_sha256"] == r["shared_evidence_sha256"],
                                    "Paired arms received different image reports")
                        predictions.append(r)
            paired_complete = variant in VARIANTS[:2] or all(
                (root / "agent" / task / arm / f"{c['case_id']}.json").exists()
                for c in cohort for arm in VARIANTS[2:])
            if len(predictions) != 250 or not paired_complete:
                cells.append(f"PENDING ({len(predictions)}/250)")
                summary.append(dict(variant=variant, task=task, status="incomplete", completed=len(predictions)))
                continue
            metrics, subs = score_rows(cohort, predictions)
            summary.append(dict(variant=variant, task=task, status="complete", **metrics))
            groups.extend(dict(variant=variant, task=task, **r) for r in subs)
            cells.append(f"{metrics['f1_macro']:.4f}")
            worst.append(metrics["worst_group_macro_f1"])
        mean = f"{sum(worst)/3:.4f}" if len(worst) == 3 else "PENDING"
        labels = dict(zip(VARIANTS, ("Simple multimodal ensemble", "Reliability-weighted fusion (no agents)",
                                    "Agents without reliability priors", "RetinAgent (paired full control)")))
        table.append((labels[variant], *cells, mean))
    audit.write_csv(root / "results.csv", summary)
    audit.write_csv(root / "subgroups.csv", groups)
    text = ["# FairVision Comparisons and Reliability Ablation", "",
            "Macro-F1 on the same 250 cases/task. Mean worst-group = arithmetic mean of the three task-specific minima.",
            "Worst groups: race, sex and age (<50, 50-69, >=70); macro-F1 with both labels, no support cutoff.",
            "Fusion rows are multimodel comparison baselines, not component removals from the RETFound-anchored agent.",
            "Agent rows must be generated together; no historical paper score is inserted.", "",
            "| Variant | Glaucoma | AMD | DR | Mean worst-group macro-F1 |", "|---|---:|---:|---:|---:|"]
    text += ["| " + " | ".join(r) + " |" for r in table]
    (root / "table4.md").write_text("\n".join(text) + "\n")
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Variant & Glaucoma & AMD & DR & Mean worst-group macro-F1 \\", r"\midrule"]
    tex += [" & ".join(r) + r" \\" for r in table]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (root / "table4.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "configure", "paths", "smoke", "run", "collect"), default="prepare")
    p.add_argument("--run-root", type=Path, default=ROOT / "OphthalmicAgent/outputs/fairvision_ablation_v1")
    p.add_argument("--locked-csv", type=Path, default=ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_case_recovery/manifest_recovered.csv")
    p.add_argument("--predictions-root", type=Path, default=ROOT / "equi-agent/outputs/predictions")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--data-root", type=Path, default=ROOT / "OphthalmicAgent",
                   help="OphthalmicAgent root or FairVision dataset root; supports task-first and split-first layouts")
    p.add_argument("--upgrade-path-layout", action="store_true", help="Adopt the path-only fix for the original unused input bundle")
    p.add_argument("--oct-weights", type=Path, default=ROOT / "OphthalmicAgent/weights/oct_model_8_slices_not_center.pth")
    p.add_argument("--deployment", default="gpt-5.1")
    p.add_argument("--api-version", default="2024-12-01-preview")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    for name in ("run_root", "locked_csv", "predictions_root", "data_root", "oct_weights"):
        setattr(args, name, getattr(args, name).resolve())
    require(len(args.models) == len(set(args.models)) and "retfound_oct" in args.models
            and any(m.endswith("_slo") for m in args.models), "Need unique models, RETFound OCT and at least one SLO model")
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "configure":
        configure(args)
    elif args.stage == "paths":
        paths(args)
    elif args.stage == "collect":
        collect(args.run_root)
    else:
        config, cases, _ = load_prepared(args.run_root)
        require(all(sha(ROOT / path) == value for path, value in config["source_code"].items()),
                "Source code changed since prepare; use a fresh run root")
        from Ablation.fairvision_live import run
        run(args, config, cases)
        collect(args.run_root)


if __name__ == "__main__":
    main()
