"""Development-only reliability estimates for the recovered single-target GDP helper.

Uses the CECSL Harvard-GDP model factory and dataset, not the six-output trainer.
No API calls, prompt edits, threshold search, or test-based checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = "td_pointwise_no_p_cut"
TARGETS = ("md", "vfi", "td_pointwise", "md_fast", "md_fast_no_p_cut", TARGET)
# This exact predecessor has the same primary-target training/evaluation protocol.
# Only allow its receipts after every other config, source and data field matches.
COMPATIBLE_PRIMARY_RUNNERS = {"24861aacab809caa786032ea08b851192832f516fac98cfeae7df0674041abc8"}
MODEL = "gdp_native_rnflt_tds_efficientnet"
PREFIX = f"gdp_progression_forecasting_{TARGET}"
THRESHOLD = 0.5
# Fingerprints of the source bundle exported from CECSL on 2026-09-28.
NATIVE_HASHES = {
    "utils/modules.py": "a0afe4e640911e7180e3393b714f809dc9334441681e67cb20be0ed1e325c609",
    "utils/image_datasets.py": "b76660ce08c7a61dd3af06beb7f9d78308cd62cfec662c8615f80a549c6e2e92",
}
FIELDS = [
    "patient_id", "eye_id", "visit_id", "image_id", "dataset", "task", "model_name",
    "y_true", "y_prob", "y_pred", "applied_threshold", "split", "race", "ethnicity",
    "sex_gender", "age", "age_group", "metadata_missing_flag", "progression_target",
    "fold", "run_fingerprint",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def case_id(row):
    value = row.get("image_id", "").strip()
    if not value or Path(value).name != value:
        raise ValueError(f"Expected a basename image_id, got {value!r}")
    return value.removesuffix(".npz")


def label(row):
    value = float(row["y_true"])
    if value not in (0, 1):
        raise ValueError(f"Nonbinary label for {case_id(row)}: {value}")
    return int(value)


def validate_cohort(rows, expected_dev=300, expected_test=200, target=TARGET):
    ids = [case_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs in manifest")
    for row in rows:
        if row.get("progression_target") != target:
            raise ValueError(f"Wrong progression target for {case_id(row)}")
        if not row.get("patient_id", "").strip():
            raise ValueError(f"Missing patient_id for {case_id(row)}; cannot enforce group separation")
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split for {case_id(row)}")
        label(row)
    dev = sorted((r for r in rows if r["split"] != "test"), key=case_id)
    test = sorted((r for r in rows if r["split"] == "test"), key=case_id)
    if (len(dev), len(test)) != (expected_dev, expected_test):
        raise ValueError(f"Expected {expected_dev}/{expected_test} development/test cases; got {len(dev)}/{len(test)}")
    overlap = {r["patient_id"] for r in dev} & {r["patient_id"] for r in test}
    if overlap:
        raise ValueError(f"Development/test patient overlap: {sorted(overlap)[:5]}")
    return dev, test


def make_folds(dev, n_folds, seed):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    groups = [r["patient_id"] for r in dev]
    labels = [label(r) for r in dev]
    if n_folds < 2 or len(set(labels)) != 2 or min(Counter(labels).values()) < n_folds:
        raise ValueError("Each development class must have at least one case per fold")
    if len(set(groups)) == len(groups):
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(dev)), labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(dev)), labels, groups)
    assignments = {}
    for fold, (training, held_out) in enumerate(splits, start=1):
        if {groups[i] for i in training} & {groups[i] for i in held_out}:
            raise ValueError("Patient appears in both sides of a fold")
        for indices in (training, held_out):
            if len({labels[i] for i in indices}) != 2:
                raise ValueError("A fold lacks one class; choose a prespecified smaller fold count")
        assignments.update({case_id(dev[i]): fold for i in held_out})
    if set(assignments) != {case_id(r) for r in dev}:
        raise ValueError("Incomplete OOF partition")
    return assignments


def validate_recipe(settings):
    required = {
        "model": "efficientnet", "data_modality": 2, "image_size": 224,
        "loss_type": "bce", "data_type": "label+unlabel",
        "progression_outcome": f"progression_outcome_{TARGET}",
        "num_epochs": 60, "batch_size": 6, "lr_vf": 2e-5, "weight_decay_vf": 0.0,
        "random_seed": 3280, "warmup_steps": 0, "use_fp16": False,
        "resume_checkpoint_vf": "", "resume_checkpoint": "",
    }
    mismatches = {key: settings.get(key) for key, value in required.items() if settings.get(key) != value}
    if mismatches:
        raise ValueError(f"Not the recovered single-target recipe: {mismatches}")
    return required


def source_hashes(native_root):
    hashes = {name: sha256(native_root / name) for name in NATIVE_HASHES}
    if hashes != NATIVE_HASHES:
        raise ValueError("Harvard-GDP model/dataset source differs from the audited CECSL bundle; review before running")
    for name in ("utils/dist_util.py", "utils/data_handler.py", "scripts/train_progression_pseudo_supervisor.py"):
        hashes[name] = sha256(native_root / name)
    return hashes


def staged_paths(data_root, settings, dev, test):
    paths = {}
    for subset, expected in (
        ([settings["data_subset"], settings["data_val_subset"]], dev),
        ([settings["data_tst_subset"]], test),
    ):
        found = {}
        for name in subset:
            directory = data_root / name
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.glob("*.npz")):
                if not path.is_file():
                    raise FileNotFoundError(f"Broken NPZ link: {path}")
                key = path.stem
                if key in found or key in paths:
                    raise ValueError(f"NPZ appears in multiple source splits: {key}")
                found[key] = path.resolve()
        expected_ids = {case_id(row) for row in expected}
        if set(found) != expected_ids:
            raise ValueError(f"Staged NPZs differ from manifest: missing={sorted(expected_ids - set(found))[:5]}, extra={sorted(set(found) - expected_ids)[:5]}")
        paths.update(found)
    if len(set(paths.values())) != len(paths):
        raise ValueError("Multiple image IDs resolve to the same NPZ")
    return paths


def audit_development_npzs(dev, paths, target=TARGET):
    import numpy as np

    hashes = {}
    for row in dev:
        key = case_id(row)
        path = paths[key]
        with np.load(path, allow_pickle=False) as data:
            rnflt = np.asarray(data["rnflt"])
            tds = np.asarray(data["tds"])
            progression = np.asarray(data["progression"]).reshape(-1)
            if rnflt.size != 225 * 225 or tds.shape != (52,) or len(progression) != 6:
                raise ValueError(f"Unexpected native input shapes: {key}")
            if not np.isfinite(rnflt).all() or not np.isfinite(tds).all():
                raise ValueError(f"Nonfinite native input: {key}")
            if float(progression[TARGETS.index(target)]) != label(row):
                raise ValueError(f"Manifest/NPZ target mismatch: {key}")
        hashes[key] = sha256(path)
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("Identical development NPZ files have different case IDs; resolve before splitting")
    return hashes


def load_native(native_root):
    sys.path.insert(0, str(native_root))
    datasets = importlib.import_module("utils.image_datasets")
    modules = importlib.import_module("utils.modules")
    for module, name in ((datasets, "utils/image_datasets.py"), (modules, "utils/modules.py")):
        if Path(module.__file__).resolve() != (native_root / name).resolve():
            raise RuntimeError(f"Imported the wrong native module: {module.__file__}")
    return modules.create_model, datasets.Longitudinal_Dataset


def native_dataset(dataset_class, rows, paths, directory, target=TARGET):
    directory.mkdir(parents=True, exist_ok=True)
    expected = {case_id(r) + ".npz" for r in rows}
    if {p.name for p in directory.iterdir()} - expected:
        raise ValueError(f"Unexpected files in staged fold: {directory}")
    for row in rows:
        dest = directory / (case_id(row) + ".npz")
        source = paths[case_id(row)]
        if dest.is_symlink():
            if dest.resolve() != source:
                raise ValueError(f"Wrong staged source: {dest}")
        elif dest.exists():
            raise ValueError(f"Refusing to replace existing file: {dest}")
        else:
            dest.symlink_to(source)
    dataset = dataset_class(str(directory.parent), subset=directory.name,
                            outcome_type=f"progression_outcome_{TARGET}", modality=2,
                            resolution=224, data_type="label+unlabel")
    if set(dataset.rnflt_data) != expected or dataset.unlabel_flags is not None:
        raise ValueError("Native loader changed cohort or enabled unlabeled training")
    # The native constructor recognizes only two target names, but __getitem__
    # indexes the six-label progression vector. Extend label selection only.
    dataset.progression_index = TARGETS.index(target)
    dataset.progression_type = f"progression_outcome_{target}"
    dataset.rnflt_data = [case_id(r) + ".npz" for r in rows]
    dataset.dataset_len = len(rows)
    return dataset


def fit_model(factory, dataset, recipe, seed, device, workers, run_name):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = torch.nn.Sequential(factory(model_type="efficientnet", in_dim=2, out_dim=1), torch.nn.Sigmoid()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr_vf"],
                                 betas=(0.0, 0.1), weight_decay=recipe["weight_decay_vf"])
    loader = DataLoader(dataset, batch_size=recipe["batch_size"], shuffle=True,
                        num_workers=workers, drop_last=True)
    if not len(loader):
        raise ValueError("No complete training batches")
    loss_function = torch.nn.BCELoss()
    # In the recovered code, unlabeled flags are disabled and the scheduler only
    # steps in the inactive unlabeled branch. Match its constant-LR supervised path.
    for epoch in range(recipe["num_epochs"]):
        model.train()
        total_loss = 0.0
        for images, labels, _ in loader:
            if not torch.isfinite(images).all() or not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Invalid input or unlabeled case in supervised training")
            optimizer.zero_grad()
            probability = model(images.to(device)).squeeze(1)
            loss = loss_function(probability, labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss: {run_name}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{run_name} epoch={epoch + 1}/{recipe['num_epochs']} train_loss={total_loss / len(loader):.6f}", flush=True)
    return model


def predict(model, dataset, rows, split, fold, run_fingerprint, device, workers, target=TARGET):
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    output = []
    offset = 0
    with torch.inference_mode():
        for images, labels, _ in DataLoader(dataset, batch_size=18, shuffle=False, num_workers=workers):
            if not torch.isfinite(images).all():
                raise ValueError("Nonfinite evaluation input")
            probabilities = model(images.to(device)).squeeze(1).cpu().tolist()
            for truth, probability in zip(labels.tolist(), probabilities):
                row = rows[offset]
                if truth != label(row) or not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(f"Invalid native evaluation output: {case_id(row)}")
                output.append({**{k: row.get(k, "") for k in FIELDS}, "image_id": case_id(row) + ".npz",
                               "dataset": "harvard_gdp", "task": "progression_forecasting", "model_name": MODEL,
                               "y_true": int(truth), "y_prob": probability, "y_pred": int(probability >= THRESHOLD),
                               "applied_threshold": THRESHOLD, "split": split, "fold": fold,
                               "run_fingerprint": run_fingerprint, "progression_target": target})
                offset += 1
    if offset != len(rows):
        raise ValueError("Incomplete evaluation output")
    return output


def validate_predictions(predictions, expected, split, fold, run_fingerprint, target=TARGET):
    by_id = {case_id(r): r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {case_id(r) for r in expected}:
        raise ValueError("Incomplete or duplicate prediction cohort")
    for row in expected:
        result = by_id[case_id(row)]
        probability = float(result["y_prob"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid cached probability")
        if (label(result) != label(row) or result["patient_id"] != row["patient_id"]
                or result["split"] != split or str(result["fold"]) != str(fold)
                or result["run_fingerprint"] != run_fingerprint or result["model_name"] != MODEL
                or result["progression_target"] != target or float(result["applied_threshold"]) != THRESHOLD
                or float(result["y_pred"]) != int(probability >= THRESHOLD)):
            raise ValueError(f"Prediction provenance mismatch: {case_id(row)}")


def metrics(rows, split):
    sys.path.insert(0, str(ROOT / "equi-agent"))
    from src.metrics.classification import binary_classification_metrics

    values = binary_classification_metrics([label(r) for r in rows], [float(r["y_prob"]) for r in rows],
                                           threshold=THRESHOLD)
    # Exactly the existing agent's global-trust formula, not a fitted coefficient.
    trust = max(0.05, (0.70 * values["balanced_accuracy"] + 0.30 * values["f1"]) * (1 - min(values["ece"], 0.8)))
    values = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in values.items()}
    return {"dataset": "harvard_gdp", "task": "progression_forecasting", "model_name": MODEL,
            "split": split, **values, "global_trust_weight": trust, "threshold": THRESHOLD,
            "f1_average": "binary", "prior_source": "development_oof" if split == "oof" else "final_test_only"}


def run_partition(args, recipe, factory, dataset_class, paths, train, heldout, name, split, seed, run_fingerprint, target=TARGET):
    directory = args.out_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "predictions.csv"
    receipt_path = directory / "complete.json"
    fold = name.removeprefix("fold_") if split == "oof" else "final"
    receipt = {"run_fingerprint": run_fingerprint, "seed": seed,
               "train_ids": [case_id(r) for r in train], "heldout_ids": [case_id(r) for r in heldout],
               "epochs": recipe["num_epochs"], "checkpoint_rule": "last_epoch_fixed_in_advance"}
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if any(saved.get(k) != v for k, v in receipt.items()) or saved.get("predictions_sha256") != sha256(csv_path):
            raise ValueError(f"Invalid completed-run receipt: {directory}")
        if split == "test" and saved.get("checkpoint_sha256") != sha256(directory / "model.pt"):
            raise ValueError("Final checkpoint changed")
        if split == "test" and saved.get("test_npz_sha256") != {case_id(r): sha256(paths[case_id(r)]) for r in heldout}:
            raise ValueError("Final test inputs changed")
        rows = read_csv(csv_path)
        validate_predictions(rows, heldout, split, fold, run_fingerprint, target)
        print(f"reuse {name}: {len(rows)} cases", flush=True)
        return rows
    if {r["patient_id"] for r in train} & {r["patient_id"] for r in heldout}:
        raise ValueError("Training/evaluation patient overlap")
    training_dataset = native_dataset(dataset_class, train, paths, directory / "inputs" / "train", target)
    model = fit_model(factory, training_dataset, recipe, seed, args.device, args.num_workers, name)
    if split == "test":
        import torch
        checkpoint = directory / "model.pt"
        temporary = directory / "model.pt.tmp"
        torch.save({"state_dict": model.state_dict(), **receipt}, temporary)
        temporary.replace(checkpoint)
        receipt["checkpoint_sha256"] = sha256(checkpoint)
        receipt["test_npz_sha256"] = {case_id(r): sha256(paths[case_id(r)]) for r in heldout}
    # Held-out NPZs are loaded only after the training loop has finished.
    evaluation_dataset = native_dataset(dataset_class, heldout, paths, directory / "inputs" / split, target)
    rows = predict(model, evaluation_dataset, heldout, split, fold, run_fingerprint, args.device, args.num_workers, target)
    validate_predictions(rows, heldout, split, fold, run_fingerprint, target)
    write_csv(csv_path, rows, FIELDS)
    write_json(receipt_path, {**receipt, "predictions_sha256": sha256(csv_path)})
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, default=TARGET)
    parser.add_argument("--native-root", type=Path, default=Path.home() / "Harvard-GDP")
    parser.add_argument("--original-args", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, help="Override saved staged-data location; preserve original train/val/test membership")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "equi-agent/outputs/gdp_native_oof_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true", help="Check source, cohorts, development NPZs and imports; no model creation/training")
    parser.add_argument("--fit-final", action="store_true", help="After OOF, fit all 300 development cases and evaluate the locked 200 test cases")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = ROOT / f"equi-agent/outputs/manifests/gdp_progression_forecasting_{args.target}.csv"
    return args


def reconcile_config(config, previous, target):
    if previous == config:
        return config
    compatible = dict(config)
    if target == TARGET and previous.get("runner_sha256") in COMPATIBLE_PRIMARY_RUNNERS:
        compatible["runner_sha256"] = previous["runner_sha256"]
        if previous == compatible:
            return compatible
    raise ValueError("Run configuration/source/data changed. Use a NEW --out-dir; existing results were not overwritten")


def execute(args):
    import numpy as np
    import sklearn
    import torch

    sys.path.insert(0, str(ROOT / "equi-agent"))
    importlib.import_module("src.metrics.classification")
    args.native_root = args.native_root.expanduser().resolve()
    target = getattr(args, "target", TARGET)
    prefix = f"gdp_progression_forecasting_{target}"
    settings = json.loads(args.original_args.read_text())
    recipe = validate_recipe(settings)
    native_hashes = source_hashes(args.native_root)
    dev, test = validate_cohort(read_csv(args.manifest), target=target)
    data_root = (args.data_root or Path(settings["data_dir"])).expanduser().resolve()
    paths = staged_paths(data_root, settings, dev, test)
    data_hashes = audit_development_npzs(dev, paths, target)
    folds = make_folds(dev, args.folds, recipe["random_seed"])
    factory, dataset_class = load_native(args.native_root)
    config = {
        "protocol": "single_target_fixed_epoch_development_oof_v1", "recipe": recipe,
        "native_source_sha256": native_hashes, "runner_sha256": sha256(__file__),
        "manifest_sha256": sha256(args.manifest), "original_args_sha256": sha256(args.original_args),
        "development_npz_sha256": data_hashes, "native_root": str(args.native_root),
        "data_root": str(data_root), "fold_assignments": folds,
        "test_ids_excluded_from_oof": [case_id(r) for r in test], "folds": args.folds,
        "threshold": THRESHOLD, "checkpoint_rule": "last_epoch_fixed_in_advance",
        "fold_seed_rule": "3280 + fold_number", "final_seed": 3280 + 1000,
        "training_path": "supervised; native unlabel_flags=None; constant LR; non-EMA weights",
        "num_workers": args.num_workers, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "torch": torch.__version__, "torchvision": importlib.import_module("torchvision").__version__,
        "pandas": importlib.import_module("pandas").__version__, "python": sys.version,
        "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "grouping": "manifest patient_id; synthetic IDs cannot establish identity beyond dataset identifiers",
        "old_epoch_selection_reproduced": False, "bitwise_reproduction_of_old_run": False,
    }
    if target != TARGET:
        config["progression_target"] = target
        config["label_index"] = TARGETS.index(target)
    config_path = args.out_dir / "resolved_config.json"
    if config_path.exists():
        config = reconcile_config(config, json.loads(config_path.read_text()), target)
    run_fingerprint = fingerprint(config)
    write_json(config_path, config)
    write_json(args.out_dir / "execution_version.json", {
        "current_runner_sha256": sha256(__file__), "protocol_runner_sha256": config["runner_sha256"],
        "compatible_primary_resume": config["runner_sha256"] != sha256(__file__),
        "progression_target": target,
    })
    write_json(args.out_dir / "original_args.json", settings)
    write_csv(args.out_dir / "fold_assignments.csv", [
        {"image_id": case_id(r), "patient_id": r["patient_id"], "y_true": label(r), "fold": folds[case_id(r)]}
        for r in dev
    ])
    print(f"development={len(dev)} test_excluded={len(test)} folds={args.folds} threshold={THRESHOLD} epochs={recipe['num_epochs']}", flush=True)
    if args.prepare_only:
        print("PREPARED ONLY: no trained models, predictions or reliability estimates were produced", flush=True)
        return
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Training requires CECSL CUDA; use --prepare-only for a non-training check")
    torch.cuda.get_device_properties(torch.device(args.device))
    oof = []
    for fold in range(1, args.folds + 1):
        training = [r for r in dev if folds[case_id(r)] != fold]
        heldout = [r for r in dev if folds[case_id(r)] == fold]
        oof.extend(run_partition(args, recipe, factory, dataset_class, paths, training, heldout,
                                 f"fold_{fold}", "oof", recipe["random_seed"] + fold, run_fingerprint, target))
    oof.sort(key=case_id)
    if len(oof) != len(dev) or {case_id(r) for r in oof} != {case_id(r) for r in dev}:
        raise ValueError("Incomplete OOF cohort; no priors exported")
    oof_metrics = metrics(oof, "oof")
    write_csv(args.out_dir / "predictions_oof.csv", oof, FIELDS)
    write_csv(args.out_dir / "oof_aggregate.csv", [oof_metrics])
    write_csv(args.out_dir / "fold_metrics.csv", [
        {"fold": fold, **metrics([r for r in oof if str(r["fold"]) == str(fold)], "oof")}
        for fold in range(1, args.folds + 1)
    ])
    prior_path = args.out_dir / "priors" / f"exp8_{prefix}_{MODEL}" / f"{prefix}_{MODEL}_aggregate.csv"
    write_csv(prior_path, [{**oof_metrics, "progression_target": target, "run_fingerprint": run_fingerprint}])
    write_json(args.out_dir / "oof_summary.json", {
        "complete_development_oof": True, "cases": len(oof), "folds": args.folds,
        "progression_target": target,
        "test_used_for_fitting_selection_or_priors": False, "run_fingerprint": run_fingerprint,
        "metrics": oof_metrics, "prior_aggregate": str(prior_path),
        "warning": "A new fixed-epoch protocol, not retrospective validation of the old test-informed run",
    })
    print(json.dumps({"oof_metrics": oof_metrics, "prior_aggregate": str(prior_path)}, indent=2), flush=True)
    if args.fit_final:
        predictions = run_partition(args, recipe, factory, dataset_class, paths, dev, test, "final",
                                    "test", recipe["random_seed"] + 1000, run_fingerprint, target)
        test_metrics = metrics(predictions, "test")
        write_csv(args.out_dir / "predictions" / f"{prefix}_{MODEL}.csv", predictions, FIELDS)
        write_csv(args.out_dir / "test_metrics" / "aggregate.csv", [test_metrics])
        write_json(args.out_dir / "final_summary.json", {"complete_test_cohort": True, "cases": len(predictions),
                   "metrics": test_metrics, "run_fingerprint": run_fingerprint,
                   "checkpoint_rule": "last_epoch_fixed_in_advance", "prior_source": str(prior_path)})
        print(json.dumps({"final_test_metrics": test_metrics}, indent=2), flush=True)


def main():
    args = parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Already running in {args.out_dir}")
        execute(args)


if __name__ == "__main__":
    main()
