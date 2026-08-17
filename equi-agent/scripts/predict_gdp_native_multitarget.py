from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any


TARGETS = [
    "md",
    "vfi",
    "td_pointwise",
    "md_fast",
    "md_fast_no_p_cut",
    "td_pointwise_no_p_cut",
]

MODEL_NAME = "gdp_native_rnflt_tds_multitask_efficientnet"

OUTPUT_FIELDS = [
    "patient_id",
    "eye_id",
    "visit_id",
    "image_id",
    "dataset",
    "task",
    "model_name",
    "y_true",
    "y_prob",
    "y_pred",
    "applied_threshold",
    "split",
    "race",
    "ethnicity",
    "sex_gender",
    "age",
    "age_group",
    "metadata_missing_flag",
    "progression_target",
]

TD_COLUMNS = [
    *[f"td{index}" for index in range(1, 25)],
    *[f"td{index}" for index in range(26, 34)],
    *[f"td{index}" for index in range(35, 55)],
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one six-output Harvard-GDP native RNFLT+TDS EfficientNet and "
            "write one standard prediction CSV per progression endpoint."
        )
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "manifests",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "predictions",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            repo_root()
            / "equi-agent"
            / "outputs"
            / "checkpoints"
            / "gdp_native_rnflt_tds_multitask_efficientnet.pt"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            repo_root()
            / "equi-agent"
            / "outputs"
            / "checkpoints"
            / "gdp_native_rnflt_tds_multitask_efficientnet_summary.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-train-cases", type=int, default=300)
    parser.add_argument("--expected-test-cases", type=int, default=200)
    parser.add_argument("--threshold-metric", choices=["f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate six-manifest identity and input files without training.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def case_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in ["patient_id", "eye_id", "image_id"])


def optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def resolve_path(raw: str, prefix_from: str, prefix_to: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    if prefix_from and prefix_to and raw.startswith(prefix_from):
        candidate = Path(prefix_to) / raw[len(prefix_from) :].lstrip("/")
        if candidate.is_file():
            return candidate
        path = candidate
    marker = "/RETFound/"
    if marker in raw:
        candidate = repo_root() / raw.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
        path = candidate
    return path


def load_cohort(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_target: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    manifest_rows: dict[str, int] = {}
    for target in TARGETS:
        path = args.manifests_root / f"gdp_progression_forecasting_{target}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing GDP manifest: {path}")
        rows = [
            row
            for row in read_csv(path)
            if str(row.get("split", "")).strip().lower() in {"train", "test"}
        ]
        expected_total = args.expected_train_cases + args.expected_test_cases
        if len(rows) != expected_total:
            raise ValueError(
                f"Expected {expected_total} train/test rows for {target}, found {len(rows)}"
            )
        keyed = {case_key(row): row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"Duplicate case identifiers in {path}")
        by_target[target] = keyed
        manifest_rows[target] = len(rows)

    reference_keys = set(by_target[TARGETS[0]])
    for target in TARGETS[1:]:
        if set(by_target[target]) != reference_keys:
            raise ValueError(f"The {target} manifest does not contain the same 500 cases")

    cases: list[dict[str, Any]] = []
    missing_files = []
    for key in sorted(reference_keys):
        rows = {target: by_target[target][key] for target in TARGETS}
        reference = rows[TARGETS[0]]
        signature = tuple(reference.get(column, "") for column in ["split", "rnflt_path", *TD_COLUMNS])
        for target in TARGETS[1:]:
            other = rows[target]
            other_signature = tuple(other.get(column, "") for column in ["split", "rnflt_path", *TD_COLUMNS])
            if other_signature != signature:
                raise ValueError(f"Input evidence differs across endpoint manifests for case={key}")

        labels = []
        for target in TARGETS:
            label = optional_float(rows[target].get("y_true"))
            if label not in {0.0, 1.0}:
                raise ValueError(f"Invalid {target} label for case={key}: {rows[target].get('y_true')!r}")
            labels.append(int(label))

        td_values = [optional_float(reference.get(column)) for column in TD_COLUMNS]
        if any(value is None for value in td_values):
            raise ValueError(f"Incomplete 52-value TDS vector for case={key}")
        rnflt_path = resolve_path(
            reference.get("rnflt_path", ""),
            args.path_prefix_from,
            args.path_prefix_to,
        )
        if not rnflt_path.is_file():
            missing_files.append(str(rnflt_path))
        cases.append(
            {
                "key": key,
                "row": reference,
                "labels": labels,
                "td_values": td_values,
                "rnflt_path": str(rnflt_path),
            }
        )

    if missing_files:
        raise FileNotFoundError(f"Missing RNFLT files, first examples: {missing_files[:3]}")
    split_counts = {
        split: sum(case["row"].get("split") == split for case in cases)
        for split in ["train", "test"]
    }
    expected_splits = {
        "train": args.expected_train_cases,
        "test": args.expected_test_cases,
    }
    if split_counts != expected_splits:
        raise ValueError(f"Expected GDP split counts {expected_splits}, found {split_counts}")
    positives = {
        target: {
            split: sum(
                case["labels"][index]
                for case in cases
                if case["row"].get("split") == split
            )
            for split in ["train", "test"]
        }
        for index, target in enumerate(TARGETS)
    }
    return cases, {
        "cases": len(cases),
        "manifest_rows": manifest_rows,
        "split_counts": split_counts,
        "positive_counts": positives,
        "targets": TARGETS,
        "input_channels": ["RNFLT thickness map", "52-point TDS map"],
    }


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vf_to_matrix(np, vector):
    matrix = np.zeros((8, 9), dtype="float32")
    matrix[0, 3:7] = vector[0:4]
    matrix[1, 2:8] = vector[4:10]
    matrix[2, 1:] = vector[10:18]
    matrix[3, :7] = vector[18:25]
    matrix[3, 8] = vector[25]
    matrix[4, :7] = vector[26:33]
    matrix[4, 8] = vector[33]
    matrix[5, 1:] = vector[34:42]
    matrix[6, 2:8] = vector[42:48]
    matrix[7, 3:7] = vector[48:52]
    matrix = matrix.repeat(28, axis=0).repeat(25, axis=1)
    return matrix[:, :-1]


def make_dataset(torch, np, cases: list[dict[str, Any]]):
    class GDPDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(cases)

        def __getitem__(self, index):
            case = cases[index]
            with np.load(case["rnflt_path"], allow_pickle=False) as data:
                rnflt = np.asarray(data[case["row"].get("rnflt_key") or "rnflt"], dtype="float32")
            rnflt = rnflt.reshape(225, 225)[1:, 1:]
            rnflt = (np.clip(rnflt, -2.0, 350.0) + 2.0) / 352.0
            tds = np.asarray(case["td_values"], dtype="float32")
            tds = (tds - (-38.0)) / (26.0 - (-38.0)) * 2.0
            tds = vf_to_matrix(np, tds)
            image = np.stack([rnflt, tds], axis=0).astype("float32")
            labels = np.asarray(case["labels"], dtype="float32")
            return torch.from_numpy(image), torch.from_numpy(labels), index

    return GDPDataset()


def make_model(torch, torchvision, pretrained: bool):
    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.efficientnet_v2_s(weights=weights)
    old = model.features[0][0]
    replacement = torch.nn.Conv2d(
        2,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            mean_weight = old.weight.mean(dim=1, keepdim=True)
            replacement.weight.copy_(mean_weight.repeat(1, 2, 1, 1))
    model.features[0][0] = replacement
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(TARGETS))
    return model


def make_loader(torch, dataset, indices, batch_size: int, workers: int, shuffle: bool):
    sampler = torch.utils.data.SubsetRandomSampler(indices) if shuffle else None
    subset = None if shuffle else torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        dataset if shuffle else subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def pos_weights(torch, cases: list[dict[str, Any]], indices: list[int], device: str):
    labels = torch.tensor([cases[index]["labels"] for index in indices], dtype=torch.float32)
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return (negatives / positives.clamp_min(1.0)).clamp(max=50.0).to(device)


def fit_model(
    torch,
    torchvision,
    dataset,
    cases,
    train_indices,
    args,
    seed: int,
):
    set_seed(torch, seed)
    model = make_model(torch, torchvision, pretrained=not args.no_pretrained).to(args.device)
    loader = make_loader(
        torch,
        dataset,
        train_indices,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights(torch, cases, train_indices, args.device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.0, 0.1),
    )
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_rows = 0
        for images, labels, _ in loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
            total_rows += images.shape[0]
        print(
            f"native seed={seed} epoch={epoch + 1}/{args.epochs} "
            f"loss={total_loss / max(total_rows, 1):.6f}",
            flush=True,
        )
    return model


def predict(torch, model, dataset, indices, args):
    loader = make_loader(
        torch,
        dataset,
        indices,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    probabilities = []
    ordered_indices = []
    model.eval()
    with torch.no_grad():
        for images, _, batch_indices in loader:
            logits = model(images.to(args.device, non_blocking=True))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            ordered_indices.extend(int(index) for index in batch_indices)
    return ordered_indices, probabilities


def score_threshold(np, y_true, y_prob, threshold: float, metric: str) -> float:
    prediction = y_prob >= threshold
    if metric == "f1":
        tp = int(((prediction == 1) & (y_true == 1)).sum())
        fp = int(((prediction == 1) & (y_true == 0)).sum())
        fn = int(((prediction == 0) & (y_true == 1)).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1)
    positive = y_true == 1
    negative = y_true == 0
    sensitivity = float(((prediction == 1) & positive).sum()) / max(int(positive.sum()), 1)
    specificity = float(((prediction == 0) & negative).sum()) / max(int(negative.sum()), 1)
    return (sensitivity + specificity) / 2.0


def select_thresholds(np, cases, train_indices, oof_probabilities, metric: str):
    labels = np.asarray([cases[index]["labels"] for index in train_indices], dtype="int64")
    probabilities = np.asarray([oof_probabilities[index] for index in train_indices], dtype="float64")
    thresholds = []
    scores = []
    for target_index in range(len(TARGETS)):
        candidates = np.linspace(0.01, 0.99, 99)
        candidate_scores = [
            score_threshold(
                np,
                labels[:, target_index],
                probabilities[:, target_index],
                float(threshold),
                metric,
            )
            for threshold in candidates
        ]
        best = max(range(len(candidates)), key=lambda index: (candidate_scores[index], -abs(candidates[index] - 0.5)))
        thresholds.append(float(candidates[best]))
        scores.append(float(candidate_scores[best]))
    return thresholds, scores


def prediction_path(root: Path, target: str) -> Path:
    return root / f"gdp_progression_forecasting_{target}_{MODEL_NAME}.csv"


def main() -> None:
    args = parse_args()
    cases, cohort_audit = load_cohort(args)
    if args.audit_only:
        write_json(args.summary, {"audit_only": True, "cohort_audit": cohort_audit})
        print(json.dumps(cohort_audit, indent=2, sort_keys=True))
        return

    import numpy as np
    import torch
    import torchvision
    from sklearn.model_selection import KFold

    set_seed(torch, args.seed)
    dataset = make_dataset(torch, np, cases)
    train_indices = [index for index, case in enumerate(cases) if case["row"].get("split") == "train"]
    test_indices = [index for index, case in enumerate(cases) if case["row"].get("split") == "test"]
    if args.folds < 2:
        raise ValueError("--folds must be at least 2 for out-of-fold threshold selection")

    oof_probabilities: dict[int, list[float]] = {}
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    train_array = np.asarray(train_indices)
    for fold, (fit_positions, valid_positions) in enumerate(splitter.split(train_array), start=1):
        fit_indices = train_array[fit_positions].tolist()
        valid_indices = train_array[valid_positions].tolist()
        print(f"native OOF fold={fold}/{args.folds} train={len(fit_indices)} val={len(valid_indices)}", flush=True)
        model = fit_model(torch, torchvision, dataset, cases, fit_indices, args, args.seed + fold)
        predicted_indices, probabilities = predict(torch, model, dataset, valid_indices, args)
        oof_probabilities.update(dict(zip(predicted_indices, probabilities)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if set(oof_probabilities) != set(train_indices):
        raise RuntimeError("Out-of-fold predictions do not cover all 300 training cases")
    thresholds, oof_scores = select_thresholds(
        np,
        cases,
        train_indices,
        oof_probabilities,
        args.threshold_metric,
    )
    print(f"native thresholds={dict(zip(TARGETS, thresholds))}", flush=True)

    final_model = fit_model(torch, torchvision, dataset, cases, train_indices, args, args.seed + 1000)
    predicted_indices, test_probabilities = predict(torch, final_model, dataset, test_indices, args)
    probability_by_index = dict(zip(predicted_indices, test_probabilities))

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "targets": TARGETS,
            "state_dict": final_model.state_dict(),
            "thresholds": thresholds,
            "oof_threshold_scores": oof_scores,
            "configuration": vars(args),
        },
        args.checkpoint,
    )

    outputs = {}
    for target_index, target in enumerate(TARGETS):
        rows = []
        threshold = thresholds[target_index]
        for index in test_indices:
            case = cases[index]
            source = case["row"]
            probability = float(probability_by_index[index][target_index])
            rows.append(
                {
                    **source,
                    "model_name": MODEL_NAME,
                    "y_true": case["labels"][target_index],
                    "y_prob": f"{probability:.8f}",
                    "y_pred": int(probability >= threshold),
                    "applied_threshold": f"{threshold:.4f}",
                    "progression_target": target,
                }
            )
        output = prediction_path(args.predictions_root, target)
        write_csv(output, rows)
        outputs[target] = str(output)

    summary = {
        "audit_only": False,
        "model_name": MODEL_NAME,
        "cohort_audit": cohort_audit,
        "protocol": {
            "architecture": "torchvision EfficientNet-V2-S with two-channel RNFLT+TDS input and six outputs",
            "threshold_selection": f"{args.folds}-fold training-set OOF {args.threshold_metric}",
            "test_used_for_selection": False,
            "epochs_per_fold_and_refit": args.epochs,
            "seed": args.seed,
        },
        "thresholds": dict(zip(TARGETS, thresholds)),
        "oof_threshold_scores": dict(zip(TARGETS, oof_scores)),
        "checkpoint": str(args.checkpoint),
        "outputs": outputs,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
