from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev


def fnum(value: object) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def int_label(value: object) -> int | None:
    value_num = fnum(value)
    if value_num is None:
        return None
    label = int(value_num)
    return label if label in {0, 1} else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stratified_split(rows: list[dict], test_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for row in rows:
        by_label[int(row["label"])].append(row)

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n_test = max(1, round(len(label_rows) * test_frac))
        test_rows.extend(label_rows[:n_test])
        train_rows.extend(label_rows[n_test:])
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    pairs = Counter(zip(y_true, y_pred))
    return {
        "tn": pairs[(0, 0)],
        "fp": pairs[(0, 1)],
        "fn": pairs[(1, 0)],
        "tp": pairs[(1, 1)],
    }


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    c = confusion(y_true, y_pred)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "n": n,
        **c,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return []
    thresholds = [unique[0] - 1e-9]
    thresholds.extend((a + b) / 2 for a, b in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + 1e-9)
    return thresholds


def tune_threshold(rows: list[dict], feature: str, objective: str) -> tuple[float, dict[str, float | int]]:
    values = [float(row[feature]) for row in rows]
    best_threshold = 0.5
    best_metrics: dict[str, float | int] = {}
    best_score = -1.0
    for threshold in candidate_thresholds(values):
        y_true = [int(row["label"]) for row in rows]
        y_pred = [int(float(row[feature]) >= threshold) for row in rows]
        item = metrics(y_true, y_pred)
        score = float(item[objective])
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = item
    return best_threshold, best_metrics


def summarize_feature(rows: list[dict], feature: str) -> None:
    print(f"\n{feature} distribution:")
    for label in [0, 1]:
        values = [float(row[feature]) for row in rows if int(row["label"]) == label]
        if not values:
            continue
        print(
            f"label={label} n={len(values)} mean={mean(values):.4f} "
            f"median={median(values):.4f} min={min(values):.4f} max={max(values):.4f}"
        )


def summarize_repeated(items: list[dict[str, float | int]], keys: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(item[key]) for item in items]
        out[key] = {
            "mean": round(mean(values), 4),
            "sd": round(stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "median": round(median(values), 4),
            "max": round(max(values), 4),
        }
    return out


def evaluate_split(
    rows: list[dict],
    feature: str,
    test_frac: float,
    seed: int,
    objective: str,
) -> tuple[float, dict[str, float | int], dict[str, float | int]]:
    train_rows, test_rows = stratified_split(rows, test_frac, seed)
    threshold, train_metrics = tune_threshold(train_rows, feature, objective)
    y_true = [int(row["label"]) for row in test_rows]
    y_pred = [int(float(row[feature]) >= threshold) for row in test_rows]
    test_metrics = metrics(y_true, y_pred)
    return threshold, train_metrics, test_metrics


def load_labeled_rows(manifest: Path, split: str, feature: str) -> list[dict]:
    rows = []
    for row in read_csv(manifest):
        if row.get("split") != split:
            continue
        label = int_label(row.get("label_glaucoma"))
        value = fnum(row.get(feature))
        if label is None or value is None:
            continue
        rows.append(
            {
                "image_id": row.get("image_id", ""),
                "split": row.get("split", ""),
                "label": label,
                feature: value,
                "image_path": row.get("image_path", ""),
                "mask_path": row.get("mask_path", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple REFUGE2 CDR structural glaucoma baselines.")
    parser.add_argument("--manifest", type=Path, default=Path("equi-agent/outputs/manifests/refuge2_manifest_v2.csv"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--features", nargs="+", default=["vertical_cup_to_disc_ratio", "cup_to_disc_area_ratio"])
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated stratified holdout splits.")
    parser.add_argument("--objective", choices=["f1", "balanced_accuracy"], default="f1")
    args = parser.parse_args()

    for feature in args.features:
        rows = load_labeled_rows(args.manifest, args.split, feature)
        print(f"\n=== {feature} ===")
        print(f"manifest={args.manifest}")
        print(f"split={args.split} labeled_rows={len(rows)} labels={dict(Counter(row['label'] for row in rows))}")
        if not rows:
            continue
        summarize_feature(rows, feature)

        threshold, train_metrics, test_metrics = evaluate_split(
            rows,
            feature,
            args.test_frac,
            args.seed,
            args.objective,
        )

        print(f"\nthreshold={threshold:.6f} objective={args.objective}")
        print("train_tuning:", {k: round(v, 4) if isinstance(v, float) else v for k, v in train_metrics.items()})
        print("holdout:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()})
        print("holdout_confusion:")
        print(f"  tn={test_metrics['tn']} fp={test_metrics['fp']} fn={test_metrics['fn']} tp={test_metrics['tp']}")

        if args.repeats > 1:
            thresholds = []
            holdouts = []
            for offset in range(args.repeats):
                item_threshold, _, item_metrics = evaluate_split(
                    rows,
                    feature,
                    args.test_frac,
                    args.seed + offset,
                    args.objective,
                )
                thresholds.append(item_threshold)
                holdouts.append(item_metrics)
            print(f"\nrepeated_holdout repeats={args.repeats} seed_start={args.seed}")
            print(
                "threshold_summary:",
                {
                    "mean": round(mean(thresholds), 6),
                    "sd": round(stdev(thresholds), 6) if len(thresholds) > 1 else 0.0,
                    "min": round(min(thresholds), 6),
                    "median": round(median(thresholds), 6),
                    "max": round(max(thresholds), 6),
                },
            )
            print(
                "metric_summary:",
                summarize_repeated(
                    holdouts,
                    ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"],
                ),
            )


if __name__ == "__main__":
    main()
