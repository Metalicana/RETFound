from __future__ import annotations

import argparse
from pathlib import Path


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return np, pd, balanced_accuracy_score, f1_score


def score_threshold(y_true, y_prob, threshold: float, metric: str, balanced_accuracy_score, f1_score) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {metric}")


def tune_one_group(group, metric: str, grid, balanced_accuracy_score, f1_score) -> dict:
    y_true = group["y_true"].astype(int).to_numpy()
    y_prob = group["y_prob"].astype(float).to_numpy()
    best = {"threshold": 0.5, "score": -1.0}
    for threshold in grid:
        score = score_threshold(y_true, y_prob, float(threshold), metric, balanced_accuracy_score, f1_score)
        if score > best["score"]:
            best = {"threshold": float(threshold), "score": score}
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-model/per-task thresholds on validation predictions.")
    parser.add_argument("--validation", type=Path, required=True, help="Validation prediction CSV.")
    parser.add_argument("--test", type=Path, default=None, help="Optional test CSV to rewrite with tuned y_pred.")
    parser.add_argument(
        "--thresholds-out",
        type=Path,
        default=equi_agent_root() / "outputs" / "metrics" / "thresholds.csv",
    )
    parser.add_argument("--test-out", type=Path, default=None, help="Optional threshold-adjusted test prediction CSV.")
    parser.add_argument("--metric", choices=("f1", "balanced_accuracy"), default="f1")
    parser.add_argument("--grid-size", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np, pd, balanced_accuracy_score, f1_score = require_runtime_libs()
    validation = pd.read_csv(args.validation)
    validation = validation[validation["split"].isin(["val", "validation"])].copy()
    if validation.empty:
        raise ValueError("No validation rows found. Generate predictions with --split val first.")

    grid = np.linspace(0.0, 1.0, args.grid_size)
    group_cols = ["dataset", "task", "model_name"]
    rows = []
    for keys, group in validation.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))
        if group["y_true"].nunique(dropna=True) < 2:
            threshold = 0.5
            score = np.nan
        else:
            best = tune_one_group(group, args.metric, grid, balanced_accuracy_score, f1_score)
            threshold = best["threshold"]
            score = best["score"]
        rows.append({**key, "metric": args.metric, "threshold": threshold, "validation_score": score, "n": len(group)})

    thresholds = pd.DataFrame(rows)
    args.thresholds_out.parent.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(args.thresholds_out, index=False)
    print(f"wrote_thresholds={args.thresholds_out}")
    print(thresholds.to_string(index=False))

    if args.test:
        test = pd.read_csv(args.test)
        merged = test.merge(thresholds[group_cols + ["threshold"]], on=group_cols, how="left")
        merged["threshold"] = merged["threshold"].fillna(0.5)
        merged["y_pred"] = (merged["y_prob"] >= merged["threshold"]).astype(int)
        out = args.test_out or args.test.with_name(args.test.stem + "_thresholded.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.drop(columns=["threshold"]).to_csv(out, index=False)
        print(f"wrote_test={out}")
        print(f"rows={len(merged)}")


if __name__ == "__main__":
    main()
