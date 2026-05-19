from __future__ import annotations

import argparse
import sys
from pathlib import Path


STANDARD_COLUMNS = [
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
    "split",
    "race",
    "ethnicity",
    "sex_gender",
    "age",
    "age_group",
    "metadata_missing_flag",
]

GDP_TD_COLUMNS = [
    "td1",
    "td2",
    "td3",
    "td4",
    "td5",
    "td6",
    "td7",
    "td8",
    "td9",
    "td10",
    "td11",
    "td12",
    "td13",
    "td14",
    "td15",
    "td16",
    "td17",
    "td18",
    "td19",
    "td20",
    "td21",
    "td22",
    "td23",
    "td24",
    "td26",
    "td27",
    "td28",
    "td29",
    "td30",
    "td31",
    "td32",
    "td33",
    "td35",
    "td36",
    "td37",
    "td38",
    "td39",
    "td40",
    "td41",
    "td42",
    "td43",
    "td44",
    "td45",
    "td46",
    "td47",
    "td48",
    "td49",
    "td50",
    "td51",
    "td52",
    "td53",
    "td54",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs():
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    return np, pd, SimpleImputer, LogisticRegression, f1_score, make_pipeline, StandardScaler, tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train lightweight classical GDP baselines and emit standard-schema predictions."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=equi_agent_root() / "outputs" / "manifests",
        help="Directory containing gdp_glaucoma_detection.csv and gdp_progression_forecasting.csv.",
    )
    parser.add_argument(
        "--task",
        choices=("glaucoma_detection", "progression_forecasting"),
        default="progression_forecasting",
    )
    parser.add_argument(
        "--feature-set",
        choices=("rnflt", "bscan", "clinical", "rnflt_clinical", "bscan_clinical", "all"),
        default="rnflt_clinical",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=equi_agent_root() / "outputs" / "predictions" / "gdp_classical_baseline.csv",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=("f1", "balanced_accuracy", "fixed_0_5"),
        default="f1",
        help="Decision threshold selection on the training split.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")
    return parser.parse_args()


def safe_array_stats(np, array, prefix: str) -> dict[str, float]:
    values = np.asarray(array, dtype="float32")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_{name}": np.nan for name in stat_names()}

    rows = {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        rows[f"{prefix}_p{q}"] = float(np.percentile(values, q))
    return rows


def stat_names() -> list[str]:
    base = ["mean", "std", "min", "max"]
    quantiles = [f"p{q}" for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
    return base + quantiles


def quadrant_stats(np, image, prefix: str) -> dict[str, float]:
    array = np.asarray(image, dtype="float32")
    if array.ndim < 2:
        return {}
    h_mid = array.shape[0] // 2
    w_mid = array.shape[1] // 2
    quadrants = {
        "superior_left": array[:h_mid, :w_mid],
        "superior_right": array[:h_mid, w_mid:],
        "inferior_left": array[h_mid:, :w_mid],
        "inferior_right": array[h_mid:, w_mid:],
    }
    rows = {}
    for name, region in quadrants.items():
        finite = region[np.isfinite(region)]
        rows[f"{prefix}_{name}_mean"] = float(finite.mean()) if finite.size else np.nan
        rows[f"{prefix}_{name}_std"] = float(finite.std()) if finite.size else np.nan
    return rows


def load_rnflt_features(np, path: str) -> dict[str, float]:
    with np.load(path) as data:
        rnflt = data["rnflt"]
    rows = safe_array_stats(np, rnflt, "rnflt")
    rows.update(quadrant_stats(np, rnflt, "rnflt"))
    return rows


def load_bscan_features(np, path: str) -> dict[str, float]:
    with np.load(path) as data:
        bscans = data["bscans"]
    rows = safe_array_stats(np, bscans, "bscan")
    center = bscans[:, :, bscans.shape[2] // 2]
    rows.update(safe_array_stats(np, center, "bscan_center"))
    rows.update(quadrant_stats(np, center, "bscan_center"))
    return rows


def clinical_features(row) -> dict[str, object]:
    rows = {
        "age": row.get("age"),
        "md": row.get("md"),
        "race": row.get("race", "missing"),
        "ethnicity": row.get("ethnicity", "missing"),
        "sex_gender": row.get("sex_gender", "missing"),
        "age_group": row.get("age_group", "missing"),
    }
    for col in GDP_TD_COLUMNS:
        if col in row:
            rows[col] = row.get(col)
    return rows


def build_feature_frame(np, pd, tqdm, manifest, feature_set: str):
    rows = []
    use_rnflt = feature_set in {"rnflt", "rnflt_clinical", "all"}
    use_bscan = feature_set in {"bscan", "bscan_clinical", "all"}
    use_clinical = feature_set in {"clinical", "rnflt_clinical", "bscan_clinical", "all"}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"GDP {feature_set} features"):
        features = {}
        if use_rnflt:
            features.update(load_rnflt_features(np, row["rnflt_path"]))
        if use_bscan:
            features.update(load_bscan_features(np, row["bscan_path"]))
        if use_clinical:
            features.update(clinical_features(row))
        rows.append(features)

    features = pd.DataFrame(rows)
    categorical = [col for col in ["race", "ethnicity", "sex_gender", "age_group"] if col in features.columns]
    if categorical:
        features[categorical] = features[categorical].fillna("missing").astype(str)
        features = pd.get_dummies(features, columns=categorical, dummy_na=False)
    return features.apply(pd.to_numeric, errors="coerce")


def threshold_grid(np, f1_score, y_true, y_prob, metric: str) -> float:
    if metric == "fixed_0_5":
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            positive = y_true == 1
            negative = y_true == 0
            sens = ((pred == 1) & positive).sum() / positive.sum() if positive.sum() else 0.0
            spec = ((pred == 0) & negative).sum() / negative.sum() if negative.sum() else 0.0
            score = (sens + spec) / 2.0
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def model_name(task: str, feature_set: str) -> str:
    return f"gdp_{task}_{feature_set}_logreg"


def standard_rows(pd, manifest, y_prob, threshold: float, task: str, feature_set: str):
    output = manifest.copy()
    output["model_name"] = model_name(task, feature_set)
    output["y_prob"] = y_prob
    output["y_pred"] = (output["y_prob"] >= threshold).astype(int)
    return output[STANDARD_COLUMNS]


def main() -> None:
    args = parse_args()
    np, pd, SimpleImputer, LogisticRegression, f1_score, make_pipeline, StandardScaler, tqdm = require_runtime_libs()

    manifest_path = args.manifest_dir / f"gdp_{args.task}.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"].isin(["train", "test"])].copy()
    manifest = manifest[manifest["y_true"].notna()].copy()
    if args.limit:
        train = manifest[manifest["split"] == "train"].head(args.limit).copy()
        test = manifest[manifest["split"] == "test"].head(args.limit).copy()
        manifest = pd.concat([train, test], ignore_index=True)

    if set(manifest["split"].unique()) != {"train", "test"}:
        raise ValueError(f"Expected train/test split for {manifest_path}; got {manifest['split'].value_counts().to_dict()}")

    features = build_feature_frame(np, pd, tqdm, manifest, args.feature_set)
    train_mask = manifest["split"] == "train"
    test_mask = manifest["split"] == "test"
    train_x = features.loc[train_mask].copy()
    test_x = features.loc[test_mask].copy()
    train_y = manifest.loc[train_mask, "y_true"].astype(int).to_numpy()

    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026),
    )
    clf.fit(train_x, train_y)
    train_prob = clf.predict_proba(train_x)[:, 1]
    test_prob = clf.predict_proba(test_x)[:, 1]
    threshold = threshold_grid(np, f1_score, train_y, train_prob, args.threshold_metric)

    predictions = standard_rows(
        pd,
        manifest.loc[test_mask].copy(),
        test_prob,
        threshold,
        args.task,
        args.feature_set,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)
    print(f"wrote={args.out}")
    print(f"rows={len(predictions)}")
    print(f"task={args.task}")
    print(f"feature_set={args.feature_set}")
    print(f"threshold={threshold:.3f}")
    print(f"positives={int(predictions['y_true'].sum())} / {len(predictions)}")


if __name__ == "__main__":
    main()
