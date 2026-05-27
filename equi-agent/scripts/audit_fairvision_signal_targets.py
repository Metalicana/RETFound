from __future__ import annotations

import argparse
from pathlib import Path


AMD_TARGETS = {
    "amd_any": {
        "normal": 0,
        "no amd": 0,
        "no.amd.diagnosis": 0,
        "not.in.icd.table": 0,
        "early amd": 1,
        "early.dry": 1,
        "intermediate amd": 1,
        "intermediate.dry": 1,
        "late amd": 1,
        "advanced.atrophic.dry.with.subfoveal.involvement": 1,
        "advanced.atrophic.dry.without.subfoveal.involvement": 1,
        "wet.amd.active.choroidal.neovascularization": 1,
        "wet.amd.inactive.choroidal.neovascularization": 1,
        "wet.amd.inactive.scar": 1,
    },
    "amd_intermediate_plus": {
        "normal": 0,
        "no amd": 0,
        "no.amd.diagnosis": 0,
        "not.in.icd.table": 0,
        "early amd": 0,
        "early.dry": 0,
        "intermediate amd": 1,
        "intermediate.dry": 1,
        "late amd": 1,
        "advanced.atrophic.dry.with.subfoveal.involvement": 1,
        "advanced.atrophic.dry.without.subfoveal.involvement": 1,
        "wet.amd.active.choroidal.neovascularization": 1,
        "wet.amd.inactive.choroidal.neovascularization": 1,
        "wet.amd.inactive.scar": 1,
    },
    "amd_advanced": {
        "normal": 0,
        "no amd": 0,
        "no.amd.diagnosis": 0,
        "not.in.icd.table": 0,
        "early amd": 0,
        "early.dry": 0,
        "intermediate amd": 0,
        "intermediate.dry": 0,
        "late amd": 1,
        "advanced.atrophic.dry.with.subfoveal.involvement": 1,
        "advanced.atrophic.dry.without.subfoveal.involvement": 1,
        "wet.amd.active.choroidal.neovascularization": 1,
        "wet.amd.inactive.choroidal.neovascularization": 1,
        "wet.amd.inactive.scar": 1,
    },
}


DR_TARGETS = {
    "dr_any": {
        "normal": 0,
        "no dr": 0,
        "no.dr.diagnosis": 0,
        "not.in.icd.table": 0,
        "non-vision threatening dr": 1,
        "non vision threatening dr": 1,
        "mild.npdr": 1,
        "moderate.npdr": 1,
        "vision threatening dr": 1,
        "vision-threatening dr": 1,
        "severe.npdr": 1,
        "pdr": 1,
    },
    "dr_vision_threatening": {
        "normal": 0,
        "no dr": 0,
        "no.dr.diagnosis": 0,
        "not.in.icd.table": 0,
        "non-vision threatening dr": 0,
        "non vision threatening dr": 0,
        "mild.npdr": 0,
        "moderate.npdr": 0,
        "vision threatening dr": 1,
        "vision-threatening dr": 1,
        "severe.npdr": 1,
        "pdr": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast signal audit for alternate FairVision AMD/DR target definitions."
    )
    parser.add_argument("--root", type=Path, default=Path("Datasets/FairVision"))
    parser.add_argument("--task", choices=("amd", "dr"), required=True)
    parser.add_argument("--modalities", nargs="+", choices=("oct", "slo"), default=["oct", "slo"])
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def metadata_path(root: Path, task: str) -> Path:
    source = {"amd": "AMD", "dr": "DR"}[task]
    path = root / "HarvardFairVision30k" / source / "ReadMe" / f"data_summary_{task}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def image_path(root: Path, filename: str, split: str) -> Path:
    folder = {"training": "Training", "validation": "Validation", "test": "Test"}[split]
    return root / folder / filename


def target_maps(task: str) -> dict[str, dict[str, int]]:
    return AMD_TARGETS if task == "amd" else DR_TARGETS


def split_name(value: object) -> str:
    key = str(value).strip().lower()
    return {"train": "training", "val": "validation"}.get(key, key)


def normalize_image(np, image):
    image = np.asarray(image, dtype="float32")
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype="float32")
    lo, hi = np.percentile(image[finite], [1.0, 99.0])
    if hi <= lo:
        lo = float(np.nanmin(image[finite]))
        hi = float(np.nanmax(image[finite]))
    if hi <= lo:
        return np.zeros_like(image, dtype="float32")
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def image_stats(np, image, prefix: str) -> dict[str, float]:
    image = normalize_image(np, image)
    gy, gx = np.gradient(image.astype("float32"))
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        f"{prefix}_mean": float(np.mean(image)),
        f"{prefix}_std": float(np.std(image)),
        f"{prefix}_p01": float(np.percentile(image, 1)),
        f"{prefix}_p10": float(np.percentile(image, 10)),
        f"{prefix}_p50": float(np.percentile(image, 50)),
        f"{prefix}_p90": float(np.percentile(image, 90)),
        f"{prefix}_p99": float(np.percentile(image, 99)),
        f"{prefix}_grad_mean": float(np.mean(grad)),
        f"{prefix}_grad_std": float(np.std(grad)),
    }


def slice_axis(np, volume) -> int:
    volume = np.asarray(volume)
    return min(range(3), key=lambda axis: volume.shape[axis])


def extract_features(np, path: Path, modalities: list[str]) -> dict[str, float]:
    rows: dict[str, float] = {}
    with np.load(path, allow_pickle=True) as data:
        if "slo" in modalities:
            rows.update(image_stats(np, data["slo_fundus"], "slo"))
        if "oct" in modalities:
            volume = np.asarray(data["oct_bscans"], dtype="float32")
            axis = slice_axis(np, volume)
            center = np.take(volume, volume.shape[axis] // 2, axis=axis)
            rows.update(image_stats(np, center, "oct_center"))
            rows.update(image_stats(np, np.mean(volume, axis=axis), "oct_mean"))
            rows.update(image_stats(np, np.max(volume, axis=axis), "oct_max"))
    return rows


def best_threshold(np, sklearn_metrics, y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1.0
    best_threshold_value = 0.5
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = float(score)
            best_threshold_value = float(threshold)
    return best_threshold_value, best_score


def evaluate(np, sklearn_metrics, y_true, y_prob, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(sklearn_metrics.roc_auc_score(y_true, y_prob)),
        "f1": float(sklearn_metrics.f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(sklearn_metrics.balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sklearn_metrics.recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(sklearn_metrics.recall_score(1 - y_true, 1 - y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(y_pred)),
    }


def main() -> None:
    import numpy as np
    import pandas as pd
    from sklearn import ensemble, impute, linear_model, metrics as sklearn_metrics, pipeline, preprocessing

    args = parse_args()
    frame = pd.read_csv(metadata_path(args.root, args.task))
    frame["split_norm"] = frame["use"].map(split_name)
    rows = []
    for _, row in frame.iterrows():
        path = image_path(args.root, str(row["filename"]), row["split_norm"])
        features = extract_features(np, path, args.modalities)
        features["filename"] = str(row["filename"])
        features["split"] = row["split_norm"]
        features["label_raw"] = str(row[args.task]).strip().lower()
        rows.append(features)
    features = pd.DataFrame(rows)

    results = []
    feature_cols = [col for col in features.columns if col not in {"filename", "split", "label_raw"}]
    models = {
        "logreg": pipeline.make_pipeline(
            impute.SimpleImputer(strategy="median"),
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "hgb": pipeline.make_pipeline(
            impute.SimpleImputer(strategy="median"),
            ensemble.HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, l2_regularization=0.1),
        ),
    }

    for target_name, mapping in target_maps(args.task).items():
        labeled = features.copy()
        labeled["y"] = labeled["label_raw"].map(mapping)
        labeled = labeled[labeled["y"].isin([0, 1])].copy()
        split_frames = {}
        for split, limit in [("training", args.limit_train), ("validation", args.limit_val), ("test", args.limit_test)]:
            split_df = labeled[labeled["split"] == split].copy()
            if limit:
                split_df = split_df.head(limit)
            split_frames[split] = split_df
        train_df = split_frames["training"]
        val_df = split_frames["validation"]
        test_df = split_frames["test"]
        if train_df["y"].nunique() < 2 or val_df["y"].nunique() < 2 or test_df["y"].nunique() < 2:
            continue
        for model_name, model in models.items():
            model.fit(train_df[feature_cols], train_df["y"].astype(int))
            val_prob = model.predict_proba(val_df[feature_cols])[:, 1]
            threshold, val_bal = best_threshold(np, sklearn_metrics, val_df["y"].astype(int).to_numpy(), val_prob)
            test_prob = model.predict_proba(test_df[feature_cols])[:, 1]
            row = evaluate(np, sklearn_metrics, test_df["y"].astype(int).to_numpy(), test_prob, threshold)
            row.update(
                {
                    "task": args.task,
                    "target": target_name,
                    "modalities": "+".join(args.modalities),
                    "model": model_name,
                    "threshold": threshold,
                    "val_balanced_accuracy": val_bal,
                    "train_n": len(train_df),
                    "val_n": len(val_df),
                    "test_n": len(test_df),
                    "train_positive_rate": float(train_df["y"].mean()),
                    "val_positive_rate": float(val_df["y"].mean()),
                    "test_positive_rate": float(test_df["y"].mean()),
                }
            )
            results.append(row)

    out = pd.DataFrame(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"wrote={args.out}")
    if out.empty:
        print("No valid target results.")
    else:
        cols = [
            "task",
            "target",
            "modalities",
            "model",
            "auroc",
            "f1",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "positive_rate",
            "threshold",
        ]
        print(out.sort_values(["auroc", "balanced_accuracy"], ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
