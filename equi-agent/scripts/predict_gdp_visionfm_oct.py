from __future__ import annotations

import argparse
import os
import pickle
import random
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs(visionfm_root: Path):
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    if not visionfm_root.exists():
        raise FileNotFoundError(f"VisionFM directory not found: {visionfm_root}")
    sys.path.insert(0, str(visionfm_root))
    import models
    import utils

    return (
        np,
        pd,
        torch,
        Image,
        LogisticRegression,
        balanced_accuracy_score,
        f1_score,
        make_pipeline,
        StandardScaler,
        DataLoader,
        transforms,
        tqdm,
        models,
        utils,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a GDP-specific linear probe on frozen VisionFM OCT features "
            "and emit standard-schema Harvard-GDP predictions."
        )
    )
    parser.add_argument("--manifest-dir", type=Path, default=equi_agent_root() / "outputs" / "manifests")
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit manifest CSV. Use this for target-specific GDP progression "
            "manifests such as gdp_progression_forecasting_md_fast_no_p_cut.csv."
        ),
    )
    parser.add_argument("--visionfm-root", type=Path, default=repo_root() / "Foundation_Models" / "VisionFM-main")
    parser.add_argument(
        "--task",
        choices=("glaucoma_detection", "progression_forecasting"),
        required=True,
    )
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default="teacher")
    parser.add_argument("--arch", default="vit_base", choices=("vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge"))
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-blocks", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Prediction CSV path. Defaults to outputs/predictions/gdp_<task>_visionfm_oct.csv.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional pickle path for the probe.")
    parser.add_argument(
        "--threshold-metric",
        choices=("f1", "balanced_accuracy", "fixed_0_5"),
        default="f1",
        help="Decision threshold selection on the GDP training split.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument(
        "--path-prefix-from",
        default=None,
        help="Optional stale prefix in manifest bscan_path values to replace before loading NPZ files.",
    )
    parser.add_argument("--path-prefix-to", default=None, help="Replacement prefix for --path-prefix-from.")
    parser.add_argument("--logreg-c", type=float, default=0.316)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rewrite_bscan_paths(manifest, path_prefix_from: str | None, path_prefix_to: str | None):
    if not path_prefix_from:
        return manifest
    if path_prefix_to is None:
        raise ValueError("--path-prefix-to is required when --path-prefix-from is set")
    manifest = manifest.copy()
    manifest["bscan_path"] = manifest["bscan_path"].astype(str).str.replace(
        path_prefix_from,
        path_prefix_to,
        n=1,
        regex=False,
    )
    return manifest


def split_frame(pd, manifest, split: str, limit: int | None):
    frame = manifest[manifest["split"] == split].copy()
    if limit:
        frame = frame.head(limit).copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame = frame[frame["y_true"].isin([0, 1])].copy()
    return frame


def select_center_bscan(np, volume):
    image = np.asarray(volume)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D B-scan array, got shape={image.shape}")
    slice_axis = min(range(3), key=lambda axis: image.shape[axis])
    center = image.shape[slice_axis] // 2
    return np.take(image, center, axis=slice_axis)


class GDPBscanVisionFMDataset:
    def __init__(self, np, Image, frame, transform):
        self.np = np
        self.Image = Image
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        key = row.get("bscan_key", "bscans")
        with self.np.load(row["bscan_path"]) as data:
            image = select_center_bscan(self.np, data[key])
        if image.max() <= 1.0:
            image = (image * 255).astype("uint8")
        else:
            image = image.astype("uint8")
        image = self.Image.fromarray(image).convert("RGB")
        return self.transform(image), int(row["y_true"]), index


def build_transform(transforms, utils, image_size: int):
    mean, std = utils.get_stats("OCT")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_visionfm_model(args, torch, models, utils, device):
    model = models.__dict__[args.arch](
        img_size=[args.image_size],
        patch_size=args.patch_size,
        num_classes=0,
        use_mean_pooling=False,
    )
    original_torch_load = torch.load

    def torch_load_visionfm_compatible(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_visionfm_compatible
    try:
        utils.load_pretrained_weights(
            model,
            str(args.pretrained_weights),
            args.checkpoint_key,
            args.arch,
            args.patch_size,
        )
    finally:
        torch.load = original_torch_load
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def extract_features(torch, tqdm, model, loader, device, feature_blocks: int):
    features_by_index = {}
    labels_by_index = {}
    with torch.no_grad():
        for images, labels, indices in tqdm(loader, desc="extract", leave=False):
            images = images.to(device, non_blocking=True).to(torch.float32)
            outputs = model.get_intermediate_layers(images, feature_blocks)
            features = torch.cat([layer[:, 0] for layer in outputs], dim=1).detach().cpu().numpy()
            for index, feature, label in zip(indices.numpy().tolist(), features, labels.numpy().tolist()):
                features_by_index[index] = feature
                labels_by_index[index] = int(label)
    return features_by_index, labels_by_index


def features_to_arrays(np, frame, features_by_index, labels_by_index):
    features = []
    labels = []
    for index in range(len(frame)):
        features.append(features_by_index[index])
        labels.append(labels_by_index[index])
    return np.stack(features), np.array(labels)


def probs_by_index(classifier, features_by_index: dict[int, object]) -> dict[int, float]:
    out = {}
    for index, feature in features_by_index.items():
        out[index] = float(classifier.predict_proba(feature.reshape(1, -1))[0, 1])
    return out


def threshold_grid(np, f1_score, balanced_accuracy_score, y_true, y_prob, metric: str) -> float:
    if metric == "fixed_0_5":
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def standard_predictions(pd, frame, probs: dict[int, float], threshold: float):
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        y_prob = float(probs[index])
        rows.append(
            {
                "patient_id": row["patient_id"],
                "eye_id": row.get("eye_id", ""),
                "visit_id": row.get("visit_id", ""),
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "model_name": "visionfm_oct",
                "y_true": int(row["y_true"]),
                "y_prob": y_prob,
                "y_pred": int(y_prob >= threshold),
                "split": row["split"],
                "race": row["race"],
                "ethnicity": row["ethnicity"],
                "sex_gender": row["sex_gender"],
                "age": row["age"],
                "age_group": row["age_group"],
                "metadata_missing_flag": bool(row["metadata_missing_flag"]),
            }
        )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def main() -> None:
    args = parse_args()
    (
        np,
        pd,
        torch,
        Image,
        LogisticRegression,
        balanced_accuracy_score,
        f1_score,
        make_pipeline,
        StandardScaler,
        DataLoader,
        transforms,
        tqdm,
        models,
        utils,
    ) = require_runtime_libs(args.visionfm_root)
    set_seed(torch, args.seed)

    if not args.pretrained_weights.exists():
        raise FileNotFoundError(f"VisionFM weights not found: {args.pretrained_weights}")

    manifest_path = args.manifest_file or (args.manifest_dir / f"gdp_{args.task}.csv")
    manifest = pd.read_csv(manifest_path)
    manifest = rewrite_bscan_paths(manifest, args.path_prefix_from, args.path_prefix_to)
    train_df = split_frame(pd, manifest, "train", args.limit_train)
    test_df = split_frame(pd, manifest, "test", args.limit_test)
    if train_df.empty or test_df.empty:
        raise ValueError(f"Missing train/test rows: train={len(train_df)}, test={len(test_df)}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = build_transform(transforms, utils, args.image_size)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(GDPBscanVisionFMDataset(np, Image, train_df, transform), **loader_kwargs)
    test_loader = DataLoader(GDPBscanVisionFMDataset(np, Image, test_df, transform), **loader_kwargs)
    model = build_visionfm_model(args, torch, models, utils, device)

    train_features, train_labels_by_index = extract_features(
        torch, tqdm, model, train_loader, device, args.feature_blocks
    )
    test_features, _ = extract_features(torch, tqdm, model, test_loader, device, args.feature_blocks)

    x_train, y_train = features_to_arrays(np, train_df, train_features, train_labels_by_index)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=args.seed,
            C=args.logreg_c,
            max_iter=args.max_iter,
            class_weight="balanced",
        ),
    )
    classifier.fit(x_train, y_train)
    train_probs = probs_by_index(classifier, train_features)
    train_prob_array = np.array([train_probs[index] for index in range(len(train_df))])
    threshold = threshold_grid(
        np,
        f1_score,
        balanced_accuracy_score,
        y_train,
        train_prob_array,
        args.threshold_metric,
    )
    test_probs = probs_by_index(classifier, test_features)

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with args.checkpoint.open("wb") as handle:
            pickle.dump(
                {
                    "classifier": classifier,
                    "model_name": "visionfm_oct",
                    "dataset": "harvard_gdp",
                    "task": args.task,
                    "manifest_path": str(manifest_path),
                    "threshold": threshold,
                    "threshold_metric": args.threshold_metric,
                    "image_size": args.image_size,
                    "feature_blocks": args.feature_blocks,
                    "pretrained_weights": str(args.pretrained_weights),
                    "checkpoint_key": args.checkpoint_key,
                    "arch": args.arch,
                },
                handle,
            )
        print(f"wrote_checkpoint={args.checkpoint}")

    predictions = standard_predictions(pd, test_df, test_probs, threshold)
    out_path = args.out or equi_agent_root() / "outputs" / "predictions" / f"gdp_{args.task}_visionfm_oct.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    print(f"wrote={out_path}")
    print(f"rows={len(predictions)}")
    print(f"task={args.task}")
    print(f"threshold={threshold:.3f}")
    print(f"positives={int(predictions['y_true'].sum())} / {len(predictions)}")


if __name__ == "__main__":
    main()
