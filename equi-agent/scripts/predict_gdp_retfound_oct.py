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


def require_runtime_libs():
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

    sys.path.insert(0, str(equi_agent_root()))
    from VisionAgent.linear_probing_oct3 import get_model_oct

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
        get_model_oct,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate standard-schema Harvard-GDP RETFound OCT predictions for "
            "glaucoma detection or progression forecasting."
        )
    )
    parser.add_argument("--manifest-dir", type=Path, default=equi_agent_root() / "outputs" / "manifests")
    parser.add_argument(
        "--task",
        choices=("glaucoma_detection", "progression_forecasting"),
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("linear-probe", "fairvision-head"),
        default="linear-probe",
        help=(
            "linear-probe trains a balanced logistic head on frozen RETFound GDP train features. "
            "fairvision-head applies the FairVision-trained glaucoma head directly and is only valid "
            "for glaucoma_detection."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=equi_agent_root() / "weights" / "oct_model_best.pth",
        help="FairVision-trained RETFound multi-head checkpoint. Required for fairvision-head mode.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=equi_agent_root() / "VisionAgent" / "weights" / "RETFound_mae_natureOCT.pth",
        help="RETFound OCT MAE backbone checkpoint.",
    )
    parser.add_argument(
        "--skip-backbone-preload",
        action="store_true",
        help="Use when --weights is a full RETFoundMultiHead state dict containing backbone and heads.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Prediction CSV path. Defaults to outputs/predictions/gdp_<task>_retfound_oct.csv.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional pickle path for linear-probe classifier.")
    parser.add_argument(
        "--threshold-metric",
        choices=("f1", "balanced_accuracy", "fixed_0_5"),
        default="f1",
        help="Decision threshold selection on the training split for linear-probe mode.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument(
        "--path-prefix-to",
        default=None,
        help="Replacement prefix for --path-prefix-from.",
    )
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=0.316,
        help="Inverse regularization strength for the balanced logistic linear probe.",
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_retfound_env(args: argparse.Namespace) -> None:
    if args.backbone_weights.exists():
        os.environ["RETFOUND_OCT_BACKBONE_WEIGHTS"] = str(args.backbone_weights)
    if args.skip_backbone_preload or not args.backbone_weights.exists():
        os.environ["RETFOUND_SKIP_OCT_BACKBONE_PRELOAD"] = "1"


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

    # GDP B-scan arrays are commonly H x W x slices; FairVision OCT arrays are
    # commonly slices x H x W. Choose the axis that looks like slice depth.
    slice_axis = min(range(3), key=lambda axis: image.shape[axis])
    center = image.shape[slice_axis] // 2
    return np.take(image, center, axis=slice_axis)


class GDPBscanDataset:
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


def build_transform(transforms, image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_backbone_features(torch, tqdm, model, loader, device):
    features_by_index = {}
    labels_by_index = {}
    model.eval()
    with torch.no_grad():
        for images, labels, indices in tqdm(loader, desc="extract", leave=False):
            images = images.to(device, non_blocking=True).to(torch.float32)
            features = model.backbone(images).detach().cpu().numpy()
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
                "model_name": "retfound_oct",
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


def fairvision_head_probs(torch, tqdm, model, loader, device) -> dict[int, float]:
    probs = {}
    model.eval()
    with torch.no_grad():
        for images, _, indices in tqdm(loader, desc="fairvision-head", leave=False):
            images = images.to(device, non_blocking=True).to(torch.float32)
            outputs = model(images)
            batch_probs = torch.sigmoid(outputs["glaucoma"][:, 0]).detach().cpu().numpy()
            for index, y_prob in zip(indices.numpy().tolist(), batch_probs.tolist()):
                probs[index] = float(y_prob)
    return probs


def main() -> None:
    args = parse_args()
    if args.mode == "fairvision-head" and args.task != "glaucoma_detection":
        raise ValueError("--mode fairvision-head is only valid for --task glaucoma_detection")

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
        get_model_oct,
    ) = require_runtime_libs()
    set_seed(torch, args.seed)
    configure_retfound_env(args)

    if args.mode == "fairvision-head" and not args.weights.exists():
        raise FileNotFoundError(f"RETFound FairVision head weights not found: {args.weights}")
    if not args.backbone_weights.exists() and not args.skip_backbone_preload:
        raise FileNotFoundError(
            f"RETFound OCT backbone weights not found: {args.backbone_weights}. "
            "Set --backbone-weights or use --skip-backbone-preload with a full --weights state dict."
        )

    manifest = pd.read_csv(args.manifest_dir / f"gdp_{args.task}.csv")
    manifest = rewrite_bscan_paths(manifest, args.path_prefix_from, args.path_prefix_to)
    train_df = split_frame(pd, manifest, "train", args.limit_train)
    test_df = split_frame(pd, manifest, "test", args.limit_test)
    if test_df.empty:
        raise ValueError(f"No test rows found for gdp_{args.task}.csv")
    if args.mode == "linear-probe" and train_df.empty:
        raise ValueError(f"No train rows found for gdp_{args.task}.csv")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = build_transform(transforms, args.image_size)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    test_loader = DataLoader(GDPBscanDataset(np, Image, test_df, transform), **loader_kwargs)
    train_loader = None
    if args.mode == "linear-probe":
        train_loader = DataLoader(GDPBscanDataset(np, Image, train_df, transform), **loader_kwargs)

    model = get_model_oct()
    if args.mode == "fairvision-head" or args.skip_backbone_preload:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    if args.mode == "fairvision-head":
        test_probs = fairvision_head_probs(torch, tqdm, model, test_loader, device)
        threshold = 0.5
    else:
        train_features, train_labels_by_index = extract_backbone_features(
            torch, tqdm, model, train_loader, device
        )
        test_features, _ = extract_backbone_features(torch, tqdm, model, test_loader, device)
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
                        "model_name": "retfound_oct",
                        "dataset": "harvard_gdp",
                        "task": args.task,
                        "threshold": threshold,
                        "threshold_metric": args.threshold_metric,
                        "backbone_weights": str(args.backbone_weights),
                    },
                    handle,
                )
            print(f"wrote_checkpoint={args.checkpoint}")

    predictions = standard_predictions(pd, test_df, test_probs, threshold)
    out_path = args.out or equi_agent_root() / "outputs" / "predictions" / f"gdp_{args.task}_retfound_oct.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    print(f"wrote={out_path}")
    print(f"rows={len(predictions)}")
    print(f"task={args.task}")
    print(f"mode={args.mode}")
    print(f"threshold={threshold:.3f}")
    print(f"positives={int(predictions['y_true'].sum())} / {len(predictions)}")


if __name__ == "__main__":
    main()
