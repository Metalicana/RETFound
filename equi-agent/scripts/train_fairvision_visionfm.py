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

VISIONFM_MODALITY = {"slo": "Fundus", "oct": "OCT"}
NPZ_KEY = {"slo": "slo_fundus", "oct": "oct_bscans"}


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
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    sys.path.insert(0, str(visionfm_root))
    import models
    import utils

    return np, pd, torch, Image, LogisticRegression, make_pipeline, StandardScaler, DataLoader, transforms, tqdm, models, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a VisionFM frozen-encoder linear probe on FairVision and emit standard predictions."
    )
    parser.add_argument("--manifest-dir", type=Path, default=equi_agent_root() / "outputs" / "manifests")
    parser.add_argument("--visionfm-root", type=Path, default=repo_root() / "Foundation_Models" / "VisionFM-main")
    parser.add_argument("--task", choices=("amd", "dr", "glaucoma"), required=True)
    parser.add_argument("--modality", choices=("slo", "oct"), required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default="teacher")
    parser.add_argument("--arch", default="vit_base", choices=("vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge"))
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-blocks", type=int, default=4)
    parser.add_argument("--out-val", type=Path, required=True)
    parser.add_argument("--out-test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument(
        "--path-prefix-from",
        default=None,
        help="Optional stale prefix in manifest image_path values to replace before loading NPZ files.",
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
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum iterations for the logistic linear probe.",
    )
    return parser.parse_args()


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_name(modality: str) -> str:
    return f"visionfm_{modality}"


def rewrite_image_paths(manifest, path_prefix_from: str | None, path_prefix_to: str | None):
    if not path_prefix_from:
        return manifest
    if path_prefix_to is None:
        raise ValueError("--path-prefix-to is required when --path-prefix-from is set")
    manifest = manifest.copy()
    manifest["image_path"] = manifest["image_path"].astype(str).str.replace(
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


class FairVisionVisionFMDataset:
    def __init__(self, np, Image, frame, modality: str, transform):
        self.np = np
        self.Image = Image
        self.frame = frame.reset_index(drop=True)
        self.modality = modality
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with self.np.load(row["image_path"]) as data:
            image = data[NPZ_KEY[self.modality]]
            if self.modality == "oct":
                image = image[image.shape[0] // 2]
        if image.max() <= 1.0:
            image = (image * 255).astype("uint8")
        else:
            image = image.astype("uint8")
        image = self.Image.fromarray(image).convert("RGB")
        return self.transform(image), int(row["y_true"]), index


def build_transform(transforms, utils, modality: str, image_size: int):
    mean, std = utils.get_stats(VISIONFM_MODALITY[modality])
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
    # VisionFM checkpoints were serialized before PyTorch 2.6 changed
    # torch.load's default to weights_only=True. The downloaded VisionFM
    # checkpoints are trusted project inputs, so keep the upstream loader's
    # behavior but make the trust decision explicit for modern PyTorch.
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


def to_standard_predictions(pd, frame, probs: dict[int, float], task: str, modality: str):
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        y_prob = probs[index]
        rows.append(
            {
                "patient_id": row["patient_id"],
                "eye_id": row.get("eye_id", ""),
                "visit_id": row.get("visit_id", ""),
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "task": task,
                "model_name": model_name(modality),
                "y_true": int(row["y_true"]),
                "y_prob": float(y_prob),
                "y_pred": int(y_prob >= 0.5),
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

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest = pd.read_csv(args.manifest_dir / f"fairvision_{args.task}.csv")
    manifest = rewrite_image_paths(manifest, args.path_prefix_from, args.path_prefix_to)

    train_df = split_frame(pd, manifest, "train", args.limit_train)
    val_df = split_frame(pd, manifest, "val", args.limit_val)
    test_df = split_frame(pd, manifest, "test", args.limit_test)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Missing split rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    transform = build_transform(transforms, utils, args.modality, args.image_size)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        FairVisionVisionFMDataset(np, Image, train_df, args.modality, transform),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        FairVisionVisionFMDataset(np, Image, val_df, args.modality, transform),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        FairVisionVisionFMDataset(np, Image, test_df, args.modality, transform),
        **loader_kwargs,
    )

    model = build_visionfm_model(args, torch, models, utils, device)

    train_features, train_labels_by_index = extract_features(
        torch, tqdm, model, train_loader, device, args.feature_blocks
    )
    val_features, _ = extract_features(torch, tqdm, model, val_loader, device, args.feature_blocks)
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

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with args.checkpoint.open("wb") as handle:
            pickle.dump(
                {
                    "classifier": classifier,
                    "model_name": model_name(args.modality),
                    "task": args.task,
                    "modality": args.modality,
                    "image_size": args.image_size,
                    "feature_blocks": args.feature_blocks,
                    "pretrained_weights": str(args.pretrained_weights),
                    "checkpoint_key": args.checkpoint_key,
                    "arch": args.arch,
                },
                handle,
            )
        print(f"wrote_checkpoint={args.checkpoint}")

    val_predictions = to_standard_predictions(
        pd, val_df, probs_by_index(classifier, val_features), args.task, args.modality
    )
    test_predictions = to_standard_predictions(
        pd, test_df, probs_by_index(classifier, test_features), args.task, args.modality
    )

    args.out_val.parent.mkdir(parents=True, exist_ok=True)
    args.out_test.parent.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(args.out_val, index=False)
    test_predictions.to_csv(args.out_test, index=False)
    print(f"wrote_val={args.out_val} rows={len(val_predictions)}")
    print(f"wrote_test={args.out_test} rows={len(test_predictions)}")


if __name__ == "__main__":
    main()
