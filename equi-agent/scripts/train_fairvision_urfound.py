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


def require_runtime_libs(urfound_root: Path):
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

    sys.path.insert(0, str(urfound_root))
    import finetune.models_vit as models_vit
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from util.pos_embed import interpolate_pos_embed

    return (
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
        models_vit,
        IMAGENET_DEFAULT_MEAN,
        IMAGENET_DEFAULT_STD,
        interpolate_pos_embed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a UrFound frozen-encoder linear probe on FairVision and emit standard predictions."
    )
    parser.add_argument("--manifest-dir", type=Path, default=equi_agent_root() / "outputs" / "manifests")
    parser.add_argument("--urfound-root", type=Path, default=repo_root() / "Foundation_Models" / "UrFound-main")
    parser.add_argument("--task", choices=("amd", "dr", "glaucoma"), required=True)
    parser.add_argument("--modality", choices=("slo", "oct"), required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--model", default="vit_base_patch16", choices=("vit_base_patch16", "vit_large_patch16"))
    parser.add_argument("--global-pool", action="store_true")
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
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--logreg-c", type=float, default=0.316)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_name(modality: str) -> str:
    return f"urfound_{modality}"


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


class FairVisionUrFoundDataset:
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
            if self.modality == "oct":
                volume = data["oct_bscans"]
                image = volume[volume.shape[0] // 2]
            else:
                image = data["slo_fundus"]
        if image.max() <= 1.0:
            image = (image * 255).astype("uint8")
        else:
            image = image.astype("uint8")
        image = self.Image.fromarray(image).convert("RGB")
        return self.transform(image), int(row["y_true"]), index


def build_transform(transforms, image_size: int, mean, std):
    crop_pct = 224 / 256 if image_size <= 224 else 1.0
    resize_size = int(image_size / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_urfound_checkpoint(torch, weights_path: Path):
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")


def build_urfound_model(args, torch, models_vit, interpolate_pos_embed, device):
    model = models_vit.__dict__[args.model](
        img_size=args.image_size,
        num_classes=0,
        global_pool=args.global_pool,
    )
    checkpoint = load_urfound_checkpoint(torch, args.pretrained_weights)
    checkpoint_model = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(checkpoint_model, dict):
        raise TypeError(f"Unsupported UrFound checkpoint type: {type(checkpoint_model)}")
    checkpoint_model = {
        key.replace("module.", "", 1): value for key, value in checkpoint_model.items()
    }
    state_dict = model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and key not in state_dict:
            del checkpoint_model[key]
        elif key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"urfound_load_msg={msg}")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def extract_features(torch, tqdm, model, loader, device):
    features_by_index = {}
    labels_by_index = {}
    with torch.no_grad():
        for images, labels, indices in tqdm(loader, desc="extract", leave=False):
            images = images.to(device, non_blocking=True).to(torch.float32)
            features = model.forward_features(images).detach().cpu().numpy()
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
        models_vit,
        mean,
        std,
        interpolate_pos_embed,
    ) = require_runtime_libs(args.urfound_root)
    set_seed(torch, args.seed)

    if not args.pretrained_weights.exists():
        raise FileNotFoundError(f"UrFound weights not found: {args.pretrained_weights}")

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

    transform = build_transform(transforms, args.image_size, mean, std)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        FairVisionUrFoundDataset(np, Image, train_df, args.modality, transform), **loader_kwargs
    )
    val_loader = DataLoader(
        FairVisionUrFoundDataset(np, Image, val_df, args.modality, transform), **loader_kwargs
    )
    test_loader = DataLoader(
        FairVisionUrFoundDataset(np, Image, test_df, args.modality, transform), **loader_kwargs
    )

    model = build_urfound_model(args, torch, models_vit, interpolate_pos_embed, device)
    train_features, train_labels_by_index = extract_features(torch, tqdm, model, train_loader, device)
    val_features, _ = extract_features(torch, tqdm, model, val_loader, device)
    test_features, _ = extract_features(torch, tqdm, model, test_loader, device)

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
                    "pretrained_weights": str(args.pretrained_weights),
                    "model": args.model,
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
