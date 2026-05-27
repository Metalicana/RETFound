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
AMD_STAGE_MAP = {
    "normal": 0,
    "no amd": 0,
    "no.amd.diagnosis": 0,
    "not.in.icd.table": 0,
    "early amd": 1,
    "early.dry": 1,
    "intermediate amd": 2,
    "intermediate.dry": 2,
    "late amd": 3,
    "advanced.atrophic.dry.with.subfoveal.involvement": 3,
    "advanced.atrophic.dry.without.subfoveal.involvement": 3,
    "wet.amd.active.choroidal.neovascularization": 3,
    "wet.amd.inactive.choroidal.neovascularization": 3,
    "wet.amd.inactive.scar": 3,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs(retizero_root: Path):
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

    sys.path.insert(0, str(retizero_root))
    from clip_modules.modeling.LoraRETFound import lora

    return np, pd, torch, Image, LogisticRegression, make_pipeline, StandardScaler, DataLoader, transforms, tqdm, lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RetiZero frozen SLO/fundus linear probe on FairVision and emit standard predictions."
    )
    parser.add_argument("--manifest-dir", type=Path, default=equi_agent_root() / "outputs" / "manifests")
    parser.add_argument("--retizero-root", type=Path, default=repo_root() / "Foundation_Models" / "RetiZero-main")
    parser.add_argument("--task", choices=("amd", "dr", "glaucoma"), required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--out-val", type=Path, required=True)
    parser.add_argument("--out-test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--logreg-c", type=float, default=0.316)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument(
        "--fairvision-amd-stages",
        action="store_true",
        help="For AMD, train a 4-class FairVision severity probe and export P(any AMD)=1-P(class 0).",
    )
    return parser.parse_args()


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def normalize_to_uint8(np, image):
    image = np.asarray(image, dtype="float32")
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype="uint8")
    lo, hi = np.percentile(image[finite], [1.0, 99.0])
    if hi <= lo:
        lo = float(np.nanmin(image[finite]))
        hi = float(np.nanmax(image[finite]))
    if hi <= lo:
        return np.zeros(image.shape, dtype="uint8")
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (image * 255.0).astype("uint8")


class FairVisionRetiZeroDataset:
    def __init__(self, np, Image, frame, transform):
        self.np = np
        self.Image = Image
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with self.np.load(row["image_path"]) as data:
            image = data["slo_fundus"]
        image = normalize_to_uint8(self.np, image)
        image = self.Image.fromarray(image).convert("RGB")
        return self.transform(image), int(row["y_true"]), index


def build_transform(transforms, image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def load_retizero_state_dict(torch, weights_path: Path):
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


def build_retizero_model(args, torch, lora, device):
    model = lora(pretrained=False, R=args.lora_rank)
    state_dict = load_retizero_state_dict(torch, args.pretrained_weights)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported RetiZero checkpoint type: {type(state_dict)}")

    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    candidates = [
        strip_prefix(state_dict, "vision_model.model."),
        strip_prefix(state_dict, "img_encoder."),
        state_dict,
    ]
    best = None
    best_loaded = -1
    for candidate in candidates:
        if not candidate:
            continue
        model_keys = set(model.state_dict().keys())
        loaded = len(model_keys.intersection(candidate.keys()))
        if loaded > best_loaded:
            best = candidate
            best_loaded = loaded
    if best is None or best_loaded == 0:
        sample_keys = list(state_dict.keys())[:10]
        raise RuntimeError(f"Could not find RetiZero vision encoder keys. Sample keys: {sample_keys}")

    msg = model.load_state_dict(best, strict=False)
    print(f"loaded_retizero_encoder_keys={best_loaded}")
    print(f"retizero_load_msg={msg}")
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
            features = model(images).detach().cpu().numpy()
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


def fairvision_probe_labels(np, frame, task: str, fairvision_amd_stages: bool):
    if task == "amd" and fairvision_amd_stages:
        raw = frame["label_raw"].astype(str).str.strip().str.lower()
        labels = raw.map(AMD_STAGE_MAP)
        if labels.isna().any():
            raise ValueError(f"Unmapped AMD labels: {sorted(raw[labels.isna()].unique())}")
        return labels.astype(int).to_numpy()
    return frame["y_true"].astype(int).to_numpy()


def probs_by_index(classifier, features_by_index: dict[int, object]) -> dict[int, float]:
    out = {}
    for index, feature in features_by_index.items():
        probs = classifier.predict_proba(feature.reshape(1, -1))[0]
        out[index] = float(1.0 - probs[0]) if len(probs) > 2 else float(probs[1])
    return out


def to_standard_predictions(pd, frame, probs: dict[int, float], task: str):
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
                "model_name": "retizero_slo",
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
        lora,
    ) = require_runtime_libs(args.retizero_root)
    set_seed(torch, args.seed)

    if not args.pretrained_weights.exists():
        raise FileNotFoundError(f"RetiZero weights not found: {args.pretrained_weights}")

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

    transform = build_transform(transforms, args.image_size)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(FairVisionRetiZeroDataset(np, Image, train_df, transform), **loader_kwargs)
    val_loader = DataLoader(FairVisionRetiZeroDataset(np, Image, val_df, transform), **loader_kwargs)
    test_loader = DataLoader(FairVisionRetiZeroDataset(np, Image, test_df, transform), **loader_kwargs)

    model = build_retizero_model(args, torch, lora, device)
    train_features, train_labels_by_index = extract_features(torch, tqdm, model, train_loader, device)
    val_features, _ = extract_features(torch, tqdm, model, val_loader, device)
    test_features, _ = extract_features(torch, tqdm, model, test_loader, device)

    x_train, _ = features_to_arrays(np, train_df, train_features, train_labels_by_index)
    y_train = fairvision_probe_labels(np, train_df, args.task, args.fairvision_amd_stages)
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
                    "model_name": "retizero_slo",
                    "task": args.task,
                    "image_size": args.image_size,
                    "pretrained_weights": str(args.pretrained_weights),
                    "fairvision_amd_stages": args.fairvision_amd_stages,
                },
                handle,
            )
        print(f"wrote_checkpoint={args.checkpoint}")

    val_predictions = to_standard_predictions(pd, val_df, probs_by_index(classifier, val_features), args.task)
    test_predictions = to_standard_predictions(pd, test_df, probs_by_index(classifier, test_features), args.task)

    args.out_val.parent.mkdir(parents=True, exist_ok=True)
    args.out_test.parent.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(args.out_val, index=False)
    test_predictions.to_csv(args.out_test, index=False)
    print(f"wrote_val={args.out_val} rows={len(val_predictions)}")
    print(f"wrote_test={args.out_test} rows={len(test_predictions)}")


if __name__ == "__main__":
    main()
