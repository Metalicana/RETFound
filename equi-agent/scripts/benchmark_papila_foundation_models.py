#!/usr/bin/env python3
"""Benchmark frozen foundation-model CFP features on the locked PAPILA split.

Each model uses its native image preprocessing and pretrained checkpoint, but
all models share the same downstream probe search, validation-F1 threshold
selection, test evaluation, subgroup definitions, and bootstrap protocol.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MODELS = ("retfound", "mirage", "ret_clip", "retizero", "urfound")
GENDER_MAP = {
    "0": "male",
    "male": "male",
    "1": "female",
    "female": "female",
}
SPLITS = ("train", "val", "test")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "OphthalmicAgent" / "data_papila" / "manifest.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "equi-agent" / "outputs" / "papila_foundation_benchmark",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default=None, help="Example: cuda:0, cuda:1, or cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Smoke tests only.")
    parser.add_argument(
        "--reuse-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--probe-kinds", nargs="+", choices=("logreg", "mlp"), default=["logreg", "mlp"])
    parser.add_argument("--logreg-c", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--mlp-hidden-dims", nargs="+", type=int, default=[0, 128, 256])
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-dropout", type=float, default=0.2)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--min-subgroup-n", type=int, default=10)
    parser.add_argument("--min-subgroup-positive", type=int, default=2)
    parser.add_argument("--min-subgroup-negative", type=int, default=2)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Load checkpoints and cache features without fitting probes.",
    )
    parser.add_argument("--summarize-only", action="store_true")

    parser.add_argument(
        "--retfound-weights",
        type=Path,
        default=root / "OphthalmicAgent" / "weights" / "cfp_model.pth",
    )
    parser.add_argument(
        "--mirage-dir",
        type=Path,
        default=root / "equi-agent" / "VisionAgent" / "MIRAGE",
    )
    parser.add_argument(
        "--mirage-feature-module",
        default=None,
        help="Optional named module whose input is used as the MIRAGE feature.",
    )
    parser.add_argument(
        "--ret-clip-root",
        type=Path,
        default=root / "Foundation_Models" / "RET-CLIP-main",
    )
    parser.add_argument(
        "--ret-clip-weights",
        type=Path,
        default=root / "Foundation_Models" / "RET-CLIP-main" / "pretrained" / "ret_clip_vit_b_16.pt",
    )
    parser.add_argument(
        "--retizero-root",
        type=Path,
        default=root / "Foundation_Models" / "RetiZero-main",
    )
    parser.add_argument(
        "--retizero-weights",
        type=Path,
        default=root / "Foundation_Models" / "RetiZero-main" / "pretrained" / "RetiZero.pth",
    )
    parser.add_argument("--retizero-lora-rank", type=int, default=8)
    parser.add_argument(
        "--urfound-root",
        type=Path,
        default=root / "Foundation_Models" / "UrFound-main",
    )
    parser.add_argument(
        "--urfound-weights",
        type=Path,
        default=root / "Foundation_Models" / "UrFound-main" / "pretrained" / "urfound_mm.pth",
    )
    parser.add_argument(
        "--urfound-model",
        choices=("vit_base_patch16", "vit_large_patch16"),
        default="vit_base_patch16",
    )
    parser.add_argument("--urfound-global-pool", action="store_true")
    return parser.parse_args()


def normalize_gender(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in GENDER_MAP:
        raise ValueError(f"Unknown PAPILA gender value: {value!r}")
    return GENDER_MAP[normalized]


def optional_float(value: object) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def resolve_image_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    ophthalmic_candidate = manifest.parent.parent / path
    if ophthalmic_candidate.exists():
        return ophthalmic_candidate
    return manifest.parent / path


def load_manifest(np, path: Path, limit_per_split: int | None) -> tuple[dict[str, list[dict[str, Any]]], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"PAPILA manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        split = str(source.get("split", "")).strip().lower()
        label = str(source.get("label", "")).strip()
        if split not in SPLITS or label not in {"0", "1"}:
            continue
        image_path = resolve_image_path(path, str(source.get("cfp_path", "")))
        row = dict(source)
        row.update(
            {
                "split": split,
                "y_true": int(label),
                "case_id": str(source.get("case_id", "")).strip(),
                "patient_id": str(source.get("patient_id", "")).strip(),
                "image_path": str(image_path),
                "sex_gender": normalize_gender(source.get("gender_code", "")),
                "age_value": optional_float(source.get("age")),
            }
        )
        if not row["case_id"] or not row["patient_id"]:
            raise ValueError("Every PAPILA row must have case_id and patient_id")
        rows.append(row)

    patient_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_splits[row["patient_id"]].add(row["split"])
    leakage = {patient: values for patient, values in patient_splits.items() if len(values) != 1}
    if leakage:
        raise ValueError(f"Patient split leakage detected for {len(leakage)} patients")

    train_ages = np.asarray(
        [row["age_value"] for row in rows if row["split"] == "train" and row["age_value"] is not None],
        dtype=float,
    )
    if len(train_ages) < 3:
        raise ValueError("At least three training ages are required for age groups")
    lower, upper = [float(value) for value in np.quantile(train_ages, [1 / 3, 2 / 3])]
    if lower >= upper:
        raise ValueError(f"Training age tertiles collapsed: lower={lower}, upper={upper}")

    for row in rows:
        age = row["age_value"]
        if age is None:
            row["age_group"] = "unknown"
        elif age <= lower:
            row["age_group"] = f"age_le_{lower:g}"
        elif age <= upper:
            row["age_group"] = f"age_{lower:g}_to_{upper:g}"
        else:
            row["age_group"] = f"age_gt_{upper:g}"

    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    if limit_per_split:
        by_split = {split: values[:limit_per_split] for split, values in by_split.items()}
    if any(not by_split[split] for split in SPLITS):
        raise ValueError(f"Missing split rows: { {key: len(value) for key, value in by_split.items()} }")
    return by_split, [lower, upper]


def set_seed(np, torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


@dataclass
class FeatureExtractor:
    model: Any
    transform: Any
    image_mode: str
    forward: Callable[[Any], Any]
    provenance: dict[str, Any]


class PapilaImageDataset:
    def __init__(self, Image, rows: list[dict[str, Any]], transform, image_mode: str):
        self.Image = Image
        self.rows = rows
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path = Path(self.rows[index]["image_path"])
        if not path.exists():
            raise FileNotFoundError(f"PAPILA CFP image not found: {path}")
        with self.Image.open(path) as source:
            image = source.convert(self.image_mode)
        return self.transform(image), index


def best_state_candidate(torch, model, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
    raw = {key.removeprefix("module."): value for key, value in state.items()}
    candidates = [raw]
    for prefix in ("backbone.", "vision_model.model.", "img_encoder."):
        stripped = {key[len(prefix) :]: value for key, value in raw.items() if key.startswith(prefix)}
        if stripped:
            candidates.append(stripped)
    model_state = model.state_dict()
    best: dict[str, Any] = {}
    for candidate in candidates:
        compatible = {
            key: value
            for key, value in candidate.items()
            if key in model_state
            and hasattr(value, "shape")
            and tuple(value.shape) == tuple(model_state[key].shape)
        }
        if len(compatible) > len(best):
            best = compatible
    return best, len(best)


def build_retfound(args, torch, transforms, device) -> FeatureExtractor:
    require_file(args.retfound_weights, "RETFound checkpoint")
    ophthalmic_root = repo_root() / "OphthalmicAgent"
    sys.path.insert(0, str(ophthalmic_root))
    from VisionAgent.models_vit import RETFound_mae

    model = RETFound_mae(img_size=224, num_classes=0, drop_path_rate=0.0, global_pool="")
    checkpoint = safe_torch_load(torch, args.retfound_weights)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported RETFound checkpoint: {type(state)}")
    compatible, loaded = best_state_candidate(torch, model, state)
    if loaded < 100:
        raise RuntimeError(f"RETFound checkpoint matched only {loaded} tensors")
    message = model.load_state_dict(compatible, strict=False)
    print(f"retfound_loaded_tensors={loaded} load_message={message}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return FeatureExtractor(
        model=model,
        transform=transform,
        image_mode="RGB",
        forward=lambda images: model(images),
        provenance={"weights": str(args.retfound_weights), "feature": "RETFound backbone output"},
    )


def mirage_feature_module(torch, model, requested: str | None):
    if requested:
        return requested, model.get_submodule(requested)
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lower = name.lower()
        score = 0
        if "glaucoma" in lower:
            score += 10000
        elif "glau" in lower or lower.startswith("gl") or ".gl" in lower:
            score += 5000
        if getattr(module, "out_features", 0) in {1, 3}:
            score += 1000
        score += int(getattr(module, "in_features", 0))
        candidates.append((score, name, module))
    if not candidates:
        raise RuntimeError("No MIRAGE linear head found; pass --mirage-feature-module")
    _, name, module = max(candidates, key=lambda item: item[0])
    return name, module


def build_mirage(args, torch, transforms, device) -> FeatureExtractor:
    require_file(args.mirage_dir / "MIRAGE-Base.pth", "MIRAGE base checkpoint")
    sys.path.insert(0, str(args.mirage_dir))
    original_dir = Path.cwd()
    os.chdir(args.mirage_dir)
    try:
        from linear_probing_slo import get_model_slo

        model = get_model_slo()
    finally:
        os.chdir(original_dir)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    module_name, module = mirage_feature_module(torch, model, args.mirage_feature_module)
    captured: dict[str, Any] = {}

    def capture_input(_module, inputs):
        if not inputs:
            raise RuntimeError(f"MIRAGE feature module {module_name} received no input")
        captured["features"] = inputs[0]

    module.register_forward_pre_hook(capture_input)

    def forward(images):
        captured.clear()
        model(images)
        if "features" not in captured:
            raise RuntimeError(f"MIRAGE hook did not fire for {module_name}")
        return captured["features"]

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    print(f"mirage_feature_module={module_name} module={module}")
    return FeatureExtractor(
        model=model,
        transform=transform,
        image_mode="L",
        forward=forward,
        provenance={
            "weights": str(args.mirage_dir / "MIRAGE-Base.pth"),
            "feature_module": module_name,
        },
    )


def build_ret_clip(args, torch, _transforms, device) -> FeatureExtractor:
    require_file(args.ret_clip_weights, "RET-CLIP checkpoint")
    sys.path.insert(0, str(args.ret_clip_root))
    from RET_CLIP.clip.utils import create_model, image_transform

    checkpoint = safe_torch_load(torch, args.ret_clip_weights)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        wrapped = checkpoint
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        wrapped = {"state_dict": checkpoint["model"]}
    elif isinstance(checkpoint, dict):
        wrapped = {"state_dict": checkpoint}
    else:
        raise TypeError(f"Unsupported RET-CLIP checkpoint: {type(checkpoint)}")
    model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", wrapped)
    model.float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    def forward(images):
        features = model.encode_image(img_l=images.to(torch.float32), img_r=None)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return FeatureExtractor(
        model=model,
        transform=image_transform(224),
        image_mode="RGB",
        forward=forward,
        provenance={"weights": str(args.ret_clip_weights), "vision_model": "ViT-B-16"},
    )


def build_retizero(args, torch, transforms, device) -> FeatureExtractor:
    require_file(args.retizero_weights, "RetiZero checkpoint")
    sys.path.insert(0, str(args.retizero_root))
    from clip_modules.modeling.LoraRETFound import lora

    model = lora(pretrained=False, R=args.retizero_lora_rank)
    checkpoint = safe_torch_load(torch, args.retizero_weights)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported RetiZero checkpoint: {type(state)}")
    compatible, loaded = best_state_candidate(torch, model, state)
    if loaded < 100:
        raise RuntimeError(f"RetiZero checkpoint matched only {loaded} tensors")
    message = model.load_state_dict(compatible, strict=False)
    print(f"retizero_loaded_tensors={loaded} load_message={message}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return FeatureExtractor(
        model=model,
        transform=transform,
        image_mode="RGB",
        forward=lambda images: model(images),
        provenance={"weights": str(args.retizero_weights), "lora_rank": args.retizero_lora_rank},
    )


def build_urfound(args, torch, transforms, device) -> FeatureExtractor:
    require_file(args.urfound_weights, "UrFound checkpoint")
    sys.path.insert(0, str(args.urfound_root))
    import finetune.models_vit as models_vit
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from util.pos_embed import interpolate_pos_embed

    model = models_vit.__dict__[args.urfound_model](
        img_size=224,
        num_classes=0,
        global_pool=args.urfound_global_pool,
    )
    checkpoint = safe_torch_load(torch, args.urfound_weights)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported UrFound checkpoint: {type(state)}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model_state = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in state and (key not in model_state or state[key].shape != model_state[key].shape):
            del state[key]
    interpolate_pos_embed(model, state)
    compatible, loaded = best_state_candidate(torch, model, state)
    if loaded < 100:
        raise RuntimeError(f"UrFound checkpoint matched only {loaded} tensors")
    message = model.load_state_dict(compatible, strict=False)
    print(f"urfound_loaded_tensors={loaded} load_message={message}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return FeatureExtractor(
        model=model,
        transform=transform,
        image_mode="RGB",
        forward=lambda images: model.forward_features(images),
        provenance={
            "weights": str(args.urfound_weights),
            "model": args.urfound_model,
            "global_pool": args.urfound_global_pool,
        },
    )


BUILDERS = {
    "retfound": build_retfound,
    "mirage": build_mirage,
    "ret_clip": build_ret_clip,
    "retizero": build_retizero,
    "urfound": build_urfound,
}


def cache_path(out_dir: Path, model_name: str, split: str) -> Path:
    return out_dir / "feature_cache" / model_name / f"{split}.npz"


def model_weight_path(args, model_name: str) -> Path:
    return {
        "retfound": args.retfound_weights,
        "mirage": args.mirage_dir / "MIRAGE-Base.pth",
        "ret_clip": args.ret_clip_weights,
        "retizero": args.retizero_weights,
        "urfound": args.urfound_weights,
    }[model_name]


def feature_cache_signature(args, model_name: str) -> str:
    weights = model_weight_path(args, model_name)
    require_file(weights, f"{model_name} checkpoint")
    require_file(args.manifest, "PAPILA manifest")
    weight_stat = weights.stat()
    manifest_stat = args.manifest.stat()
    payload = {
        "model_name": model_name,
        "weights": str(weights.resolve()),
        "weights_size": weight_stat.st_size,
        "weights_mtime_ns": weight_stat.st_mtime_ns,
        "manifest": str(args.manifest.resolve()),
        "manifest_size": manifest_stat.st_size,
        "manifest_mtime_ns": manifest_stat.st_mtime_ns,
        "limit_per_split": args.limit_per_split,
        "mirage_feature_module": args.mirage_feature_module if model_name == "mirage" else None,
        "retizero_lora_rank": args.retizero_lora_rank if model_name == "retizero" else None,
        "urfound_model": args.urfound_model if model_name == "urfound" else None,
        "urfound_global_pool": args.urfound_global_pool if model_name == "urfound" else None,
    }
    return json.dumps(payload, sort_keys=True)


def load_cached_features(np, path: Path, rows: list[dict[str, Any]], signature: str):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        cached_signature = str(payload["signature"].item()) if "signature" in payload.files else ""
        if cached_signature != signature:
            print(f"feature_cache_invalidated={path}")
            return None
        case_ids = payload["case_ids"].astype(str).tolist()
        expected = [row["case_id"] for row in rows]
        if case_ids != expected:
            raise ValueError(f"Feature cache case order mismatch: {path}")
        return payload["features"].astype("float32")


def feature_tensor(torch, value):
    if isinstance(value, (list, tuple)):
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Feature extractor returned {type(value)}, expected Tensor")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim > 2:
        value = value.flatten(start_dim=1)
    return value


def extract_model_features(np, torch, Image, DataLoader, transforms, tqdm, args, model_name, by_split, device):
    signature = feature_cache_signature(args, model_name)
    cached = {}
    if args.reuse_feature_cache:
        cached = {
            split: load_cached_features(
                np,
                cache_path(args.out_dir, model_name, split),
                rows,
                signature,
            )
            for split, rows in by_split.items()
        }
    if all(cached.get(split) is not None for split in SPLITS):
        print(f"model={model_name} feature_cache=hit")
        return cached, {"feature_cache": "reused"}

    extractor = BUILDERS[model_name](args, torch, transforms, device)
    features = {}
    for split, rows in by_split.items():
        if cached.get(split) is not None:
            features[split] = cached[split]
            continue
        dataset = PapilaImageDataset(Image, rows, extractor.transform, extractor.image_mode)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        batches = []
        with torch.inference_mode():
            for images, _indices in tqdm(loader, desc=f"{model_name}:{split}", leave=False):
                images = images.to(device, non_blocking=True)
                values = feature_tensor(torch, extractor.forward(images))
                batches.append(values.detach().to(torch.float32).cpu().numpy())
        matrix = np.concatenate(batches, axis=0).astype("float32")
        if matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
            raise RuntimeError(f"Invalid {model_name}/{split} feature matrix: {matrix.shape}")
        path = cache_path(args.out_dir, model_name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=matrix,
            case_ids=np.asarray([row["case_id"] for row in rows]),
            signature=np.asarray(signature),
        )
        features[split] = matrix
        print(f"model={model_name} split={split} features={matrix.shape} cache={path}")

    provenance = extractor.provenance
    del extractor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, provenance


def confusion_metrics(np, y_true, y_prob, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "n": int(len(y_true)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "accuracy": divide(tp + tn, len(y_true)),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_candidates(np, probabilities):
    values = np.unique(np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0))
    return np.unique(np.concatenate(([0.0], values, [np.nextafter(1.0, 2.0)])))


def select_f1_threshold(np, y_true, probabilities) -> tuple[float, dict[str, Any]]:
    best = None
    for threshold in threshold_candidates(np, probabilities):
        metrics = confusion_metrics(np, y_true, probabilities, float(threshold))
        key = (
            metrics["f1"] if metrics["f1"] is not None else -1.0,
            metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return best[1], best[2]


def roc_auc(np, y_true, probabilities) -> float | None:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None


def torch_probabilities(np, torch, probe, features, device):
    matrix = np.nan_to_num(features.astype("float32"), copy=False)
    matrix = (matrix - probe.mean) / probe.std
    probe.model.to(device).eval()
    with torch.no_grad():
        logits = probe.model(torch.tensor(matrix, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def train_probe_search(np, torch, args, features, by_split, device, model_dir: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from torch_feature_probe import train_torch_probe, torch_probe_checkpoint

    x_train, x_val = features["train"], features["val"]
    y_train = np.asarray([row["y_true"] for row in by_split["train"]], dtype=int)
    y_val = np.asarray([row["y_true"] for row in by_split["val"]], dtype=int)
    search_rows = []
    best = None

    def consider(name, kind, fitted, val_probs, config):
        nonlocal best
        threshold, metrics = select_f1_threshold(np, y_val, val_probs)
        auc = roc_auc(np, y_val, val_probs)
        row = {
            "candidate": name,
            "probe_kind": kind,
            "threshold": threshold,
            "val_f1": metrics["f1"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_sensitivity": metrics["sensitivity"],
            "val_specificity": metrics["specificity"],
            "val_auroc": auc,
            "config_json": json.dumps(config, sort_keys=True),
        }
        search_rows.append(row)
        key = (
            metrics["f1"] if metrics["f1"] is not None else -1.0,
            metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
            auc if auc is not None else -1.0,
            -abs(threshold - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, name, kind, fitted, threshold, config, val_probs)

    if "logreg" in args.probe_kinds:
        for c_value in args.logreg_c:
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=args.seed,
                ),
            )
            classifier.fit(x_train, y_train)
            val_probs = classifier.predict_proba(x_val)[:, 1]
            consider(
                f"logreg_c_{c_value:g}",
                "logreg",
                classifier,
                val_probs,
                {"C": c_value, "class_weight": "balanced", "scaled": True},
            )

    if "mlp" in args.probe_kinds:
        for hidden_dim in args.mlp_hidden_dims:
            torch.manual_seed(args.seed + hidden_dim)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + hidden_dim)
            probe = train_torch_probe(
                np,
                torch,
                x_train,
                y_train,
                x_val,
                y_val,
                device=device,
                seed=args.seed + hidden_dim,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                weight_decay=args.probe_weight_decay,
                hidden_dim=hidden_dim,
                dropout=args.probe_dropout,
                batch_size=args.probe_batch_size,
            )
            val_probs = torch_probabilities(np, torch, probe, x_val, device)
            consider(
                f"torch_hidden_{hidden_dim}",
                "mlp",
                probe,
                val_probs,
                {
                    "hidden_dim": hidden_dim,
                    "epochs": args.probe_epochs,
                    "best_epoch": probe.best_epoch,
                    "lr": args.probe_lr,
                    "weight_decay": args.probe_weight_decay,
                    "dropout": args.probe_dropout,
                },
            )

    if best is None:
        raise RuntimeError("Probe search produced no candidates")
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "probe_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(search_rows[0]))
        writer.writeheader()
        writer.writerows(search_rows)

    _, candidate, kind, fitted, threshold, config, val_probs = best
    if kind == "logreg":
        with (model_dir / "selected_probe.pkl").open("wb") as handle:
            pickle.dump(fitted, handle)
        test_probs = fitted.predict_proba(features["test"])[:, 1]
    else:
        torch.save(
            torch_probe_checkpoint(
                fitted,
                {"candidate": candidate, "threshold": threshold, "config": config},
            ),
            model_dir / "selected_probe.pth",
        )
        test_probs = torch_probabilities(np, torch, fitted, features["test"], device)
    return {
        "candidate": candidate,
        "kind": kind,
        "threshold": float(threshold),
        "config": config,
        "probabilities": {"val": np.asarray(val_probs), "test": np.asarray(test_probs)},
        "search": search_rows,
    }


def subgroup_metrics(np, rows, probabilities, threshold, args):
    output = []
    for attribute in ("sex_gender", "age_group"):
        values = sorted({str(row.get(attribute, "unknown")) for row in rows})
        for value in values:
            indices = [index for index, row in enumerate(rows) if str(row.get(attribute, "unknown")) == value]
            labels = np.asarray([rows[index]["y_true"] for index in indices], dtype=int)
            probs = np.asarray([probabilities[index] for index in indices], dtype=float)
            metrics = confusion_metrics(np, labels, probs, threshold)
            eligible = (
                metrics["n"] >= args.min_subgroup_n
                and metrics["positive_n"] >= args.min_subgroup_positive
                and metrics["negative_n"] >= args.min_subgroup_negative
            )
            output.append(
                {
                    "attribute": attribute,
                    "group": value,
                    "eligible_for_worst_group": eligible,
                    **metrics,
                }
            )
    eligible_f1 = [row["f1"] for row in output if row["eligible_for_worst_group"] and row["f1"] is not None]
    return output, min(eligible_f1) if eligible_f1 else None


def bootstrap_intervals(np, rows, probabilities, threshold, args):
    if args.bootstrap_replicates <= 0:
        return {}
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_patient[row["patient_id"]].append(index)
    patients = sorted(by_patient)
    rng = np.random.default_rng(args.seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(args.bootstrap_replicates):
        selected_patients = rng.choice(patients, size=len(patients), replace=True)
        indices = [index for patient in selected_patients for index in by_patient[str(patient)]]
        sampled_rows = [rows[index] for index in indices]
        sampled_probs = np.asarray([probabilities[index] for index in indices], dtype=float)
        labels = np.asarray([row["y_true"] for row in sampled_rows], dtype=int)
        metrics = confusion_metrics(np, labels, sampled_probs, threshold)
        _, worst_f1 = subgroup_metrics(np, sampled_rows, sampled_probs, threshold, args)
        metrics["worst_group_f1"] = worst_f1
        for key in ("f1", "sensitivity", "specificity", "balanced_accuracy", "worst_group_f1"):
            if metrics.get(key) is not None:
                samples[key].append(float(metrics[key]))
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        }
        for key, values in samples.items()
        if values
    }


def write_predictions(path, model_name, rows, probabilities, threshold):
    fields = [
        "dataset",
        "task",
        "model_name",
        "patient_id",
        "case_id",
        "eye",
        "split",
        "y_true",
        "y_prob",
        "threshold",
        "y_pred",
        "sex_gender",
        "age",
        "age_group",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            writer.writerow(
                {
                    "dataset": "papila",
                    "task": "glaucoma",
                    "model_name": model_name,
                    "patient_id": row["patient_id"],
                    "case_id": row["case_id"],
                    "eye": row.get("eye", ""),
                    "split": row["split"],
                    "y_true": row["y_true"],
                    "y_prob": float(probability),
                    "threshold": threshold,
                    "y_pred": int(probability >= threshold),
                    "sex_gender": row["sex_gender"],
                    "age": row.get("age", ""),
                    "age_group": row["age_group"],
                }
            )


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_model(np, torch, Image, DataLoader, transforms, tqdm, args, model_name, by_split, age_bounds, device):
    print(f"\n=== PAPILA model={model_name} ===", flush=True)
    model_dir = args.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    features, provenance = extract_model_features(
        np, torch, Image, DataLoader, transforms, tqdm, args, model_name, by_split, device
    )
    if args.extract_only:
        payload = {
            "model_name": model_name,
            "feature_provenance": provenance,
            "feature_shapes": {split: list(matrix.shape) for split, matrix in features.items()},
        }
        (model_dir / "feature_smoke.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    selected = train_probe_search(np, torch, args, features, by_split, device, model_dir)
    threshold = selected["threshold"]
    split_metrics = {}
    all_subgroups = []
    for split in ("val", "test"):
        rows = by_split[split]
        probs = selected["probabilities"][split]
        labels = np.asarray([row["y_true"] for row in rows], dtype=int)
        metrics = confusion_metrics(np, labels, probs, threshold)
        metrics["auroc"] = roc_auc(np, labels, probs)
        groups, worst_f1 = subgroup_metrics(np, rows, probs, threshold, args)
        metrics["worst_group_f1"] = worst_f1
        split_metrics[split] = metrics
        for group in groups:
            all_subgroups.append({"split": split, **group})
        write_predictions(model_dir / f"predictions_{split}.csv", model_name, rows, probs, threshold)

    intervals = bootstrap_intervals(
        np,
        by_split["test"],
        selected["probabilities"]["test"],
        threshold,
        args,
    )
    write_csv_rows(model_dir / "subgroup_metrics.csv", all_subgroups)
    summary = {
        "dataset": "papila",
        "task": "glaucoma",
        "model_name": model_name,
        "manifest": str(args.manifest),
        "rows": {split: len(rows) for split, rows in by_split.items()},
        "age_group_train_tertiles": age_bounds,
        "gender_normalization": GENDER_MAP,
        "selected_probe": {
            "candidate": selected["candidate"],
            "kind": selected["kind"],
            "threshold_selected_on": "validation_f1",
            "threshold": threshold,
            "config": selected["config"],
        },
        "feature_provenance": provenance,
        "metrics": split_metrics,
        "test_patient_bootstrap_95_ci": intervals,
        "worst_group_policy": {
            "attributes": ["sex_gender", "age_group"],
            "age_groups": "tertiles derived from training ages only",
            "minimum_n": args.min_subgroup_n,
            "minimum_positive": args.min_subgroup_positive,
            "minimum_negative": args.min_subgroup_negative,
            "intersections": False,
        },
        "seed": args.seed,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"model": model_name, "selected_probe": summary["selected_probe"], "test": split_metrics["test"]}, indent=2))


def summarize_results(out_dir: Path) -> None:
    rows = []
    for model_name in MODELS:
        path = out_dir / model_name / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]["test"]
        intervals = summary.get("test_patient_bootstrap_95_ci", {})
        row = {
            "model_name": model_name,
            "f1": metrics.get("f1"),
            "worst_group_f1": metrics.get("worst_group_f1"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "auroc": metrics.get("auroc"),
            "threshold": summary["selected_probe"]["threshold"],
            "selected_probe": summary["selected_probe"]["candidate"],
            "f1_ci": json.dumps(intervals.get("f1", {}), sort_keys=True),
            "worst_group_f1_ci": json.dumps(intervals.get("worst_group_f1", {}), sort_keys=True),
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No model summaries found under {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_dir / "papila_glaucoma_benchmark.csv", rows)
    lines = [
        "# PAPILA Glaucoma Foundation Benchmark",
        "",
        "| Model | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = lambda key: "NA" if row[key] is None else f"{float(row[key]):.3f}"
        lines.append(
            f"| {row['model_name']} | {value('f1')} | {value('worst_group_f1')} | "
            f"{value('sensitivity')} | {value('specificity')} | "
            f"{value('balanced_accuracy')} | {value('auroc')} |"
        )
    (out_dir / "papila_glaucoma_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        summarize_results(args.out_dir)
        return

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    set_seed(np, torch, args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    by_split, age_bounds = load_manifest(np, args.manifest, args.limit_per_split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "manifest": str(args.manifest),
        "models": args.models,
        "split_rows": {split: len(rows) for split, rows in by_split.items()},
        "split_labels": {
            split: dict(Counter(str(row["y_true"]) for row in rows))
            for split, rows in by_split.items()
        },
        "age_group_train_tertiles": age_bounds,
        "gender_mapping_verified_from_244_bilateral_pairs": GENDER_MAP,
        "selection_objective": "validation_f1",
        "test_used_for_selection": False,
        "device": str(device),
        "seed": args.seed,
    }
    protocol_name = (
        "protocol.json"
        if set(args.models) == set(MODELS)
        else f"protocol_{'_'.join(args.models)}.json"
    )
    (args.out_dir / protocol_name).write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))

    for model_name in args.models:
        run_model(np, torch, Image, DataLoader, transforms, tqdm, args, model_name, by_split, age_bounds, device)
    if args.extract_only:
        print("Feature extraction smoke test complete.")
    elif set(args.models) == set(MODELS):
        summarize_results(args.out_dir)
    else:
        print(
            "Partial model set complete. After all jobs finish, run with "
            "--summarize-only to build the combined table."
        )


if __name__ == "__main__":
    main()
