from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


TASKS = ("amd", "dr", "glaucoma")
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
    from torchvision import transforms
    from tqdm import tqdm

    sys.path.insert(0, str(equi_agent_root()))
    from VisionAgent.linear_probing_oct3 import get_model_oct

    return np, pd, torch, Image, transforms, tqdm, get_model_oct


def load_oct_image(np, Image, transforms, image_path: str, transform):
    with np.load(image_path) as data:
        oct_volume = data["oct_bscans"]
        oct_slice = oct_volume[oct_volume.shape[0] // 2]
        if oct_slice.max() <= 1.0:
            oct_slice = (oct_slice * 255).astype("uint8")
        else:
            oct_slice = oct_slice.astype("uint8")
    return transform(Image.fromarray(oct_slice).convert("RGB"))


def output_probability(torch, outputs, task: str, batch_index: int) -> float:
    if task == "amd":
        # Binary AMD in the manifest means any AMD. OCT head node 0 is P(stage >= early).
        return float(torch.sigmoid(outputs["amd"])[batch_index, 0].detach().cpu().item())
    if task == "dr":
        return float(torch.sigmoid(outputs["dr"])[batch_index, 0].detach().cpu().item())
    if task == "glaucoma":
        return float(torch.sigmoid(outputs["glaucoma"])[batch_index, 0].detach().cpu().item())
    raise ValueError(f"Unknown task: {task}")


def select_device(torch, requested: str | None):
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        required_arch = f"sm_{major}{minor}"
        compiled_arches = set(torch.cuda.get_arch_list())
        if compiled_arches and required_arch not in compiled_arches:
            raise RuntimeError(
                f"PyTorch was not compiled for this GPU architecture ({required_arch}). "
                f"Compiled CUDA arches: {sorted(compiled_arches)}. "
                "For a smoke test, rerun with --device cpu. For full experiments, use a "
                "PyTorch/CUDA build or GPU node that supports this architecture."
            )
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standard-schema FairVision OCT predictions.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "manifests",
        help="Directory containing fairvision_amd/dr/glaucoma.csv manifests.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=equi_agent_root() / "weights" / "oct_model_best.pth",
        help="Path to trained OCT multi-head weights.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=equi_agent_root() / "VisionAgent" / "weights" / "RETFound_mae_natureOCT.pth",
        help="Path to RETFound OCT backbone checkpoint used to construct the model.",
    )
    parser.add_argument(
        "--skip-backbone-preload",
        action="store_true",
        help="Use when --weights is a full RETFoundMultiHead state dict containing backbone and heads.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=equi_agent_root() / "outputs" / "predictions" / "fairvision_oct_retfound.csv",
    )
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for future Dataset/DataLoader path.")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu. Defaults to CUDA if available.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backbone_weights.exists():
        os.environ["RETFOUND_OCT_BACKBONE_WEIGHTS"] = str(args.backbone_weights)
    if args.skip_backbone_preload or not args.backbone_weights.exists():
        os.environ["RETFOUND_SKIP_OCT_BACKBONE_PRELOAD"] = "1"
    np, pd, torch, Image, transforms, tqdm, get_model_oct = require_runtime_libs()

    device = select_device(torch, args.device)
    if not args.weights.exists():
        raise FileNotFoundError(f"OCT weights not found: {args.weights}")
    if not args.backbone_weights.exists():
        print(
            "Backbone checkpoint not found; constructing architecture first and "
            "loading --weights as the full model state dict."
        )

    manifests = []
    for task in TASKS:
        path = args.manifest_dir / f"fairvision_{task}.csv"
        df = pd.read_csv(path)
        if args.split != "all":
            df = df[df["split"] == args.split].copy()
        df["task"] = task
        manifests.append(df)
    manifest = pd.concat(manifests, ignore_index=True)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = get_model_oct()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    rows = []
    for start in tqdm(range(0, len(manifest), args.batch_size), desc="FairVision OCT"):
        batch_df = manifest.iloc[start : start + args.batch_size]
        images = [load_oct_image(np, Image, transforms, path, transform) for path in batch_df["image_path"]]
        tensor = torch.stack(images).to(device)

        with torch.no_grad():
            outputs = model(tensor)

        for batch_index, (_, row) in enumerate(batch_df.iterrows()):
            y_prob = output_probability(torch, outputs, row["task"], batch_index)
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

    output = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"wrote={args.out}")
    print(f"rows={len(output)}")
    print(f"by_task={output['task'].value_counts().to_dict()}")
    print(f"by_split={output['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
