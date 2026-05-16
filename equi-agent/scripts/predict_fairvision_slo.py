from __future__ import annotations

import argparse
import importlib.util
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


def require_runtime_libs(mirage_dir: Path):
    missing = [
        package
        for package in ["skimage", "einops"]
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise ImportError(
            "Missing MIRAGE dependency/dependencies: "
            f"{', '.join(missing)}. Install with: pip install "
            f"{' '.join('scikit-image' if pkg == 'skimage' else pkg for pkg in missing)}"
        )

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    if not mirage_dir.exists():
        raise FileNotFoundError(f"MIRAGE directory not found: {mirage_dir}")
    sys.path.insert(0, str(mirage_dir))
    from linear_probing_slo import get_model_slo

    return np, pd, torch, Image, transforms, tqdm, get_model_slo


def load_slo_image(np, Image, image_path: str, transform):
    with np.load(image_path) as data:
        fundus_img = data["slo_fundus"]
        if fundus_img.max() <= 1.0:
            fundus_img = (fundus_img * 255).astype("uint8")
        else:
            fundus_img = fundus_img.astype("uint8")
    return transform(Image.fromarray(fundus_img).convert("L"))


def output_probability(torch, outputs, task: str, batch_index: int) -> float:
    if task == "amd":
        # Binary AMD in the manifest means any AMD. SLO head node 0 is P(stage >= early).
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
    parser = argparse.ArgumentParser(description="Generate standard-schema FairVision SLO/MIRAGE predictions.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "manifests",
        help="Directory containing fairvision_amd/dr/glaucoma.csv manifests.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=equi_agent_root() / "weights" / "slo_model_best.pth",
        help="Path to trained SLO/MIRAGE multi-head weights.",
    )
    parser.add_argument(
        "--mirage-dir",
        type=Path,
        default=equi_agent_root() / "VisionAgent" / "MIRAGE",
        help="Path to MIRAGE source/checkpoint directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=equi_agent_root() / "outputs" / "predictions" / "fairvision_slo_mirage.csv",
    )
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu. Defaults to CUDA if available.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np, pd, torch, Image, transforms, tqdm, get_model_slo = require_runtime_libs(args.mirage_dir)

    device = select_device(torch, args.device)
    if not args.weights.exists():
        raise FileNotFoundError(f"SLO/MIRAGE weights not found: {args.weights}")
    if not (args.mirage_dir / "MIRAGE-Base.pth").exists():
        print(f"Warning: MIRAGE-Base.pth not found under {args.mirage_dir}; continuing with get_model_slo().")

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
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    original_dir = Path.cwd()
    os.chdir(args.mirage_dir)
    try:
        model = get_model_slo().to(device)
    finally:
        os.chdir(original_dir)

    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    rows = []
    for start in tqdm(range(0, len(manifest), args.batch_size), desc="FairVision SLO/MIRAGE"):
        batch_df = manifest.iloc[start : start + args.batch_size]
        images = [load_slo_image(np, Image, path, transform) for path in batch_df["image_path"]]
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
                    "model_name": "mirage_slo",
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
