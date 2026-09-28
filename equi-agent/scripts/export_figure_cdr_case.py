"""Export one real CFP segmentation for an architecture figure, on CECSL.

Uses the exact external-CDR preprocessing and ratio helper already in the repo.
No foundation-model retraining, LLM calls or changes to existing result files.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
from precompute_external_cdr import cdr_from_mask, resolve_path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_measurements(mask):
    if mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 1, 2}):
        raise ValueError("Expected a 2D mask with background=0, disc=1, cup=2")
    vertical, horizontal = cdr_from_mask(mask)
    if vertical is None:
        raise ValueError("Segmentation has no usable disc/cup; do not manufacture a ratio")
    disc, cup = mask > 0, mask == 2
    dy, dx = np.where(disc)
    cy, cx = np.where(cup)
    return dict(vertical_cdr=vertical, horizontal_cdr=horizontal,
                area_cdr=float(cup.sum() / disc.sum()),
                disc_area_pixels=int(disc.sum()), cup_area_pixels=int(cup.sum()),
                disc_height_pixels=int(dy.max()-dy.min()+1),
                cup_height_pixels=int(cy.max()-cy.min()+1),
                disc_bbox_xyxy=[int(dx.min()), int(dy.min()), int(dx.max()+1), int(dy.max()+1)],
                cup_bbox_xyxy=[int(cx.min()), int(cy.min()), int(cx.max()+1), int(cy.max()+1)])


def overlay_mask(image, mask):
    from scipy.ndimage import binary_erosion
    rgb = np.asarray(image.convert("RGB")).copy()
    if rgb.shape[:2] != mask.shape:
        raise ValueError("Mask and image sizes differ")
    # Contours are computed from the actual prediction, not manually drawn anatomy.
    disc, cup = mask > 0, mask == 2
    rgb[disc ^ binary_erosion(disc, iterations=3)] = (0, 195, 220)
    rgb[cup ^ binary_erosion(cup, iterations=3)] = (235, 95, 186)
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "OphthalmicAgent/data_drishti/manifest.csv")
    parser.add_argument("--case-id", default="drishtiGS_053")
    parser.add_argument("--model", default="pamixsun/segformer_for_optic_disc_cup_segmentation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/retinagent_architecture_case"))
    parser.add_argument("--archive", type=Path, default=Path("/tmp/retinagent_architecture_case.tar.gz"))
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("case_id") == args.case_id]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one manifest case {args.case_id}; found {len(rows)}")
    row = rows[0]
    source_path = resolve_path(args.manifest, row["cfp_path"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {args.out_dir}; choose another --out-dir")
    if args.archive.exists():
        raise FileExistsError(f"Archive already exists: {args.archive}; choose another --archive")

    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run this one-case export on CECSL; no CPU fallback was used.")
    image = Image.open(source_path).convert("RGB")
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(args.device).eval()
    # Match OphthalmicAgent/scripts/precompute_external_cdr.py exactly.
    enhanced = ImageOps.equalize(image)
    inputs = processor(images=enhanced, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    measurements = mask_measurements(mask)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.out_dir / "fundus.png")
    enhanced.save(args.out_dir / "model_input_equalized.png")
    Image.fromarray(mask).save(args.out_dir / "mask.png")
    overlay_mask(image, mask).save(args.out_dir / "overlay.png")
    measurements.update(case_id=args.case_id, dataset=row["dataset"], split=row["split"],
                        source_image_sha256=sha256(source_path), source_path=str(source_path),
                        rgb_pixel_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                        model=args.model, preprocessing="RGB -> PIL ImageOps.equalize -> SegformerImageProcessor",
                        postprocessing="bilinear upsample logits to source dimensions, align_corners=False; argmax",
                        class_mapping={"0": "background", "1": "disc excluding cup", "2": "cup"},
                        inference_device=args.device, torch_version=torch.__version__,
                        model_commit=getattr(model.config, "_commit_hash", None),
                        clinically_verified=False,
                        note="New one-case tool export. Compare with saved CDR before using it as the historical mask.")
    measurements["files"] = {p.name: sha256(p) for p in args.out_dir.glob("*.png")}
    (args.out_dir / "measurements.json").write_text(json.dumps(measurements, indent=2)+"\n")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        for name in (*measurements["files"], "measurements.json"):
            archive.add(args.out_dir / name, arcname=f"retinagent_architecture_case/{name}", recursive=False)
    print(json.dumps({k: measurements[k] for k in ("case_id", "vertical_cdr", "horizontal_cdr", "area_cdr")}, indent=2))
    print(f"archive={args.archive}")


if __name__ == "__main__":
    main()
