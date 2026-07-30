#!/usr/bin/env python3
"""Precompute SegFormer vertical CDR for canonical CFP manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="pamixsun/segformer_for_optic_disc_cup_segmentation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = manifest.parent.parent / path
    return candidate if candidate.exists() else manifest.parent / path


def cdr_from_mask(mask: np.ndarray) -> tuple[float | None, float | None]:
    disc = (mask == 1) | (mask == 2)
    cup = mask == 2
    if not disc.any() or not cup.any():
        return None, None
    disc_y, disc_x = np.where(disc)
    cup_y, cup_x = np.where(cup)
    disc_height = int(disc_y.max() - disc_y.min() + 1)
    cup_height = int(cup_y.max() - cup_y.min() + 1)
    disc_width = int(disc_x.max() - disc_x.min() + 1)
    cup_width = int(cup_x.max() - cup_x.min() + 1)
    return cup_height / disc_height, cup_width / disc_width


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image, ImageOps
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("cfp_path", "").strip()]
    if args.limit:
        rows = rows[: args.limit]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    output_rows = []
    for index, row in enumerate(rows, start=1):
        path = resolve_path(args.manifest, row["cfp_path"])
        image = ImageOps.equalize(Image.open(path).convert("RGB"))
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        mask = logits.argmax(dim=1)[0].cpu().numpy()
        vertical, horizontal = cdr_from_mask(mask)
        output_rows.append({
            "dataset": row["dataset"], "case_id": row["case_id"], "split": row["split"],
            "label": row["label"], "cfp_path": row["cfp_path"],
            "vertical_cdr": "" if vertical is None else round(vertical, 6),
            "horizontal_cdr": "" if horizontal is None else round(horizontal, 6),
            "cdr_available": vertical is not None, "cdr_model": args.model,
        })
        print(f"cdr {index}/{len(rows)} case={row['case_id']} vertical={vertical}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]) if output_rows else ["dataset", "case_id"])
        writer.writeheader(); writer.writerows(output_rows)
    print(f"wrote={args.out} rows={len(output_rows)}")


if __name__ == "__main__":
    main()
