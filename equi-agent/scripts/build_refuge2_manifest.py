from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def resolve_refuge2_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root
    nested = root / "REFUGE2"
    if (nested / "train").is_dir() and (nested / "val").is_dir():
        return nested
    raise FileNotFoundError(f"Could not find REFUGE2 split folders under {root}")


def infer_label(stem: str) -> tuple[int | None, str]:
    name = stem.lower()
    if name.startswith("g"):
        return 1, "filename_prefix_g"
    if name.startswith("n"):
        return 0, "filename_prefix_n"
    return None, "unknown_filename_prefix"


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)
    except Exception:
        return None, None


def mask_features(path: Path) -> dict[str, Any]:
    """Compute simple cup/disc features from REFUGE-style masks.

    REFUGE masks are commonly encoded as background, optic-disc, and optic-cup
    label values. Some mirrors use 255 as background, 128 as disc/rim, and 0 as
    cup, so treating nonzero as foreground is wrong. This function infers the
    background as the largest pixel class, treats every other substantial class
    as optic-disc foreground, and treats the smaller substantial foreground
    class as cup.
    """
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"))
        values, counts = np.unique(arr, return_counts=True)
        pairs = sorted((int(v), int(c)) for v, c in zip(values, counts))
        total = int(arr.size)
        background_value, _ = max(pairs, key=lambda item: item[1])
        min_substantial_count = max(100, int(total * 0.0001))
        foreground = [(v, c) for v, c in pairs if v != background_value]
        substantial_foreground = [(v, c) for v, c in foreground if c >= min_substantial_count]
        out: dict[str, Any] = {
            "mask_unique_values": ";".join(str(v) for v, _ in pairs),
            "mask_unique_counts": ";".join(f"{v}:{c}" for v, c in pairs),
            "mask_foreground_levels": len(foreground),
            "mask_background_value": background_value,
        }
        if not substantial_foreground:
            out.update(
                {
                    "disc_area_px": "",
                    "cup_area_px": "",
                    "cup_to_disc_area_ratio": "",
                    "vertical_cup_to_disc_ratio": "",
                    "mask_feature_rule": "no_substantial_foreground",
                    "cup_mask_value": "",
                }
            )
            return out

        foreground_values = [v for v, _ in substantial_foreground]
        disc_mask = np.isin(arr, foreground_values)
        if len(substantial_foreground) >= 2:
            cup_value, _ = min(substantial_foreground, key=lambda item: item[1])
            cup_mask = arr == cup_value
            rule = "background_largest_cup_smaller_substantial_foreground"
        else:
            cup_value = foreground_values[0]
            cup_mask = arr == cup_value
            rule = "single_substantial_foreground_level"

        disc_area = int(disc_mask.sum())
        cup_area = int(cup_mask.sum())

        def vertical_extent(mask: Any) -> int:
            rows = np.where(mask.any(axis=1))[0]
            if rows.size == 0:
                return 0
            return int(rows.max() - rows.min() + 1)

        disc_vertical = vertical_extent(disc_mask)
        cup_vertical = vertical_extent(cup_mask)
        out.update(
            {
                "disc_area_px": disc_area,
                "cup_area_px": cup_area,
                "cup_to_disc_area_ratio": round(cup_area / disc_area, 6) if disc_area else "",
                "vertical_cup_to_disc_ratio": round(cup_vertical / disc_vertical, 6) if disc_vertical else "",
                "mask_feature_rule": rule,
                "cup_mask_value": cup_value,
            }
        )
        return out
    except Exception as exc:
        return {
            "mask_unique_values": "",
            "mask_unique_counts": "",
            "mask_foreground_levels": "",
            "disc_area_px": "",
            "cup_area_px": "",
            "cup_to_disc_area_ratio": "",
            "vertical_cup_to_disc_ratio": "",
            "mask_feature_rule": f"error:{type(exc).__name__}",
            "mask_background_value": "",
            "cup_mask_value": "",
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "image_id",
        "label_glaucoma",
        "label_source",
        "image_path",
        "mask_path",
        "image_width",
        "image_height",
        "mask_width",
        "mask_height",
        "disc_area_px",
        "cup_area_px",
        "cup_to_disc_area_ratio",
        "vertical_cup_to_disc_ratio",
        "mask_unique_values",
        "mask_unique_counts",
        "mask_foreground_levels",
        "mask_background_value",
        "mask_feature_rule",
        "cup_mask_value",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a REFUGE2 glaucoma manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=Path("equi-agent/outputs/manifests/refuge2_manifest.csv"))
    parser.add_argument("--skip-mask-features", action="store_true")
    args = parser.parse_args()

    root = resolve_refuge2_root(args.root)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        image_dir = root / split / "images"
        mask_dir = root / split / "mask"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS):
            label, label_source = infer_label(image_path.stem)
            mask_path = next((mask_dir / f"{image_path.stem}{ext}" for ext in [".bmp", ".png", ".jpg"] if (mask_dir / f"{image_path.stem}{ext}").exists()), None)
            image_width, image_height = image_size(image_path)
            mask_width, mask_height = image_size(mask_path) if mask_path else (None, None)
            row: dict[str, Any] = {
                "dataset": "REFUGE2",
                "split": split,
                "image_id": image_path.stem,
                "label_glaucoma": "" if label is None else label,
                "label_source": label_source,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else "",
                "image_width": image_width or "",
                "image_height": image_height or "",
                "mask_width": mask_width or "",
                "mask_height": mask_height or "",
            }
            if mask_path and not args.skip_mask_features:
                row.update(mask_features(mask_path))
            rows.append(row)

    write_csv(args.out_csv, rows)
    by_split: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row["split"])
        by_split.setdefault(split, {"rows": 0, "positive": 0, "negative": 0, "missing_label": 0, "missing_mask": 0})
        by_split[split]["rows"] += 1
        if row["label_glaucoma"] == 1:
            by_split[split]["positive"] += 1
        elif row["label_glaucoma"] == 0:
            by_split[split]["negative"] += 1
        else:
            by_split[split]["missing_label"] += 1
        if not row["mask_path"]:
            by_split[split]["missing_mask"] += 1

    print(f"root={root}")
    print(f"wrote={args.out_csv}")
    print(f"rows={len(rows)}")
    for split in SPLITS:
        if split in by_split:
            print(f"{split}: {by_split[split]}")


if __name__ == "__main__":
    main()
