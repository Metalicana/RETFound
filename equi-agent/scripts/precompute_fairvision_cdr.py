from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


TASKS = ["glaucoma"]
METHOD = "zero_shot_brightness_structural_cdr_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def resolve_path(path_value: str, prefix_from: str | None, prefix_to: str | None) -> Path:
    path_text = str(path_value or "")
    path = Path(path_text)
    if path.exists():
        return path
    if prefix_from and prefix_to and path_text.startswith(prefix_from):
        return Path(prefix_to + path_text[len(prefix_from) :])
    return path


def normalize_to_unit(array: Any) -> Any:
    import numpy as np

    image = np.asarray(array)
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        image = image[..., :3]
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    elif image.ndim == 3 and image.shape[0] in {3, 4}:
        image = np.moveaxis(image[:3], 0, -1)
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    elif image.ndim == 3:
        slice_axis = min(range(3), key=lambda axis: image.shape[axis])
        image = np.take(image, image.shape[slice_axis] // 2, axis=slice_axis)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D SLO/fundus image after normalization, got shape={image.shape}")

    image = image.astype("float32")
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype="float32")
    min_value = float(image[finite].min())
    max_value = float(image[finite].max())
    if max_value <= 1.0 and min_value >= 0.0:
        out = image.copy()
    elif max_value > min_value:
        out = (image - min_value) / (max_value - min_value)
    else:
        out = np.zeros(image.shape, dtype="float32")
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def disk_structure(radius: int) -> Any:
    import numpy as np

    radius = max(1, int(radius))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def remove_small_components(mask: Any, min_size: int) -> Any:
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(mask.astype(bool))
    if count == 0:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(1, int(min_size))
    keep[0] = False
    return keep[labels]


def clean_mask(mask: Any, min_size: int, radius: int) -> Any:
    from scipy import ndimage

    cleaned = remove_small_components(mask.astype(bool), min_size=max(1, int(min_size)))
    cleaned = ndimage.binary_closing(cleaned, structure=disk_structure(radius))
    cleaned = ndimage.binary_fill_holes(cleaned)
    cleaned = remove_small_components(cleaned, min_size=max(1, int(min_size)))
    cleaned = ndimage.binary_opening(cleaned, structure=disk_structure(1))
    return cleaned


def vertical_extent(mask: Any) -> int:
    import numpy as np

    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0
    return int(rows.max() - rows.min() + 1)


def mask_bbox_touches_border(region: Any, shape: tuple[int, int], margin: int = 2) -> bool:
    min_row, min_col, max_row, max_col = region
    height, width = shape
    return min_row <= margin or min_col <= margin or max_row >= height - margin or max_col >= width - margin


def choose_region(mask: Any, intensity: Any, point_yx: tuple[float, float]) -> Any | None:
    import math
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(mask.astype(bool))
    if count == 0:
        return None
    height, width = mask.shape
    diag = math.hypot(height, width)
    objects = ndimage.find_objects(labels)
    best_label = None
    best_score = -1e9
    point_y, point_x = point_yx
    for label_value, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        region_mask = labels[slices] == label_value
        area = int(region_mask.sum())
        if area == 0:
            continue
        min_row, max_row = slices[0].start, slices[0].stop
        min_col, max_col = slices[1].start, slices[1].stop
        bbox_area = max(1, (max_row - min_row) * (max_col - min_col))
        compactness = min(1.0, float(area) / bbox_area)
        coords = np.argwhere(region_mask)
        centroid_y = float(coords[:, 0].mean() + min_row)
        centroid_x = float(coords[:, 1].mean() + min_col)
        distance = math.hypot(centroid_y - point_y, centroid_x - point_x) / max(diag, 1.0)
        mean_intensity = float(intensity[slices][region_mask].mean())
        border_penalty = 0.35 if mask_bbox_touches_border((min_row, min_col, max_row, max_col), mask.shape) else 0.0
        score = 2.0 * mean_intensity + compactness + 1.5 * (1.0 - distance) - border_penalty
        if score > best_score:
            best_score = score
            best_label = label_value
    if best_label is None:
        return None
    return labels == best_label


def brightest_seed(gray: Any, seed_percentile: float) -> tuple[float, float, list[str]]:
    import numpy as np
    from scipy import ndimage

    reasons: list[str] = []
    smooth = ndimage.gaussian_filter(gray, sigma=2.0)
    threshold = float(np.percentile(smooth, seed_percentile))
    seed_mask = clean_mask(smooth >= threshold, min_size=max(8, int(gray.size * 0.00005)), radius=2)
    region_mask = choose_region(seed_mask, smooth, tuple(float(x) for x in np.unravel_index(int(np.argmax(smooth)), smooth.shape)))
    if region_mask is None:
        y, x = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
        reasons.append("seed_from_global_brightest_pixel")
        return float(y), float(x), reasons
    ys, xs = np.where(region_mask)
    reasons.append("seed_from_bright_component")
    return float(ys.mean()), float(xs.mean()), reasons


def contrast_enhance(gray: Any) -> Any:
    import numpy as np

    low, high = np.percentile(gray, [1.0, 99.0])
    if high <= low:
        return gray
    return np.clip((gray - low) / (high - low), 0.0, 1.0)


def crop_bounds(center_y: float, center_x: float, shape: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    height, width = shape
    y0 = max(0, int(round(center_y)) - radius)
    y1 = min(height, int(round(center_y)) + radius + 1)
    x0 = max(0, int(round(center_x)) - radius)
    x1 = min(width, int(round(center_x)) + radius + 1)
    return y0, y1, x0, x1


def percentile_candidates(base: float) -> list[float]:
    values = [base, base - 5, base + 5, base - 10, base + 10, base - 15]
    return [min(95.0, max(35.0, value)) for value in values]


def estimate_cdr(
    image_array: Any,
    sensitivity_threshold: float,
    precision_threshold: float,
    seed_percentile: float,
    disc_percentile: float,
    cup_percentile: float,
    roi_radius_frac: float,
    min_quality: float,
) -> dict[str, Any]:
    import numpy as np
    from scipy import ndimage

    gray = normalize_to_unit(image_array)
    if gray.size == 0:
        raise ValueError("empty image")
    enhanced = contrast_enhance(gray)
    smooth = ndimage.gaussian_filter(enhanced, sigma=1.25)
    height, width = smooth.shape
    total_area = float(height * width)
    reasons: list[str] = []

    seed_y, seed_x, seed_reasons = brightest_seed(smooth, seed_percentile)
    reasons.extend(seed_reasons)
    radius = max(24, int(round(min(height, width) * roi_radius_frac)))
    y0, y1, x0, x1 = crop_bounds(seed_y, seed_x, smooth.shape, radius)
    roi = smooth[y0:y1, x0:x1]
    local_seed = (seed_y - y0, seed_x - x0)

    disc_mask_local = None
    disc_percentile_used = None
    disc_area = 0
    min_disc_area = max(16, int(total_area * 0.002))
    max_disc_area = max(min_disc_area + 1, int(total_area * 0.30))
    for percentile in percentile_candidates(disc_percentile):
        threshold = float(np.percentile(roi, percentile))
        candidate = clean_mask(roi >= threshold, min_size=max(8, int(total_area * 0.0003)), radius=3)
        selected = choose_region(candidate, roi, local_seed)
        if selected is None:
            continue
        area = int(selected.sum())
        if min_disc_area <= area <= max_disc_area:
            disc_mask_local = selected
            disc_percentile_used = percentile
            disc_area = area
            break
    if disc_mask_local is None:
        reasons.append("disc_component_not_found")
        return unavailable_result(reasons, min_quality)

    disc_pixels = roi[disc_mask_local]
    if disc_pixels.size == 0:
        reasons.append("disc_pixels_empty")
        return unavailable_result(reasons, min_quality)

    cup_mask_local = None
    cup_percentile_used = None
    cup_area = 0
    disc_centroid = tuple(float(x) for x in np.argwhere(disc_mask_local).mean(axis=0))
    for percentile in percentile_candidates(cup_percentile):
        threshold = float(np.percentile(disc_pixels, percentile))
        candidate = clean_mask(disc_mask_local & (roi >= threshold), min_size=max(4, int(disc_area * 0.02)), radius=2)
        selected = choose_region(candidate, roi, disc_centroid)
        if selected is None:
            continue
        area = int(selected.sum())
        if 0 < area < int(disc_area * 0.95):
            cup_mask_local = selected
            cup_percentile_used = percentile
            cup_area = area
            break
    if cup_mask_local is None:
        cup_mask_local = np.zeros_like(disc_mask_local, dtype=bool)
        reasons.append("cup_component_not_found")

    disc_vertical = vertical_extent(disc_mask_local)
    cup_vertical = vertical_extent(cup_mask_local)
    vcdr = cup_vertical / disc_vertical if disc_vertical else None
    area_cdr = cup_area / disc_area if disc_area else None

    disc_background = roi[~disc_mask_local]
    contrast = float(disc_pixels.mean() - disc_background.mean()) if disc_background.size else 0.0
    disc_area_frac = disc_area / total_area
    cup_area_frac = cup_area / total_area
    touches_border = bool(y0 == 0 or x0 == 0 or y1 == height or x1 == width)
    plausible_vcdr = vcdr is not None and 0.05 <= vcdr <= 0.98
    plausible_area = 0.002 <= disc_area_frac <= 0.30 and (area_cdr is None or 0.0 <= area_cdr <= 0.95)

    quality = 0.0
    quality += 0.30
    quality += 0.20 if cup_area > 0 else 0.0
    quality += 0.15 if plausible_area else 0.0
    quality += 0.15 if plausible_vcdr else 0.0
    quality += 0.10 if contrast >= 0.03 else max(0.0, contrast / 0.03) * 0.10
    quality += 0.10 if not touches_border else 0.03
    quality = round(min(1.0, quality), 4)

    if disc_area < min_disc_area:
        reasons.append("disc_area_too_small")
    if disc_area > max_disc_area:
        reasons.append("disc_area_too_large")
    if touches_border:
        reasons.append("disc_roi_touches_image_border")
    if contrast < 0.03:
        reasons.append("low_disc_contrast")
    if quality < min_quality:
        reasons.append("low_segmentation_quality")

    zone = cdr_zone(vcdr, sensitivity_threshold, precision_threshold)
    confidence = measurement_confidence(quality, zone)
    return {
        "cdr_available": int(vcdr is not None and quality >= min_quality),
        "method": METHOD,
        "vertical_cup_to_disc_ratio": round(vcdr, 6) if vcdr is not None else "",
        "cup_to_disc_area_ratio": round(area_cdr, 6) if area_cdr is not None else "",
        "disc_area_px": disc_area,
        "cup_area_px": cup_area,
        "disc_vertical_px": disc_vertical,
        "cup_vertical_px": cup_vertical,
        "disc_area_frac": round(disc_area_frac, 6),
        "cup_area_frac": round(cup_area_frac, 6),
        "disc_center_y": round(y0 + disc_centroid[0], 3),
        "disc_center_x": round(x0 + disc_centroid[1], 3),
        "disc_percentile_used": disc_percentile_used if disc_percentile_used is not None else "",
        "cup_percentile_used": cup_percentile_used if cup_percentile_used is not None else "",
        "segmentation_quality_score": quality,
        "measurement_confidence": confidence,
        "cdr_zone": zone,
        "structural_escalate_to_human": int(zone == "borderline" or quality < min_quality),
        "structural_agent_summary": structural_summary(vcdr, zone, quality, min_quality),
        "quality_reasons": ";".join(reasons),
        "_debug_gray": gray,
        "_debug_disc_mask": paste_local_mask(disc_mask_local, smooth.shape, y0, x0),
        "_debug_cup_mask": paste_local_mask(cup_mask_local, smooth.shape, y0, x0),
    }


def paste_local_mask(local_mask: Any, shape: tuple[int, int], y0: int, x0: int) -> Any:
    import numpy as np

    out = np.zeros(shape, dtype=bool)
    height, width = local_mask.shape
    out[y0 : y0 + height, x0 : x0 + width] = local_mask
    return out


def unavailable_result(reasons: list[str], min_quality: float) -> dict[str, Any]:
    return {
        "cdr_available": 0,
        "method": METHOD,
        "vertical_cup_to_disc_ratio": "",
        "cup_to_disc_area_ratio": "",
        "disc_area_px": "",
        "cup_area_px": "",
        "disc_vertical_px": "",
        "cup_vertical_px": "",
        "disc_area_frac": "",
        "cup_area_frac": "",
        "disc_center_y": "",
        "disc_center_x": "",
        "disc_percentile_used": "",
        "cup_percentile_used": "",
        "segmentation_quality_score": 0.0,
        "measurement_confidence": "unavailable",
        "cdr_zone": "unavailable",
        "structural_escalate_to_human": 1,
        "structural_agent_summary": f"CDR unavailable from zero-shot SLO segmentation; quality threshold {min_quality:.2f} not met",
        "quality_reasons": ";".join(reasons),
    }


def cdr_zone(vcdr: float | None, sensitivity_threshold: float, precision_threshold: float) -> str:
    if vcdr is None:
        return "unavailable"
    if vcdr >= precision_threshold:
        return "strong"
    if vcdr >= sensitivity_threshold:
        return "borderline"
    return "weak_or_negative"


def measurement_confidence(quality: float, zone: str) -> str:
    if zone == "unavailable":
        return "unavailable"
    if quality >= 0.80 and zone in {"strong", "weak_or_negative"}:
        return "medium"
    if quality >= 0.65:
        return "low"
    return "low"


def structural_summary(vcdr: float | None, zone: str, quality: float, min_quality: float) -> str:
    if vcdr is None:
        return "CDR unavailable from zero-shot SLO segmentation"
    if quality < min_quality:
        return f"low-quality zero-shot CDR estimate vCDR={vcdr:.3f}; do not use as decisive evidence"
    if zone == "strong":
        return f"strong structural glaucoma evidence from estimated vCDR={vcdr:.3f}"
    if zone == "borderline":
        return f"borderline structural glaucoma evidence from estimated vCDR={vcdr:.3f}; human review recommended"
    return f"weak structural glaucoma evidence from estimated vCDR={vcdr:.3f}"


def save_debug_overlay(path: Path, gray: Any, disc_mask: Any, cup_mask: Any) -> None:
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    base = np.stack([gray, gray, gray], axis=-1)
    disc_boundary = disc_mask.astype(bool) ^ ndimage.binary_erosion(disc_mask.astype(bool))
    cup_boundary = cup_mask.astype(bool) ^ ndimage.binary_erosion(cup_mask.astype(bool))
    base[disc_boundary] = [0.0, 1.0, 0.0]
    base[cup_boundary] = [1.0, 0.0, 0.0]
    image = Image.fromarray((np.clip(base, 0.0, 1.0) * 255).astype("uint8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def manifest_file(manifests_root: Path, task: str) -> Path:
    return manifests_root / f"fairvision_{task}.csv"


def load_slo_array(npz_path: Path, key: str) -> Any:
    import numpy as np

    with np.load(npz_path) as data:
        if key not in data:
            raise KeyError(f"missing NPZ key {key!r}; available={list(data.files)}")
        return data[key]


def output_fieldnames() -> list[str]:
    return [
        "patient_id",
        "eye_id",
        "visit_id",
        "image_id",
        "dataset",
        "task",
        "split",
        "y_true",
        "image_path",
        "fundus_key",
        "cdr_available",
        "method",
        "vertical_cup_to_disc_ratio",
        "cup_to_disc_area_ratio",
        "disc_area_px",
        "cup_area_px",
        "disc_vertical_px",
        "cup_vertical_px",
        "disc_area_frac",
        "cup_area_frac",
        "disc_center_y",
        "disc_center_x",
        "disc_percentile_used",
        "cup_percentile_used",
        "segmentation_quality_score",
        "measurement_confidence",
        "cdr_zone",
        "structural_escalate_to_human",
        "structural_agent_summary",
        "quality_reasons",
        "error",
    ]


def clean_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_debug_")}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if str(row.get("cdr_available", "")) == "1"]
    vcdr_values = [
        float(row["vertical_cup_to_disc_ratio"])
        for row in available
        if str(row.get("vertical_cup_to_disc_ratio", "")).strip()
    ]
    zones: dict[str, int] = {}
    for row in rows:
        zone = str(row.get("cdr_zone", ""))
        zones[zone] = zones.get(zone, 0) + 1
    return {
        "rows": len(rows),
        "available": len(available),
        "availability_rate": round(len(available) / len(rows), 6) if rows else 0.0,
        "cdr_zone_counts": zones,
        "vcdr_mean": round(mean(vcdr_values), 6) if vcdr_values else None,
        "vcdr_min": round(min(vcdr_values), 6) if vcdr_values else None,
        "vcdr_max": round(max(vcdr_values), 6) if vcdr_values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute zero-training optic cup/disc ratio estimates from FairVision SLO/fundus images. "
            "This is a structural evidence side-channel for glaucoma arbitration; it does not use MD or labels."
        )
    )
    parser.add_argument("--manifests-root", type=Path, default=Path("equi-agent/outputs/manifests"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--out-csv", type=Path, default=Path("equi-agent/outputs/structural/fairvision_cdr_zero_shot.csv"))
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument("--fundus-key", default="", help="Override manifest fundus_key; default uses row value or slo_fundus.")
    parser.add_argument("--max-cases", type=int, default=0, help="Use <=0 for all rows.")
    parser.add_argument("--sensitivity-threshold", type=float, default=0.55)
    parser.add_argument("--precision-threshold", type=float, default=0.62)
    parser.add_argument("--seed-percentile", type=float, default=98.0)
    parser.add_argument("--disc-percentile", type=float, default=62.0)
    parser.add_argument("--cup-percentile", type=float, default=78.0)
    parser.add_argument("--roi-radius-frac", type=float, default=0.20)
    parser.add_argument("--min-quality", type=float, default=0.55)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--debug-limit", type=int, default=0)
    args = parser.parse_args()

    rows_out: list[dict[str, Any]] = []
    debug_written = 0
    for task in args.tasks:
        manifest = manifest_file(args.manifests_root, task)
        if not manifest.exists():
            raise FileNotFoundError(f"Missing manifest for task={task}: {manifest}")
        rows = read_csv(manifest)
        if args.max_cases > 0:
            rows = rows[: args.max_cases]
        for row in rows:
            key = args.fundus_key or row.get("fundus_key") or "slo_fundus"
            npz_path = resolve_path(row.get("image_path", ""), args.path_prefix_from, args.path_prefix_to)
            base = {
                "patient_id": row.get("patient_id", ""),
                "eye_id": row.get("eye_id", ""),
                "visit_id": row.get("visit_id", ""),
                "image_id": row.get("image_id", ""),
                "dataset": row.get("dataset", ""),
                "task": row.get("task", task),
                "split": row.get("split", ""),
                "y_true": row.get("y_true", ""),
                "image_path": str(npz_path),
                "fundus_key": key,
            }
            try:
                if not npz_path.exists():
                    raise FileNotFoundError(str(npz_path))
                estimate = estimate_cdr(
                    load_slo_array(npz_path, key),
                    sensitivity_threshold=args.sensitivity_threshold,
                    precision_threshold=args.precision_threshold,
                    seed_percentile=args.seed_percentile,
                    disc_percentile=args.disc_percentile,
                    cup_percentile=args.cup_percentile,
                    roi_radius_frac=args.roi_radius_frac,
                    min_quality=args.min_quality,
                )
                if args.debug_dir and args.debug_limit > debug_written:
                    save_debug_overlay(
                        args.debug_dir / f"{task}_{row.get('image_id', '')}.png",
                        estimate["_debug_gray"],
                        estimate["_debug_disc_mask"],
                        estimate["_debug_cup_mask"],
                    )
                    debug_written += 1
                rows_out.append({**base, **clean_output_row(estimate), "error": ""})
            except Exception as exc:
                rows_out.append(
                    {
                        **base,
                        **unavailable_result([f"error:{type(exc).__name__}"], args.min_quality),
                        "error": str(exc),
                    }
                )

    fieldnames = output_fieldnames()
    write_csv(args.out_csv, rows_out, fieldnames)
    summary = {
        "method": METHOD,
        "out_csv": str(args.out_csv),
        "tasks": args.tasks,
        "sensitivity_threshold": args.sensitivity_threshold,
        "precision_threshold": args.precision_threshold,
        **summarize(rows_out),
    }
    if args.out_summary:
        write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
