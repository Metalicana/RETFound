"""Inspect transferred NPZs offline without loading models or extracting a tar tree."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import tarfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = [
    "OphthalmicAgent/data/loader.py",
    "OphthalmicAgent/VisionAgent/vision_oct_glaucoma.py",
    "OphthalmicAgent/VisionAgent/vision_slo_glaucoma.py",
    "OphthalmicAgent/VisionAgent/linear_probing_oct3.py",
    "OphthalmicAgent/ReplayV2/bundle.json",
    "OphthalmicAgent/outputs/glaucoma_v2_replay/failed_case_results.csv",
]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def archive_case(member):
    p = PurePosixPath(member.name)
    if not member.isfile() or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe archive member: {member.name}")
    if p.parent != PurePosixPath("Datasets/FairVision/Glaucoma/Test") or p.suffix != ".npz":
        raise ValueError(f"Unexpected archive member: {member.name}")
    if not 0 < member.size < 128 * 1024 * 1024:
        raise ValueError(f"Unexpected member size: {member.name}")
    return p.stem


def indices(depth):
    if depth < 8:
        raise ValueError("Expected at least eight slices")
    sampled = np.linspace(0, depth - 1, 8, dtype=int)
    return depth // 2, sampled, sampled[[1, 2, 4, 5]]


def uint8_slice(array):
    return (array * 255).astype(np.uint8) if array.max() <= 1 else array.astype(np.uint8)


def slo_loader(array):
    result = array.astype(np.float32)
    low, high = result.min(), result.max()
    if high > low:
        result = 255 * (result - low) / (high - low)
    return result.astype(np.uint8)


def enhanced(array):
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(array)


def font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def panel(array, title, size=200):
    im = Image.new("RGB", (size, size + 28), "white")
    ImageDraw.Draw(im).text((4, 5), title, fill="black", font=font(14 if size == 200 else 18))
    # Nearest-neighbor enlargement exposes stored pixels without smoothing them.
    im.paste(Image.fromarray(array).convert("RGB").resize((size, size), Image.Resampling.NEAREST), (0, 28))
    return im


def grid(panels, columns, title, footer=""):
    w, h = panels[0].size
    out = Image.new("RGB", (columns * (w + 12) + 12,
                            ((len(panels) + columns - 1) // columns) * (h + 12) + 62), "white")
    draw = ImageDraw.Draw(out)
    draw.text((12, 8), title, fill="black", font=font(21))
    for i, p in enumerate(panels):
        out.paste(p, (12 + i % columns * (w + 12), 38 + i // columns * (h + 12)))
    if footer:
        draw.text((12, out.height - 20), footer, fill="black", font=font(12))
    return out


def array_digest(array):
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


def inspect_case(case_id, payload, expected, out):
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        oct_array = data["oct_bscans"]
        slo = data["slo_fundus"]
        reference = data["glaucoma"]
        keys = sorted(data.files)
    if oct_array.ndim != 3 or slo.ndim != 2:
        raise ValueError(f"Unexpected image dimensions for {case_id}")
    if not np.isfinite(oct_array).all() or not np.isfinite(slo).all():
        raise ValueError(f"Nonfinite input for {case_id}")
    if reference.shape != () or reference.item() not in (0, 1):
        raise ValueError(f"Nonbinary reference for {case_id}")
    middle, sampled, context = indices(oct_array.shape[0])
    slo_prepared = slo_loader(slo)
    oct_mid = uint8_slice(oct_array[middle])
    overview = grid([
        panel(slo_prepared, "SLO: loader scaling"),
        panel(enhanced(slo_prepared), "SLO: current CLAHE"),
        panel(oct_mid, f"OCT: raw slice {middle}"),
        panel(enhanced(oct_mid), f"OCT: CLAHE slice {middle}"),
    ], 2, case_id)
    overview.save(out / "cases" / f"{case_id}_overview.png")
    tiles = [
        panel(slo_prepared, "SLO: loader scaling", 320),
        panel(enhanced(slo_prepared), "SLO: current CLAHE", 320),
        panel(oct_mid, f"OCT: raw slice {middle}", 320),
        panel(enhanced(oct_mid), f"OCT: CLAHE slice {middle}", 320),
    ]
    for i in sampled:
        tiles.append(panel(enhanced(uint8_slice(oct_array[i])), f"OCT: CLAHE slice {i}", 320))
    grid(tiles, 4, case_id + " | actual eight sampled indices",
         "Offline rendering from NPZ; not a recovered historical API image; no reference labels displayed.").save(
             out / "cases" / f"{case_id}_slices.png")
    axis_panels = [panel(enhanced(uint8_slice(np.take(oct_array, oct_array.shape[i] // 2, axis=i))),
                         f"Array axis {i}: central section", 320) for i in range(3)]
    grid(axis_panels, 3, case_id + " | orthogonal array sections",
         "Array axes are not verified anatomical orientation metadata.").save(out / "cases" / f"{case_id}_axes.png")
    parity = all(np.array_equal(
        np.asarray(Image.fromarray(oct_array[i]).convert("RGB")),
        np.asarray(Image.fromarray(uint8_slice(oct_array[i])).convert("RGB"))) for i in sampled)
    row = dict(case_id=case_id, npz_sha256=hashlib.sha256(payload).hexdigest(),
               oct_sha256=array_digest(oct_array), slo_sha256=array_digest(slo),
               oct_shape="x".join(map(str, oct_array.shape)), oct_dtype=str(oct_array.dtype),
               oct_min=int(oct_array.min()), oct_max=int(oct_array.max()),
               oct_zero_fraction=float((oct_array == 0).mean()),
               slo_shape="x".join(map(str, slo.shape)), slo_dtype=str(slo.dtype),
               slo_min=int(slo.min()), slo_max=int(slo.max()),
               slo_zero_fraction=float((slo == 0).mean()), middle_index=middle,
               sampled_indices=" ".join(map(str, sampled)), context_indices=" ".join(map(str, context)),
               current_caption_context="16 32 80 96", current_caption_middle=64,
               pillow_rgb_train_inference_equal=parity, npz_reference=int(reference.item()),
               saved_reference=int(expected["truth"]),
               reference_matches=int(reference.item()) == int(expected["truth"]),
               npz_keys=" ".join(keys))
    return row, overview


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing inspection directory")
    with (ROOT / SOURCE_FILES[-1]).open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {Path(r["case_id"]).stem: r for r in rows}
    if len(expected) != len(rows) or len(rows) != 57:
        raise ValueError("Expected 57 distinct historical errors")
    with tarfile.open(args.archive, "r:gz") as archive:
        members = archive.getmembers()
        names = [archive_case(m) for m in members]
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError("Archive case set does not match the 57 selected errors")
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir()
    (args.output / "contact_sheets").mkdir()
    summaries, previews = [], []
    # Stream the gzip once rather than repeatedly seeking through compressed NPZs.
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            case_id = archive_case(member)
            payload = archive.extractfile(member).read()
            row, preview = inspect_case(case_id, payload, expected[case_id], args.output)
            summaries.append(row)
            previews.append((case_id, preview))
            print(f"Inspected {case_id}: {row['oct_shape']} {row['oct_dtype']}", flush=True)
    summaries.sort(key=lambda r: r["case_id"])
    with (args.output / "image_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    previews.sort(key=lambda x: x[0])
    for start in range(0, len(previews), 6):
        group = previews[start:start + 6]
        w, h = group[0][1].size
        sheet = Image.new("RGB", (w * 3, h * 2), "#dddddd")
        for i, (_, im) in enumerate(group):
            sheet.paste(im, (i % 3 * w, i // 3 * h))
        sheet.save(args.output / "contact_sheets" / f"page_{start // 6 + 1:02}.png")
    provenance = dict(archive=str(args.archive.resolve()), archive_sha256=sha256(args.archive),
                      source_sha256={p: sha256(ROOT / p) for p in SOURCE_FILES},
                      script_sha256=sha256(__file__), python=platform.python_version(),
                      numpy=np.__version__, opencv=cv2.__version__, pillow=PIL.__version__,
                      npz_count=len(summaries), historical_rendered_images_received=0,
                      segmentation_masks_received=0, api_calls=0, model_inference_calls=0,
                      label_mismatches=[r["case_id"] for r in summaries if not r["reference_matches"]],
                      rgb_conversion_mismatches=[r["case_id"] for r in summaries if not r["pillow_rgb_train_inference_equal"]],
                      scope="New offline visualizations from original NPZs; not historical request recovery")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ("npz_count", "label_mismatches", "rgb_conversion_mismatches")}))


if __name__ == "__main__":
    main()
