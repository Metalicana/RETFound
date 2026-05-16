from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FAIRVISION_DISEASES = ("AMD", "DR", "Glaucoma")
FAIRVISION_SPLITS = ("Training", "Validation", "Test")
GDP_MODALITIES = ("Bscan", "RNFLT")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def summarize_array(value: np.ndarray) -> str:
    arr = np.asarray(value)
    summary = f"shape={arr.shape}, dtype={arr.dtype}"
    if arr.size == 0:
        return f"{summary}, empty=True"

    if arr.ndim == 0:
        return f"{summary}, scalar={arr.item()!r}"

    flat = arr.reshape(-1)
    preview_count = min(6, flat.size)
    preview = flat[:preview_count].tolist()
    numeric = np.issubdtype(arr.dtype, np.number)
    if numeric:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            summary += f", min={finite.min():.6g}, max={finite.max():.6g}"
    return f"{summary}, preview={preview}"


def print_npz_summary(path: Path, indent: str = "  ") -> None:
    print(f"{indent}NPZ: {path}")
    if not path.exists():
        print(f"{indent}  MISSING")
        return

    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        print(f"{indent}  keys={keys}")
        for key in keys:
            try:
                print(f"{indent}  - {key}: {summarize_array(data[key])}")
            except Exception as exc:
                print(f"{indent}  - {key}: ERROR reading key: {exc}")


def head_records(csv_path: Path, n: int) -> list[dict]:
    if not csv_path.exists():
        print(f"  Metadata CSV missing: {csv_path}")
        return []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = []
        total = 0
        for row in reader:
            total += 1
            if len(rows) < n:
                rows.append(row)

    print(f"  Metadata CSV: {csv_path}")
    print(f"  rows={total}, columns={columns}")
    if rows:
        print(f"  first_row={rows[0]}")
    return rows


def fairvision_summary(fairvision_root: Path, max_per_group: int) -> None:
    print("\n=== FairVision ===")
    print(f"root={fairvision_root}")

    for disease in FAIRVISION_DISEASES:
        csv_name = f"data_summary_{disease.lower()}.csv"
        csv_path = fairvision_root / "HarvardFairVision30k" / disease / "ReadMe" / csv_name
        print(f"\n[{disease}]")
        rows = head_records(csv_path, max_per_group)
        for split in FAIRVISION_SPLITS:
            split_key = split.lower()
            split_rows = [row for row in rows if str(row.get("use", "")).lower() == split_key]
            if not split_rows:
                continue
            print(f"  Split {split}:")
            for row in split_rows[:max_per_group]:
                npz_path = fairvision_root / split / str(row["filename"])
                print(f"    row_label={row.get(disease.lower())!r}, filename={row['filename']}")
                print_npz_summary(npz_path, indent="    ")

    print("\n  Direct split-folder sample:")
    for split in FAIRVISION_SPLITS:
        sample = first_existing_npz(fairvision_root / split)
        if sample:
            print_npz_summary(sample, indent=f"  {split} ")
        else:
            print(f"  {split}: no .npz files found")


def gdp_summary(gdp_root: Path, max_per_group: int) -> None:
    print("\n=== GDP ===")
    print(f"root={gdp_root}")
    csv_path = gdp_root / "ReadMe" / "data_summary.csv"
    rows = head_records(csv_path, max_per_group)

    for modality in GDP_MODALITIES:
        print(f"\n[{modality}]")
        for row in rows[:max_per_group]:
            filename = str(row["filename"])
            if not filename.endswith(".npz"):
                filename = f"{filename}.npz"
            print(
                "  "
                f"filename={filename}, glaucoma={row.get('glaucoma')!r}, "
                f"detection_use={row.get('glaucoma_detection_use')!r}, "
                f"progression_use={row.get('progression_forecasting_use')!r}"
            )
            print_npz_summary(gdp_root / modality / filename, indent="  ")

        sample = first_existing_npz(gdp_root / modality)
        if sample:
            print(f"  First file by directory listing:")
            print_npz_summary(sample, indent="  ")


def first_existing_npz(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    return next(iter(sorted(directory.glob("*.npz"))), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print metadata, NPZ keys, shapes, dtypes, and value previews for FairVision and GDP."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=repo_root() / "Datasets",
        help="Path to RETFound/Datasets. Defaults to ../Datasets relative to this script.",
    )
    parser.add_argument("--max-per-group", type=int, default=1, help="Number of rows/files to inspect per group.")
    parser.add_argument(
        "--dataset",
        choices=("all", "fairvision", "gdp"),
        default="all",
        help="Which dataset to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_root = args.datasets_root.expanduser().resolve()
    print(f"datasets_root={datasets_root}")
    print(f"max_per_group={args.max_per_group}")

    if args.dataset in {"all", "fairvision"}:
        fairvision_summary(datasets_root / "FairVision", args.max_per_group)
    if args.dataset in {"all", "gdp"}:
        gdp_summary(datasets_root / "GDP", args.max_per_group)


if __name__ == "__main__":
    main()
