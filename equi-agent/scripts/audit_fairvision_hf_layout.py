from __future__ import annotations

import argparse
from pathlib import Path


TASKS = {
    "amd": ("AMD", "data_summary_amd.csv", "amd"),
    "dr": ("DR", "data_summary_dr.csv", "dr"),
    "glaucoma": ("Glaucoma", "data_summary_glaucoma.csv", "glaucoma"),
}


def metadata_path(root: Path, source: str, csv_name: str) -> Path:
    candidates = [
        root / "HarvardFairVision30k" / source / "ReadMe" / csv_name,
        root / csv_name,
        root / source / csv_name,
        root / source / "ReadMe" / csv_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {csv_name}. Checked: {candidates}")


def split_folder(root: Path, split: str, source: str) -> Path:
    folder_name = {"train": "Training", "val": "Validation", "test": "Test"}[split]
    candidates = [root / folder_name, root / folder_name / source, root / source / folder_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find split={split}. Checked: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HuggingFace-style Harvard-FairVision files and labels.")
    parser.add_argument("--root", type=Path, default=Path("Datasets/FairVision"))
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=["amd", "dr"])
    parser.add_argument("--samples-per-split", type=int, default=6)
    return parser.parse_args()


def describe_array(np, name: str, arr):
    arr = np.asarray(arr)
    return {
        f"{name}_shape": "x".join(str(x) for x in arr.shape),
        f"{name}_dtype": str(arr.dtype),
        f"{name}_min": float(np.nanmin(arr)) if arr.size else None,
        f"{name}_max": float(np.nanmax(arr)) if arr.size else None,
        f"{name}_mean": float(np.nanmean(arr)) if arr.size else None,
        f"{name}_std": float(np.nanstd(arr)) if arr.size else None,
    }


def main() -> None:
    import numpy as np
    import pandas as pd

    args = parse_args()
    for task in args.tasks:
        source, csv_name, label_col = TASKS[task]
        meta = pd.read_csv(metadata_path(args.root, source, csv_name))
        print(f"\n=== task={task} source={source} rows={len(meta)} ===")
        print("columns=" + ",".join(meta.columns.astype(str)))
        print("use counts:")
        print(meta["use"].value_counts(dropna=False).to_string())
        print("label counts by split:")
        print(pd.crosstab(meta["use"], meta[label_col], dropna=False).to_string())

        for split in ["train", "val", "test"]:
            folder = split_folder(args.root, split, source)
            rows = meta[meta["use"].astype(str).str.lower().isin({
                split,
                {"train": "training", "val": "validation", "test": "test"}[split],
            })].head(args.samples_per_split)
            print(f"\n-- split={split} folder={folder} sample_n={len(rows)} --")
            for _, row in rows.iterrows():
                filename = str(row["filename"])
                path = folder / filename
                if not path.exists():
                    alt = folder / source / filename
                    path = alt if alt.exists() else path
                print(f"file={filename} exists={path.exists()} label={row[label_col]!r}")
                if not path.exists():
                    continue
                with np.load(path, allow_pickle=True) as data:
                    print("  keys=" + ",".join(data.files))
                    stats = {}
                    if "slo_fundus" in data.files:
                        stats.update(describe_array(np, "slo", data["slo_fundus"]))
                    if "oct_bscans" in data.files:
                        stats.update(describe_array(np, "oct", data["oct_bscans"]))
                    print("  " + " ".join(f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
