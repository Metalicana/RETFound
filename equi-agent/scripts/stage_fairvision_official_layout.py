from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


TASK_TO_SOURCE = {
    "amd": "AMD",
    "dr": "DR",
    "glaucoma": "Glaucoma",
}


SPLIT_TO_FOLDER = {
    "training": "Training",
    "train": "Training",
    "validation": "Validation",
    "val": "Validation",
    "test": "Test",
}


SPLIT_TO_OFFICIAL = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a flat HuggingFace FairVision checkout into the disease/split folder layout "
            "expected by FairVision-main, using symlinks by default."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_TO_SOURCE), default=["amd", "dr"])
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def metadata_path(source_root: Path, source: str, task: str) -> Path:
    candidates = [
        source_root / f"data_summary_{task}.csv",
        source_root / "HarvardFairVision30k" / source / "ReadMe" / f"data_summary_{task}.csv",
        source_root / source / f"data_summary_{task}.csv",
        source_root / source / "ReadMe" / f"data_summary_{task}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find metadata CSV for {task}. Checked: {candidates}")


def source_file(source_root: Path, source: str, split_folder: str, filename: str) -> Path:
    candidates = [
        source_root / split_folder / filename,
        source_root / split_folder / source / filename,
        source_root / source / split_folder / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} for {source}/{split_folder}. Checked: {candidates}")


def link_or_copy(src: Path, dst: Path, copy: bool, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def official_metadata_frame(summary, task: str):
    import pandas as pd

    output = pd.DataFrame()
    output["filename"] = summary["filename"].astype(str)
    if task == "amd":
        output["amd_condition"] = summary.get("amd", "").astype(str)
        output["dr_subtype"] = ""
        output["glaucoma"] = ""
    elif task == "dr":
        output["amd_condition"] = ""
        output["dr_subtype"] = summary.get("dr", "").astype(str)
        output["glaucoma"] = ""
    else:
        output["amd_condition"] = ""
        output["dr_subtype"] = ""
        output["glaucoma"] = summary.get("glaucoma", "").astype(str)

    output["race"] = summary.get("race", "missing").astype(str)
    gender = summary.get("gender", "")
    output["male"] = gender.astype(str).str.lower().map({"male": 1, "m": 1, "female": 0, "f": 0}).fillna(-1).astype(int)
    ethnicity = summary.get("ethnicity", "")
    output["hispanic"] = (
        ethnicity.astype(str)
        .str.lower()
        .map({"hispanic": 1, "yes": 1, "non-hispanic": 0, "not hispanic": 0, "no": 0})
        .fillna(-1)
        .astype(int)
    )
    output["age"] = summary.get("age", "")
    output["use"] = summary.get("use", "")
    return output


def main() -> None:
    args = parse_args()
    import pandas as pd

    args.out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for task in args.tasks:
        source = TASK_TO_SOURCE[task]
        csv_path = metadata_path(args.source_root, source, task)
        summary = pd.read_csv(csv_path)
        if "filename" not in summary.columns or "use" not in summary.columns:
            raise ValueError(f"{csv_path} must contain filename and use columns.")

        rows = 0
        for _, row in summary.iterrows():
            split_key = str(row["use"]).strip().lower()
            if split_key not in SPLIT_TO_FOLDER:
                continue
            filename = str(row["filename"]).strip()
            src = source_file(args.source_root, source, SPLIT_TO_FOLDER[split_key], filename)
            dst = args.out_root / source / SPLIT_TO_OFFICIAL[split_key] / filename
            link_or_copy(src, dst, copy=args.copy, overwrite=args.overwrite)
            rows += 1
        official_metadata_frame(summary, task).to_csv(args.out_root / source / "metadata_lookup.csv", index=False)
        total += rows
        print(f"staged task={task} source={source} rows={rows} csv={csv_path}")

    print(f"out_root={args.out_root}")
    print(f"total_staged={total}")


if __name__ == "__main__":
    main()
