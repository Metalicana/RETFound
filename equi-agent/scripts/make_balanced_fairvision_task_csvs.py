from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any


TASK_FOLDERS = {
    "amd": "AMD",
    "dr": "DR",
    "glaucoma": "Glaucoma",
}
OUTPUT_COLUMNS = ["filename", "Task_Folder", "Ground_Truth", "GT_old", "Age", "Gender", "Race", "Ethnicity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one class-balanced FairVision test CSV per disease task in the "
            "legacy OphthalmicAgent CSV format."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("equi-agent/outputs/fairvision_reliability_selective_arbitration/selective_arbitration_predictions.csv"),
        help="Case-level FairVision test CSV with task, y_true, image_id, and metadata columns.",
    )
    parser.add_argument(
        "--legacy-csv",
        type=Path,
        default=Path("OphthalmicAgent/data/fairvision_250each.csv"),
        help="Optional old CSV used only to recover GT_old values when filenames overlap.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("OphthalmicAgent/data"))
    parser.add_argument("--per-class", type=int, default=125)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def task_filename(task: str, image_id: str) -> str:
    return f"data/{TASK_FOLDERS[task]}/Test/{image_id}"


def load_gt_old_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {row.get("filename", ""): row.get("GT_old", "") for row in read_csv(path)}


def legacy_row(row: dict[str, str], gt_old_lookup: dict[str, str]) -> dict[str, str]:
    task = row["task"].lower()
    filename = task_filename(task, row["image_id"])
    return {
        "filename": filename,
        "Task_Folder": TASK_FOLDERS[task],
        "Ground_Truth": str(int(float(row["y_true"]))),
        "GT_old": gt_old_lookup.get(filename, ""),
        "Age": row.get("age", ""),
        "Gender": row.get("sex_gender", ""),
        "Race": row.get("race", ""),
        "Ethnicity": row.get("ethnicity", ""),
    }


def choose_balanced(rows: list[dict[str, str]], task: str, per_class: int, rng: random.Random) -> list[dict[str, str]]:
    task_rows = [row for row in rows if row.get("task", "").lower() == task]
    by_label = {
        "0": [row for row in task_rows if str(row.get("y_true", "")).strip() in {"0", "0.0"}],
        "1": [row for row in task_rows if str(row.get("y_true", "")).strip() in {"1", "1.0"}],
    }
    missing = {label: per_class - len(label_rows) for label, label_rows in by_label.items() if len(label_rows) < per_class}
    if missing:
        raise SystemExit(f"Not enough cases for {task}: {missing}")
    selected: list[dict[str, str]] = []
    for label in ["0", "1"]:
        label_rows = list(by_label[label])
        rng.shuffle(label_rows)
        selected.extend(label_rows[:per_class])
    selected.sort(key=lambda row: (int(float(row["y_true"])), row.get("image_id", "")))
    return selected


def main() -> None:
    args = parse_args()
    rows = read_csv(args.source)
    gt_old_lookup = load_gt_old_lookup(args.legacy_csv)
    rng = random.Random(args.seed)
    summary_rows = []

    for task in ["amd", "dr", "glaucoma"]:
        selected = choose_balanced(rows, task, args.per_class, rng)
        output_rows = [legacy_row(row, gt_old_lookup) for row in selected]
        out_path = args.out_dir / f"fairvision_{task}_balanced_{args.per_class * 2}.csv"
        write_csv(out_path, output_rows, OUTPUT_COLUMNS)
        counts = Counter(row["Ground_Truth"] for row in output_rows)
        summary_rows.append(
            {
                "task": task,
                "path": str(out_path),
                "n": len(output_rows),
                "negative": counts.get("0", 0),
                "positive": counts.get("1", 0),
                "seed": args.seed,
                "source": str(args.source),
            }
        )

    summary_path = args.out_dir / f"fairvision_balanced_{args.per_class * 2}_summary.csv"
    write_csv(summary_path, summary_rows, ["task", "path", "n", "negative", "positive", "seed", "source"])
    print(f"wrote_summary={summary_path}")
    for row in summary_rows:
        print(f"{row['task']}: n={row['n']} negative={row['negative']} positive={row['positive']} path={row['path']}")


if __name__ == "__main__":
    main()
