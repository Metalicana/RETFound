#!/usr/bin/env python3
"""Build a canonical REFUGE manifest from the labeled official splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXPECTED_SPLIT_COUNTS = {"train": 400, "val": 400, "test": 400}
EXPECTED_SPLIT_LABEL_COUNTS = {
    ("train", 0): 360,
    ("train", 1): 40,
    ("val", 0): 360,
    ("val", 1): 40,
    ("test", 0): 360,
    ("test", 1): 40,
}
SOURCE_SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def ophthalmic_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = ophthalmic_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=root / "data_refuge" / "data.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data_refuge" / "manifest.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "data_refuge" / "prepare_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source_csv.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"filename", "Ground_Truth"}
    missing = required - set(source_rows[0] if source_rows else {})
    if missing:
        raise SystemExit(f"REFUGE source CSV is missing columns: {sorted(missing)}")

    rows = []
    for source in source_rows:
        relative_path = Path(str(source["filename"]).strip())
        if len(relative_path.parts) < 3:
            raise SystemExit(f"Unexpected REFUGE image path: {relative_path}")
        source_split = relative_path.parts[-2].strip().lower()
        split = SOURCE_SPLITS.get(source_split)
        if split is None:
            raise SystemExit(f"Unexpected REFUGE split in path: {relative_path}")
        try:
            label = int(source["Ground_Truth"])
        except ValueError as exc:
            raise SystemExit(
                f"Invalid REFUGE label {source['Ground_Truth']!r}: {relative_path}"
            ) from exc
        if label not in {0, 1}:
            raise SystemExit(f"Non-binary REFUGE label {label}: {relative_path}")

        case_id = f"refuge_{split}_{relative_path.stem.lower()}"
        rows.append(
            {
                "dataset": "refuge",
                "case_id": case_id,
                "patient_id": case_id,
                "split": split,
                "label": label,
                "cfp_path": relative_path.as_posix(),
                "official_partition": source_split,
                "label_source": str(args.source_csv),
            }
        )

    if len(rows) != 1200:
        raise SystemExit(f"Expected 1,200 REFUGE rows, found {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise SystemExit("REFUGE case IDs are not unique")

    split_counts = Counter(row["split"] for row in rows)
    split_label_counts = Counter((row["split"], row["label"]) for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(
            f"REFUGE split counts changed: expected={EXPECTED_SPLIT_COUNTS}, "
            f"observed={dict(split_counts)}"
        )
    if dict(split_label_counts) != EXPECTED_SPLIT_LABEL_COUNTS:
        raise SystemExit(
            "REFUGE split-label counts changed: "
            f"expected={EXPECTED_SPLIT_LABEL_COUNTS}, observed={dict(split_label_counts)}"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "refuge",
        "cases": len(rows),
        "manifest": str(args.manifest),
        "source_csv": str(args.source_csv),
        "official_split_rule": (
            "Official Train fits probes; official Validation selects probe and "
            "threshold; official Test is used only for final evaluation."
        ),
        "test_used_for_selection": False,
        "split_counts": dict(split_counts),
        "split_label_counts": {
            f"{split}|{label}": count
            for (split, label), count in sorted(split_label_counts.items())
        },
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
