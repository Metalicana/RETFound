#!/usr/bin/env python3
"""Build a deterministic Drishti-GS1 manifest using its official split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_PATTERN = re.compile(r"drishtiGS_(\d{3})\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data_drishti")
    parser.add_argument("--manifest", type=Path, default=root / "data_drishti" / "manifest.csv")
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "data_drishti" / "prepare_summary.json",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Stratified validation fraction drawn only from official development IDs 001-050.",
    )
    return parser.parse_args()


def repository_path(path: Path, ophthalmic_root: Path) -> str:
    path = path.expanduser().absolute()
    root = ophthalmic_root.expanduser().absolute()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def validation_ids_by_label(
    development: list[dict[str, object]],
    fraction: float,
    seed: int,
) -> set[int]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one")
    by_label: dict[int, list[int]] = defaultdict(list)
    for row in development:
        by_label[int(row["label"])].append(int(row["image_number"]))

    validation: set[int] = set()
    for label, image_numbers in sorted(by_label.items()):
        rng = random.Random(f"{seed}:drishti:{label}")
        values = sorted(image_numbers)
        rng.shuffle(values)
        count = max(1, round(len(values) * fraction))
        if count >= len(values):
            raise ValueError(f"Validation allocation consumed label {label} development stratum")
        validation.update(values[:count])
    return validation


def main() -> None:
    args = parse_args()
    ophthalmic_root = Path(__file__).resolve().parents[1]
    source_rows: list[dict[str, object]] = []
    for folder_name, label in (("Normal", 0), ("Glaucoma", 1)):
        folder = args.data_root / folder_name
        if not folder.is_dir():
            raise SystemExit(f"Required Drishti folder does not exist: {folder}")
        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file():
                continue
            match = IMAGE_PATTERN.fullmatch(image_path.name)
            if not match:
                continue
            image_number = int(match.group(1))
            source_rows.append(
                {
                    "image_number": image_number,
                    "label": label,
                    "cfp_path": image_path,
                }
            )

    observed_ids = {int(row["image_number"]) for row in source_rows}
    expected_ids = set(range(1, 102))
    if len(source_rows) != 101 or observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        unexpected = sorted(observed_ids - expected_ids)
        raise SystemExit(
            "Expected exactly Drishti IDs 001-101; "
            f"rows={len(source_rows)}, missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    development = [row for row in source_rows if int(row["image_number"]) <= 50]
    official_test = [row for row in source_rows if int(row["image_number"]) >= 51]
    if len(development) != 50 or len(official_test) != 51:
        raise SystemExit(
            f"Official split mismatch: development={len(development)}, test={len(official_test)}"
        )
    validation_ids = validation_ids_by_label(
        development,
        args.validation_fraction,
        args.seed,
    )

    rows = []
    for source in sorted(source_rows, key=lambda row: int(row["image_number"])):
        image_number = int(source["image_number"])
        case_id = f"drishtiGS_{image_number:03d}"
        if image_number >= 51:
            split = "test"
        elif image_number in validation_ids:
            split = "val"
        else:
            split = "train"
        rows.append(
            {
                "dataset": "drishti",
                "case_id": case_id,
                "patient_id": case_id,
                "split": split,
                "label": int(source["label"]),
                "cfp_path": repository_path(Path(source["cfp_path"]), ophthalmic_root),
                "official_partition": "test" if image_number >= 51 else "development",
                "development_split_seed": "" if image_number >= 51 else args.seed,
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "drishti",
        "cases": len(rows),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "official_split_rule": "IDs 001-050 development; IDs 051-101 held-out test",
        "test_used_for_selection": False,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "label_counts": dict(Counter(str(row["label"]) for row in rows)),
        "split_label_counts": {
            f"{split}|{label}": count
            for (split, label), count in sorted(
                Counter((row["split"], str(row["label"])) for row in rows).items()
            )
        },
        "manifest": str(args.manifest),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
