#!/usr/bin/env python3
"""Validate canonical external-glaucoma manifests without loading images."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = {"dataset", "case_id", "patient_id", "split", "label"}
VALID_SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-cfp", action="store_true")
    parser.add_argument("--require-oct", action="store_true")
    parser.add_argument("--allow-missing-files", action="store_true")
    return parser.parse_args()


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = manifest.parent.parent / path
    return repo_candidate if repo_candidate.exists() else manifest.parent / path


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED - columns
        if missing_columns:
            raise SystemExit(f"Missing required columns: {sorted(missing_columns)}")
        rows = list(reader)
    if not rows:
        raise SystemExit("Manifest contains no rows")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise SystemExit("case_id values must be unique")

    errors: list[str] = []
    patients_by_split: dict[str, set[str]] = defaultdict(set)
    missing_files: Counter[str] = Counter()
    for index, row in enumerate(rows, start=2):
        split = row["split"].strip().lower()
        if split not in VALID_SPLITS:
            errors.append(f"line {index}: invalid split {split!r}")
        patients_by_split[split].add(row["patient_id"].strip())
        try:
            label = int(float(row["label"]))
        except ValueError:
            errors.append(f"line {index}: invalid label {row['label']!r}")
            continue
        if label not in (0, 1):
            errors.append(f"line {index}: label must be binary, got {label}")
        for column, required in (("cfp_path", args.require_cfp), ("oct_path", args.require_oct)):
            value = row.get(column, "").strip()
            if required and not value:
                errors.append(f"line {index}: {column} is required")
            if value and not resolve_path(args.manifest, value).exists():
                missing_files[column] += 1

    split_names = sorted(patients_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = patients_by_split[left] & patients_by_split[right]
            if overlap:
                errors.append(f"patient leakage between {left} and {right}: {len(overlap)} patients")
    if missing_files and not args.allow_missing_files:
        errors.append(f"missing files: {dict(missing_files)}")
    if errors:
        raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors[:30]))

    print(f"rows={len(rows)}")
    print(f"datasets={dict(Counter(row['dataset'] for row in rows))}")
    print(f"splits={dict(Counter(row['split'] for row in rows))}")
    print(f"labels={dict(Counter(int(float(row['label'])) for row in rows))}")
    print(f"patients={len({row['patient_id'] for row in rows})}")
    print("validation=passed")


if __name__ == "__main__":
    main()
