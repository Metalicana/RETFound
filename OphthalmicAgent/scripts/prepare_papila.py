#!/usr/bin/env python3
"""Extract the PAPILA HF Parquet mirror and build patient-safe manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NEGATIVE_LABELS = {"healthy", "no glaucoma/healthy eye"}
POSITIVE_LABELS = {"glaucoma present"}
SUSPECT_LABELS = {"glaucoma suspect", "glaucoma-suspicious"}
IMAGE_FIELDS = ("retina", "cup_exp1", "cup_exp2", "disc_exp1", "disc_exp2", "opht_cont")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=root / "data_papila" / "raw" / "data")
    parser.add_argument("--extract-dir", type=Path, default=root / "data_papila" / "raw" / "extracted")
    parser.add_argument("--manifest", type=Path, default=root / "data_papila" / "manifest.csv")
    parser.add_argument("--suspect-manifest", type=Path, default=root / "data_papila" / "suspect_manifest.csv")
    parser.add_argument("--summary", type=Path, default=root / "data_papila" / "prepare_summary.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_metadata(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in text.split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        parsed[normalized] = value.strip()
    return parsed


def label_kind(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in NEGATIVE_LABELS:
        return "negative"
    if normalized in POSITIVE_LABELS:
        return "positive"
    if normalized in SUSPECT_LABELS:
        return "suspect"
    raise ValueError(f"Unknown PAPILA label: {value!r}")


def allocate_group(values: list[str], rng: random.Random) -> dict[str, str]:
    values = sorted(values)
    rng.shuffle(values)
    count = len(values)
    n_train = max(1, int(round(count * 0.60)))
    n_val = max(1, int(round(count * 0.20))) if count >= 3 else 0
    if n_train + n_val >= count:
        n_train = max(1, count - 2) if count >= 3 else count
        n_val = 1 if count >= 3 else 0
    assignments = {}
    for index, patient in enumerate(values):
        assignments[patient] = "train" if index < n_train else ("val" if index < n_train + n_val else "test")
    return assignments


def relative_to_ophthalmic(path: Path, ophthalmic_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(ophthalmic_root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_bytes(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    content = value.get("bytes")
    if not isinstance(content, bytes):
        raise ValueError(f"No embedded bytes for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        path.write_bytes(content)


def main() -> None:
    args = parse_args()
    import pyarrow.parquet as pq

    shards = sorted(args.parquet_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No Parquet shards found under {args.parquet_dir}")

    rows: list[dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(shard)
        rows.extend(table.to_pylist())
    if len(rows) != 488:
        raise SystemExit(f"Expected 488 PAPILA eyes, found {len(rows)}")

    patients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prepared: list[dict[str, Any]] = []
    for source in rows:
        filename = Path(source["retina"]["path"])
        patient_id = filename.stem
        metadata = parse_metadata(source.get("metadata", ""))
        eye = metadata.get("eye_side", "").upper()
        if eye not in {"OS", "OD"}:
            raise SystemExit(f"Invalid eye side for patient {patient_id}: {eye!r}")
        case_id = f"{patient_id}_{eye.lower()}"
        raw_label = source["sparse text"].strip().lower()
        kind = label_kind(raw_label)
        record = {
            "source": source,
            "patient_id": patient_id,
            "case_id": case_id,
            "eye": eye,
            "raw_label": raw_label,
            "label_kind": kind,
            "metadata": metadata,
            "metadata_raw": source.get("metadata", ""),
        }
        patients[patient_id].append(record)
        prepared.append(record)

    bad_pairs = {
        patient: values
        for patient, values in patients.items()
        if len(values) != 2 or {item["eye"] for item in values} != {"OS", "OD"}
    }
    if bad_pairs:
        raise SystemExit(f"Invalid bilateral pairs: {len(bad_pairs)}")

    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for patient, values in patients.items():
        strata[tuple(sorted(item["label_kind"] for item in values))].append(patient)
    rng = random.Random(args.seed)
    patient_split: dict[str, str] = {}
    for stratum in sorted(strata):
        patient_split.update(allocate_group(strata[stratum], rng))

    ophthalmic_root = Path(__file__).resolve().parents[1]
    binary_rows: list[dict[str, Any]] = []
    suspect_rows: list[dict[str, Any]] = []
    for item in sorted(prepared, key=lambda value: value["case_id"]):
        case_id = item["case_id"]
        paths = {}
        for field in IMAGE_FIELDS:
            original = Path(item["source"][field]["path"])
            suffix = original.suffix or ".png"
            destination = args.extract_dir / field / f"{case_id}{suffix}"
            write_bytes(destination, item["source"][field], args.overwrite)
            paths[field] = relative_to_ophthalmic(destination, ophthalmic_root)

        metadata = item["metadata"]
        common = {
            "dataset": "papila",
            "case_id": case_id,
            "patient_id": item["patient_id"],
            "eye": item["eye"].lower(),
            "split": patient_split[item["patient_id"]],
            "raw_label": item["raw_label"],
            "cfp_path": paths["retina"],
            "disc_mask_path": paths["disc_exp1"],
            "cup_mask_path": paths["cup_exp1"],
            "disc_mask_exp2_path": paths["disc_exp2"],
            "cup_mask_exp2_path": paths["cup_exp2"],
            "opht_cont_path": paths["opht_cont"],
            "age": metadata.get("age", ""),
            "gender_code": metadata.get("gender", ""),
            "pneumatic": metadata.get("pneumatic", ""),
            "perkins": metadata.get("perkins", ""),
            "pachymetry": metadata.get("pachymetry", ""),
            "axial_length": metadata.get("axial_length", ""),
            "vf_md": metadata.get("vf_md", ""),
            "metadata_json": json.dumps(metadata, sort_keys=True),
        }
        if item["label_kind"] == "suspect":
            suspect_rows.append({**common, "label": ""})
        else:
            binary_rows.append({**common, "label": 1 if item["label_kind"] == "positive" else 0})

    fields = list(binary_rows[0])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(binary_rows)
    with args.suspect_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(suspect_rows)

    summary = {
        "seed": args.seed,
        "parquet_shards": len(shards),
        "eyes": len(prepared),
        "patients": len(patients),
        "binary_eyes": len(binary_rows),
        "suspect_eyes": len(suspect_rows),
        "binary_split_counts": dict(Counter(row["split"] for row in binary_rows)),
        "binary_label_counts": dict(Counter(str(row["label"]) for row in binary_rows)),
        "binary_split_label_counts": {
            f"{split}|{label}": count
            for (split, label), count in sorted(Counter((row["split"], str(row["label"])) for row in binary_rows).items())
        },
        "suspect_split_counts": dict(Counter(row["split"] for row in suspect_rows)),
        "patient_split_counts": dict(Counter(patient_split.values())),
        "patient_strata": {"|".join(key): len(value) for key, value in sorted(strata.items())},
        "manifest": str(args.manifest),
        "suspect_manifest": str(args.suspect_manifest),
        "extract_dir": str(args.extract_dir),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
