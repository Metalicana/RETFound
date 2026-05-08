from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TASK_LABEL_COLUMNS = {
    "amd": "amd",
    "dr": "dr",
    "glaucoma": "glaucoma",
}

AMD_BINARY_MAP = {
    "no amd": 0,
    "no.amd.diagnosis": 0,
    "not.in.icd.table": 0,
    "early amd": 1,
    "early.dry": 1,
    "intermediate amd": 1,
    "intermediate.dry": 1,
    "late amd": 1,
    "advanced.atrophic.dry.with.subfoveal.involvement": 1,
    "advanced.atrophic.dry.without.subfoveal.involvement": 1,
    "wet.amd.active.choroidal.neovascularization": 1,
    "wet.amd.inactive.choroidal.neovascularization": 1,
    "wet.amd.inactive.scar": 1,
}
DR_BINARY_MAP = {
    "no dr": 0,
    "no.dr.diagnosis": 0,
    "not.in.icd.table": 0,
    "non-vision threatening dr": 0,
    "mild.npdr": 0,
    "moderate.npdr": 0,
    "vision threatening dr": 1,
    "severe.npdr": 1,
    "pdr": 1,
}
GLAUCOMA_BINARY_MAP = {"no": 0, "yes": 1, "0": 0, "1": 1}


def age_to_group(age: float | int | str | None) -> str:
    if pd.isna(age):
        return "missing"
    age_value = float(age)
    if age_value < 50:
        return "younger"
    if age_value < 70:
        return "middle-aged"
    return "older"


def infer_patient_id(filename: str) -> str:
    stem = Path(str(filename)).stem
    return stem.replace("data_", "")


def normalize_binary_label(value: object, task: str) -> int | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    key = str(value).strip().lower()
    if task == "amd":
        return AMD_BINARY_MAP.get(key, np.nan)
    if task == "dr":
        return DR_BINARY_MAP.get(key, np.nan)
    if task == "glaucoma":
        return GLAUCOMA_BINARY_MAP.get(key, np.nan)
    raise ValueError(f"Unknown task: {task}")


def load_fairvision_manifest(csv_path: str | Path, dataset: str, task: str) -> pd.DataFrame:
    """Load a Harvard-FairVision-style summary CSV into the project manifest schema."""
    task_key = task.lower()
    raw = pd.read_csv(csv_path)
    label_col = TASK_LABEL_COLUMNS.get(task_key)
    if label_col is None or label_col not in raw.columns:
        raise ValueError(f"Could not find label column for task={task!r} in {csv_path}")

    manifest = pd.DataFrame()
    manifest["patient_id"] = raw["filename"].map(infer_patient_id)
    manifest["eye_id"] = raw.get("eye_id", "")
    manifest["visit_id"] = raw.get("visit_id", "")
    manifest["image_id"] = raw["filename"].astype(str)
    manifest["filename"] = raw["filename"].astype(str)
    manifest["dataset"] = dataset
    manifest["task"] = task_key
    manifest["y_true"] = raw[label_col].map(lambda value: normalize_binary_label(value, task_key))
    manifest["split"] = raw["use"].replace({"training": "train", "validation": "val"}).astype(str)
    manifest["race"] = raw.get("race", "missing").fillna("missing").astype(str).str.lower()
    manifest["ethnicity"] = raw.get("ethnicity", "missing").fillna("missing").astype(str).str.lower()
    manifest["sex_gender"] = raw.get("gender", "missing").fillna("missing").astype(str).str.lower()
    manifest["age"] = pd.to_numeric(raw.get("age", np.nan), errors="coerce")
    manifest["age_group"] = manifest["age"].map(age_to_group)
    metadata_cols = ["race", "ethnicity", "sex_gender", "age"]
    manifest["metadata_missing_flag"] = manifest[metadata_cols].isna().any(axis=1) | (
        manifest[["race", "ethnicity", "sex_gender"]] == "missing"
    ).any(axis=1)
    return manifest


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "patient_id",
        "image_id",
        "dataset",
        "task",
        "y_true",
        "split",
        "race",
        "ethnicity",
        "sex_gender",
        "age",
        "age_group",
        "metadata_missing_flag",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    invalid_splits = set(manifest["split"].dropna().unique()) - {"train", "val", "validation", "test"}
    if invalid_splits:
        raise ValueError(f"Manifest contains invalid splits: {sorted(invalid_splits)}")

