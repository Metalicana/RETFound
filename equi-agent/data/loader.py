from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DISEASES = ("AMD", "DR", "Glaucoma")
SPLIT_MAP = {
    "train": "Training",
    "training": "Training",
    "val": "Validation",
    "valid": "Validation",
    "validation": "Validation",
    "test": "Test",
}

AMD_STAGE_MAP = {
    "not.in.icd.table": 0,
    "no.amd.diagnosis": 0,
    "no amd": 0,
    "early.dry": 1,
    "early amd": 1,
    "intermediate.dry": 2,
    "intermediate amd": 2,
    "advanced.atrophic.dry.with.subfoveal.involvement": 3,
    "advanced.atrophic.dry.without.subfoveal.involvement": 3,
    "wet.amd.active.choroidal.neovascularization": 3,
    "wet.amd.inactive.choroidal.neovascularization": 3,
    "wet.amd.inactive.scar": 3,
    "late amd": 3,
}

DR_STAGE_MAP = {
    "not.in.icd.table": 0,
    "no.dr.diagnosis": 0,
    "no dr": 0,
    "mild.npdr": 0,
    "moderate.npdr": 0,
    "non-vision threatening dr": 0,
    "severe.npdr": 1,
    "pdr": 1,
    "vision threatening dr": 1,
}

GLAUCOMA_STAGE_MAP = {
    "0": 0,
    "1": 1,
    "no": 0,
    "yes": 1,
    "false": 0,
    "true": 1,
}


def default_fairvision_root() -> Path:
    return Path(__file__).resolve().parents[2] / "Datasets" / "FairVision"


def default_gdp_root() -> Path:
    return Path(__file__).resolve().parents[2] / "Datasets" / "GDP"


def normalize_split(split: object) -> str:
    return SPLIT_MAP.get(str(split).strip().lower(), str(split))


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def normalize_stage(value: object, disease: str) -> int | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)

    key = str(scalar(value)).strip().lower()
    if disease == "AMD":
        return AMD_STAGE_MAP.get(key, np.nan)
    if disease == "DR":
        return DR_STAGE_MAP.get(key, np.nan)
    if disease == "Glaucoma":
        return GLAUCOMA_STAGE_MAP.get(key, np.nan)
    raise ValueError(f"Unknown disease: {disease}")


class GenericEyeLoader:
    """FairVision loader for the local RETFound/Datasets/FairVision mirror."""

    def __init__(self, base_path: str | Path | None = None):
        self.base_path = Path(base_path).expanduser().resolve() if base_path else default_fairvision_root()

    def get_metadata(self, disease: str) -> pd.DataFrame:
        disease = self._normalize_disease(disease)
        csv_name = f"data_summary_{disease.lower()}.csv"
        csv_path = self.base_path / "HarvardFairVision30k" / disease / "ReadMe" / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {csv_path}. Expected FairVision root to contain "
                "HarvardFairVision30k/{AMD,DR,Glaucoma}/ReadMe/."
            )
        return pd.read_csv(csv_path)

    def iter_split(self, disease: str, split: str = "test") -> pd.DataFrame:
        metadata = self.get_metadata(disease)
        wanted = str(split).strip().lower()
        return metadata[metadata["use"].astype(str).str.lower().map(lambda value: value in {wanted, normalize_split(wanted).lower()})]

    def resolve_npz_path(self, disease: str, filename: str, split: object) -> Path:
        disease = self._normalize_disease(disease)
        split_folder = normalize_split(split)
        candidates = [
            self.base_path / split_folder / filename,
            self.base_path / split_folder / disease / filename,
            self.base_path / disease / split_folder / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def load_patient(self, disease: str, patient_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        disease = self._normalize_disease(disease)
        row = patient_row.to_dict() if hasattr(patient_row, "to_dict") else dict(patient_row)
        npz_path = self.resolve_npz_path(disease, row["filename"], row["use"])
        if not npz_path.exists():
            raise FileNotFoundError(f"Patient NPZ not found: {npz_path}")

        container = np.load(npz_path)
        stage = self._stage_from_row_or_npz(disease, row, container)

        oct_volume = container["oct_bscans"]
        oct_slice = oct_volume[oct_volume.shape[0] // 2]
        if oct_slice.max() <= 1.0:
            oct_slice = (oct_slice * 255).astype(np.uint8)
        else:
            oct_slice = oct_slice.astype(np.uint8)

        fundus_img = container["slo_fundus"].astype(np.float32)
        f_min, f_max = fundus_img.min(), fundus_img.max()
        if f_max - f_min > 0:
            fundus_img = 255 * (fundus_img - f_min) / (f_max - f_min)
        fundus_img = fundus_img.astype(np.uint8)

        return {
            "metadata": row,
            "oct_tensors": oct_volume,
            "fundus_img": fundus_img,
            "directory": str(npz_path),
            "stage": stage,
            "oct_img": oct_slice.astype(np.uint8),
        }

    def _stage_from_row_or_npz(self, disease: str, row: dict[str, Any], container: Any) -> int | float:
        row_key = {"AMD": "amd", "DR": "dr", "Glaucoma": "glaucoma"}[disease]
        if row_key in row:
            stage = normalize_stage(row[row_key], disease)
            if not pd.isna(stage):
                return stage

        npz_key = {"AMD": "amd_condition", "DR": "dr_subtype", "Glaucoma": "glaucoma"}[disease]
        if npz_key in container:
            return normalize_stage(scalar(container[npz_key]), disease)
        return np.nan

    def _normalize_disease(self, disease: str) -> str:
        aliases = {item.lower(): item for item in DISEASES}
        key = str(disease).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown disease {disease!r}; expected one of {DISEASES}")
        return aliases[key]


class HarvardGDPLoader:
    """Loader helpers for RETFound/Datasets/GDP."""

    def __init__(self, base_path: str | Path | None = None):
        self.base_path = Path(base_path).expanduser().resolve() if base_path else default_gdp_root()

    def get_metadata(self) -> pd.DataFrame:
        csv_path = self.base_path / "ReadMe" / "data_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"GDP metadata CSV not found: {csv_path}")
        return pd.read_csv(csv_path)

    def iter_split(self, task: str = "glaucoma_detection", split: str = "test") -> pd.DataFrame:
        metadata = self.get_metadata()
        split_col = self._split_column(task)
        wanted = str(split).strip().lower()
        return metadata[metadata[split_col].astype(str).str.lower() == wanted]

    def resolve_data_path(self, filename: str, modality: str = "Bscan", split: object | None = None) -> Path:
        modality = self._normalize_modality(modality)
        names = [filename]
        if Path(filename).suffix == "":
            names.append(f"{filename}.npz")

        split_folder = normalize_split(split) if split is not None else None
        candidates: list[Path] = []
        for name in names:
            if split_folder:
                candidates.extend(
                    [
                        self.base_path / modality / split_folder / name,
                        self.base_path / split_folder / modality / name,
                    ]
                )
            candidates.append(self.base_path / modality / name)

        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def load_record(
        self,
        row: pd.Series | dict[str, Any],
        modality: str = "Bscan",
        split_col: str = "glaucoma_detection_use",
    ) -> dict[str, Any]:
        record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        data_path = self.resolve_data_path(record["filename"], modality, record.get(split_col))
        if not data_path.exists():
            raise FileNotFoundError(f"GDP data file not found: {data_path}")
        data = np.load(data_path)
        return {
            "metadata": record,
            "data": data,
            "directory": str(data_path),
            "stage": normalize_stage(record.get("glaucoma"), "Glaucoma"),
        }

    def _split_column(self, task: str) -> str:
        task_key = str(task).strip().lower()
        if task_key in {"glaucoma", "glaucoma_detection", "detection"}:
            return "glaucoma_detection_use"
        if task_key in {"progression", "progression_forecasting", "forecasting"}:
            return "progression_forecasting_use"
        raise ValueError("GDP task must be glaucoma_detection or progression_forecasting")

    def _normalize_modality(self, modality: str) -> str:
        aliases = {"bscan": "Bscan", "bscans": "Bscan", "rnflt": "RNFLT"}
        key = str(modality).strip().lower()
        if key not in aliases:
            raise ValueError("GDP modality must be Bscan or RNFLT")
        return aliases[key]
