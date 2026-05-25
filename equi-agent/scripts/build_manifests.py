from __future__ import annotations

import argparse
from pathlib import Path


FAIRVISION_TASKS = {
    "amd": "AMD",
    "dr": "DR",
    "glaucoma": "Glaucoma",
}

GDP_PROGRESSION_TARGETS = {
    "md": "progression.md",
    "vfi": "progression.vfi",
    "td_pointwise": "progression.td_pointwise",
    "md_fast": "progression.md_fast",
    "md_fast_no_p_cut": "progression.md_fast_no_p_cut",
    "td_pointwise_no_p_cut": "progression.td_pointwise_no_p_cut",
}

GDP_TD_COLUMNS = [
    "td1",
    "td2",
    "td3",
    "td4",
    "td5",
    "td6",
    "td7",
    "td8",
    "td9",
    "td10",
    "td11",
    "td12",
    "td13",
    "td14",
    "td15",
    "td16",
    "td17",
    "td18",
    "td19",
    "td20",
    "td21",
    "td22",
    "td23",
    "td24",
    "td26",
    "td27",
    "td28",
    "td29",
    "td30",
    "td31",
    "td32",
    "td33",
    "td35",
    "td36",
    "td37",
    "td38",
    "td39",
    "td40",
    "td41",
    "td42",
    "td43",
    "td44",
    "td45",
    "td46",
    "td47",
    "td48",
    "td49",
    "td50",
    "td51",
    "td52",
    "td53",
    "td54",
]

np = None
pd = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_data_libs() -> None:
    global np, pd
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    if pd is None:
        import pandas as pandas_module

        pd = pandas_module


def age_to_group(age: object) -> str:
    if pd.isna(age):
        return "missing"
    age_value = float(age)
    if age_value < 50:
        return "younger"
    if age_value < 70:
        return "middle-aged"
    return "older"


def normalize_split(value: object) -> str:
    key = str(value).strip().lower()
    return {"training": "train", "validation": "val", "valid": "val"}.get(key, key)


def infer_patient_id(filename: object) -> str:
    return Path(str(filename)).stem.replace("data_", "")


def gdp_data_summary_path(gdp_root: Path) -> Path:
    candidates = [
        gdp_root / "ReadMe" / "data_summary.csv",
        gdp_root / "data_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find GDP data_summary.csv under {gdp_root}")


def normalize_binary_label(value: object, task: str) -> int | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)

    key = str(value).strip().lower()
    if task == "amd":
        return {
            "normal": 0,
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
        }.get(key, np.nan)
    if task == "dr":
        return {
            "no dr": 0,
            "no.dr.diagnosis": 0,
            "not.in.icd.table": 0,
            "non-vision threatening dr": 0,
            "mild.npdr": 0,
            "moderate.npdr": 0,
            "vision threatening dr": 1,
            "severe.npdr": 1,
            "pdr": 1,
        }.get(key, np.nan)
    if task == "glaucoma":
        return {"no": 0, "yes": 1, "0": 0, "1": 1}.get(key, np.nan)
    raise ValueError(f"Unknown task: {task}")


def build_fairvision_task_manifest(fairvision_root: Path, task: str) -> pd.DataFrame:
    disease = FAIRVISION_TASKS[task]
    csv_path = fairvision_root / "HarvardFairVision30k" / disease / "ReadMe" / f"data_summary_{task}.csv"
    raw = pd.read_csv(csv_path)

    manifest = pd.DataFrame()
    manifest["patient_id"] = raw["filename"].map(infer_patient_id)
    manifest["eye_id"] = ""
    manifest["visit_id"] = ""
    manifest["image_id"] = raw["filename"].astype(str)
    manifest["filename"] = raw["filename"].astype(str)
    manifest["dataset"] = "harvard_fairvision"
    manifest["task"] = task
    manifest["label_raw"] = raw[task]
    manifest["y_true"] = raw[task].map(lambda value: normalize_binary_label(value, task))
    manifest["split"] = raw["use"].map(normalize_split)
    manifest["race"] = raw.get("race", "missing").fillna("missing").astype(str).str.lower()
    manifest["ethnicity"] = raw.get("ethnicity", "missing").fillna("missing").astype(str).str.lower()
    manifest["sex_gender"] = raw.get("gender", "missing").fillna("missing").astype(str).str.lower()
    manifest["age"] = pd.to_numeric(raw.get("age", np.nan), errors="coerce")
    manifest["age_group"] = manifest["age"].map(age_to_group)
    manifest["metadata_missing_flag"] = metadata_missing_flag(manifest)
    manifest["image_path"] = manifest.apply(
        lambda row: str(fairvision_root / split_folder(row["split"]) / row["filename"]),
        axis=1,
    )
    manifest["oct_key"] = "oct_bscans"
    manifest["fundus_key"] = "slo_fundus"
    return manifest


def build_gdp_detection_manifest(gdp_root: Path) -> pd.DataFrame:
    raw = pd.read_csv(gdp_data_summary_path(gdp_root))
    filename_npz = raw["filename"].astype(str).map(lambda name: name if name.endswith(".npz") else f"{name}.npz")

    manifest = pd.DataFrame()
    manifest["patient_id"] = raw["filename"].map(infer_patient_id)
    manifest["eye_id"] = ""
    manifest["visit_id"] = ""
    manifest["image_id"] = filename_npz
    manifest["filename"] = filename_npz
    manifest["dataset"] = "harvard_gdp"
    manifest["task"] = "glaucoma_detection"
    manifest["label_raw"] = raw["glaucoma"]
    manifest["y_true"] = raw["glaucoma"].map(lambda value: normalize_binary_label(value, "glaucoma"))
    manifest["split"] = raw["glaucoma_detection_use"].map(normalize_split)
    manifest["race"] = raw.get("race", "missing").fillna("missing").astype(str).str.lower()
    manifest["ethnicity"] = raw.get("hispanic", "missing").map(hispanic_to_ethnicity)
    manifest["sex_gender"] = raw.get("gender", "missing").fillna("missing").astype(str).str.lower()
    manifest["age"] = pd.to_numeric(raw.get("age", np.nan), errors="coerce")
    manifest["age_group"] = manifest["age"].map(age_to_group)
    manifest["metadata_missing_flag"] = metadata_missing_flag(manifest)
    manifest["bscan_path"] = filename_npz.map(lambda name: str(gdp_root / "Bscan" / name))
    manifest["rnflt_path"] = filename_npz.map(lambda name: str(gdp_root / "RNFLT" / name))
    manifest["bscan_key"] = "bscans"
    manifest["rnflt_key"] = "rnflt"
    manifest["md"] = pd.to_numeric(raw.get("md", np.nan), errors="coerce")
    for col in GDP_TD_COLUMNS:
        manifest[col] = pd.to_numeric(raw.get(col, np.nan), errors="coerce")
    return manifest


def build_gdp_progression_manifest(gdp_root: Path, target_key: str = "md") -> pd.DataFrame:
    if target_key not in GDP_PROGRESSION_TARGETS:
        raise ValueError(f"Unknown GDP progression target {target_key!r}; choose from {sorted(GDP_PROGRESSION_TARGETS)}")
    target_col = GDP_PROGRESSION_TARGETS[target_key]
    raw = pd.read_csv(gdp_data_summary_path(gdp_root))
    progression_split = raw["progression_forecasting_use"].astype(str).str.strip().str.lower()
    raw = raw[
        progression_split.isin(["training", "test"])
        & raw[target_col].notna()
        & (raw[target_col].astype(str).str.strip().str.lower() != "nan")
        & (raw[target_col].astype(str).str.strip().str.upper() != "NA")
    ].copy()
    filename_npz = raw["filename"].astype(str).map(lambda name: name if name.endswith(".npz") else f"{name}.npz")

    manifest = pd.DataFrame()
    manifest["patient_id"] = raw["filename"].map(infer_patient_id)
    manifest["eye_id"] = ""
    manifest["visit_id"] = ""
    manifest["image_id"] = filename_npz
    manifest["filename"] = filename_npz
    manifest["dataset"] = "harvard_gdp"
    manifest["task"] = "progression_forecasting"
    manifest["split"] = raw["progression_forecasting_use"].map(normalize_split)
    manifest["label_raw"] = raw[target_col]
    manifest["y_true"] = pd.to_numeric(raw[target_col], errors="coerce")
    manifest["progression_target"] = target_key
    manifest["progression_target_column"] = target_col
    manifest["race"] = raw.get("race", "missing").fillna("missing").astype(str).str.lower()
    manifest["ethnicity"] = raw.get("hispanic", "missing").map(hispanic_to_ethnicity)
    manifest["sex_gender"] = raw.get("gender", "missing").fillna("missing").astype(str).str.lower()
    manifest["age"] = pd.to_numeric(raw.get("age", np.nan), errors="coerce")
    manifest["age_group"] = manifest["age"].map(age_to_group)
    manifest["metadata_missing_flag"] = metadata_missing_flag(manifest)
    manifest["bscan_path"] = filename_npz.map(lambda name: str(gdp_root / "Bscan" / name))
    manifest["rnflt_path"] = filename_npz.map(lambda name: str(gdp_root / "RNFLT" / name))
    manifest["bscan_key"] = "bscans"
    manifest["rnflt_key"] = "rnflt"
    manifest["md"] = pd.to_numeric(raw.get("md", np.nan), errors="coerce")
    manifest["progression_md"] = pd.to_numeric(raw.get("progression.md", np.nan), errors="coerce")
    manifest["progression_vfi"] = pd.to_numeric(raw.get("progression.vfi", np.nan), errors="coerce")
    manifest["progression_td_pointwise"] = pd.to_numeric(raw.get("progression.td_pointwise", np.nan), errors="coerce")
    manifest["progression_md_fast"] = pd.to_numeric(raw.get("progression.md_fast", np.nan), errors="coerce")
    manifest["progression_md_fast_no_p_cut"] = pd.to_numeric(raw.get("progression.md_fast_no_p_cut", np.nan), errors="coerce")
    manifest["progression_td_pointwise_no_p_cut"] = pd.to_numeric(
        raw.get("progression.td_pointwise_no_p_cut", np.nan), errors="coerce"
    )
    for col in GDP_TD_COLUMNS:
        manifest[col] = pd.to_numeric(raw.get(col, np.nan), errors="coerce")
    return manifest


def metadata_missing_flag(df: pd.DataFrame) -> pd.Series:
    demographic_cols = ["race", "ethnicity", "sex_gender"]
    return df[demographic_cols].isna().any(axis=1) | (df[demographic_cols] == "missing").any(axis=1) | df[
        "age"
    ].isna()


def split_folder(split: str) -> str:
    return {"train": "Training", "val": "Validation", "test": "Test"}[split]


def hispanic_to_ethnicity(value: object) -> str:
    key = str(value).strip().lower()
    if key == "yes":
        return "hispanic"
    if key == "no":
        return "non-hispanic"
    if key in {"", "nan", "none"}:
        return "missing"
    return key


def file_exists_series(paths: pd.Series) -> pd.Series:
    return paths.map(lambda path: Path(path).exists())


def summarize_manifest(name: str, manifest: pd.DataFrame) -> None:
    print(f"{name}: rows={len(manifest)}")
    print(f"  splits={manifest['split'].value_counts(dropna=False).to_dict()}")
    print(f"  labels={manifest['y_true'].value_counts(dropna=False).to_dict()}")
    unknown = manifest[manifest["y_true"].isna()]
    if not unknown.empty:
        print(f"  unknown_label_raw={unknown['label_raw'].value_counts(dropna=False).to_dict()}")
    print(f"  races={manifest['race'].value_counts(dropna=False).head(10).to_dict()}")
    path_cols = [col for col in ["image_path", "bscan_path", "rnflt_path"] if col in manifest.columns]
    for col in path_cols:
        missing = int((~file_exists_series(manifest[col])).sum())
        print(f"  missing_{col}={missing}")


def save_manifest(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical Equi-Agent dataset manifests.")
    parser.add_argument("--datasets-root", type=Path, default=repo_root() / "Datasets")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "equi-agent" / "outputs" / "manifests")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("fairvision", "gdp"),
        default=("fairvision", "gdp"),
        help="Dataset families to build.",
    )
    parser.add_argument(
        "--gdp-progression-targets",
        nargs="+",
        choices=sorted(GDP_PROGRESSION_TARGETS),
        default=sorted(GDP_PROGRESSION_TARGETS),
        help="GDP progression labels to emit as target-specific manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_data_libs()
    datasets_root = args.datasets_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    fairvision_root = datasets_root / "FairVision"
    gdp_root = datasets_root / "GDP"

    manifests = []
    if "fairvision" in args.datasets:
        for task in FAIRVISION_TASKS:
            manifest = build_fairvision_task_manifest(fairvision_root, task)
            manifests.append(manifest)
            save_manifest(manifest, out_dir / f"fairvision_{task}.csv")
            summarize_manifest(f"fairvision_{task}", manifest)

    if "gdp" in args.datasets:
        gdp_detection = build_gdp_detection_manifest(gdp_root)
        manifests.append(gdp_detection)
        save_manifest(gdp_detection, out_dir / "gdp_glaucoma_detection.csv")
        summarize_manifest("gdp_glaucoma_detection", gdp_detection)

        gdp_progression_legacy = None
        for target_key in args.gdp_progression_targets:
            gdp_progression = build_gdp_progression_manifest(gdp_root, target_key=target_key)
            manifests.append(gdp_progression)
            save_manifest(gdp_progression, out_dir / f"gdp_progression_forecasting_{target_key}.csv")
            summarize_manifest(f"gdp_progression_forecasting_{target_key}", gdp_progression)
            if target_key == "md":
                gdp_progression_legacy = gdp_progression

        if gdp_progression_legacy is not None:
            save_manifest(gdp_progression_legacy, out_dir / "gdp_progression_forecasting.csv")

    if manifests:
        combined = pd.concat(manifests, ignore_index=True, sort=False)
        save_manifest(combined, out_dir / "all_manifest.csv")
        print(f"combined: rows={len(combined)}")
    print(f"wrote={out_dir}")


if __name__ == "__main__":
    main()
