from __future__ import annotations

import argparse
import os
from pathlib import Path


STANDARD_COLUMNS = [
    "patient_id",
    "eye_id",
    "visit_id",
    "image_id",
    "dataset",
    "task",
    "model_name",
    "y_true",
    "y_prob",
    "y_pred",
    "split",
    "race",
    "ethnicity",
    "sex_gender",
    "age",
    "age_group",
    "metadata_missing_flag",
]


AMD_MAP = {
    "normal": 0,
    "no amd": 0,
    "not.in.icd.table": 0,
    "no.amd.diagnosis": 0,
    "early.dry": 1,
    "intermediate.dry": 2,
    "advanced.atrophic.dry.with.subfoveal.involvement": 3,
    "advanced.atrophic.dry.without.subfoveal.involvement": 3,
    "wet.amd.active.choroidal.neovascularization": 3,
    "wet.amd.inactive.choroidal.neovascularization": 3,
    "wet.amd.inactive.scar": 3,
}


DR_MAP = {
    "normal": 0,
    "no dr": 0,
    "non-vision threatening dr": 0,
    "vision threatening dr": 1,
    "not.in.icd.table": 0,
    "no.dr.diagnosis": 0,
    "mild.npdr": 0,
    "moderate.npdr": 0,
    "severe.npdr": 1,
    "pdr": 1,
}


RACE_LABELS = {
    0: "asian",
    1: "black",
    2: "white",
    "Asian": "asian",
    "Black or African American": "black",
    "White or Caucasian": "white",
}


def age_to_group(age):
    if age in ("", None):
        return "missing"
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "missing"
    if value < 50:
        return "younger"
    if value < 70:
        return "middle-aged"
    return "older"


def scalar(value):
    try:
        return value.item()
    except AttributeError:
        return value


def text_scalar(value) -> str:
    raw = scalar(value)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw)


def race_value(raw) -> str:
    raw = scalar(raw)
    return RACE_LABELS.get(raw, str(raw).strip().lower() or "missing")


def sex_value(raw) -> str:
    raw = scalar(raw)
    if str(raw) == "1":
        return "male"
    if str(raw) == "0":
        return "female"
    return str(raw).strip().lower() or "missing"


def ethnicity_value(raw) -> str:
    raw = scalar(raw)
    if str(raw) == "1":
        return "hispanic"
    if str(raw) == "0":
        return "non-hispanic"
    return str(raw).strip().lower() or "missing"


def infer_patient_id(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("data_", "")


def find_split_dir(data_dir: Path, split: str) -> Path:
    candidates = [data_dir / split, data_dir / split.capitalize(), data_dir / split.title()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find split directory under {data_dir}: {candidates}")


def official_file_order(split_dir: Path, sort_files: bool) -> list[str]:
    files = [name for name in os.listdir(split_dir) if (split_dir / name).is_file() and name.endswith(".npz")]
    if sort_files:
        files.sort()
    return files


def metadata_lookup(data_dir: Path):
    import pandas as pd

    path = data_dir / "metadata_lookup.csv"
    if not path.exists():
        return {}
    return pd.read_csv(path).set_index("filename").to_dict("index")


def npz_or_meta(data, meta: dict, npz_key: str, meta_key: str | None = None, default=""):
    if npz_key in data.files:
        return data[npz_key]
    return meta.get(meta_key or npz_key, default)


def prediction_probability(pred_row, task: str) -> float:
    if task == "amd":
        if pred_row.ndim == 0:
            return float(pred_row)
        if len(pred_row) == 1:
            return float(pred_row[0])
        return float(1.0 - pred_row[0])
    return float(pred_row if pred_row.ndim == 0 else pred_row[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FairVision-main pred_gt_best_epoch.npz output to the equi-agent standard prediction CSV."
    )
    parser.add_argument("--task", choices=("amd", "dr"), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Disease-specific FairVision directory, e.g. /path/FairVision/AMD or /path/FairVision/DR.",
    )
    parser.add_argument("--pred-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--sort-files",
        action="store_true",
        help="Use only if the FairVision-main training script was changed to sort files before evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import pandas as pd

    payload = np.load(args.pred_npz, allow_pickle=True)
    preds = payload["test_pred"]
    gts = payload["test_gt"]

    split_dir = find_split_dir(args.data_dir, args.split)
    files = official_file_order(split_dir, sort_files=args.sort_files)
    meta_by_file = metadata_lookup(args.data_dir)
    if len(files) != len(preds):
        raise ValueError(
            f"File/prediction count mismatch: files={len(files)} preds={len(preds)}. "
            "Check --data-dir, --split, and --sort-files."
        )

    rows = []
    for index, filename in enumerate(files):
        npz_path = split_dir / filename
        with np.load(npz_path, allow_pickle=True) as data:
            meta = meta_by_file.get(filename, {})
            if args.task == "amd":
                disease_stage = AMD_MAP.get(text_scalar(npz_or_meta(data, meta, "amd_condition")), 0)
                y_true = int(disease_stage > 0)
            else:
                disease_stage = DR_MAP.get(text_scalar(npz_or_meta(data, meta, "dr_subtype")), 0)
                y_true = int(disease_stage > 0)

            official_gt = scalar(gts[index])
            if args.task == "amd":
                expected = int(float(official_gt) > 0)
            else:
                expected = int(float(official_gt) > 0)
            if y_true != expected:
                raise ValueError(
                    f"Label mismatch at row {index} file={filename}: npz={y_true} official_gt={official_gt}"
                )

            age = scalar(npz_or_meta(data, meta, "age", default=""))
            race = race_value(npz_or_meta(data, meta, "race", default="missing"))
            ethnicity = ethnicity_value(npz_or_meta(data, meta, "hispanic", default="missing"))
            sex_gender = sex_value(npz_or_meta(data, meta, "male", default="missing"))

        y_prob = prediction_probability(np.asarray(preds[index]), args.task)
        rows.append(
            {
                "patient_id": infer_patient_id(filename),
                "eye_id": "",
                "visit_id": "",
                "image_id": filename,
                "dataset": "harvard_fairvision",
                "task": args.task,
                "model_name": args.model_name,
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": int(y_prob >= args.threshold),
                "split": args.split,
                "race": race,
                "ethnicity": ethnicity,
                "sex_gender": sex_gender,
                "age": age,
                "age_group": age_to_group(age),
                "metadata_missing_flag": any(value in {"", "missing"} for value in [race, ethnicity, sex_gender]),
            }
        )

    output = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"wrote={args.out}")
    print(f"rows={len(output)}")
    print(f"task={args.task} model_name={args.model_name}")
    print(f"positive_rate={output['y_true'].mean():.6f} pred_positive_rate={output['y_pred'].mean():.6f}")


if __name__ == "__main__":
    main()
