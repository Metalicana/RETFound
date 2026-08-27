#!/usr/bin/env python3
"""Prepare blinded FairVision case montages and metadata for Google Forms.

The script joins a doctor-facing ``manifest.csv`` to its private
``manifest_XX_selected_rows.csv`` audit file, extracts only approved
demographics, creates one blinded SLO+OCT montage per case, and writes
``form_cases.csv``. Upload the generated directory contents to one Google Drive
folder, then run ``create_google_form.gs`` using that folder's ID.

Ground truth, model predictions, probabilities, correctness, and agent text are
never copied to the Google Form bundle. Mixed manifests are ordered as
glaucoma, AMD, then DR.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


FILE_CANDIDATES = (
    "filename", "Filename", "filepath", "Filepath", "file_path", "path", "npz_path"
)
CASE_CANDIDATES = ("case_id", "Case_ID", "patient_id", "Patient_ID")
AGE_CANDIDATES = ("Age", "age")
GENDER_CANDIDATES = ("Gender", "gender", "Sex", "sex")
RACE_CANDIDATES = ("Race", "race")
ETHNICITY_CANDIDATES = ("Ethnicity", "ethnicity")
TRUTH_CANDIDATES = ("Ground_Truth", "ground_truth", "groundtruth", "label", "y_true", "_truth")
PREDICTION_CANDIDATES = {
    "glaucoma": ("Pred_GL", "pred_gl", "prediction", "predicted_label", "_prediction"),
    "amd": ("Pred_AMD", "pred_amd", "prediction", "predicted_label", "_prediction"),
    "dr": ("Pred_DR", "pred_dr", "prediction", "predicted_label", "_prediction"),
}
FORBIDDEN_OUTPUT_TERMS = (
    "ground_truth", "groundtruth", "prediction", "pred_", "probability",
    "correct", "reasoning", "decision", "response", "label", "_truth", "_prediction",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a blinded FairVision Google Form image/metadata bundle."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--disease", choices=("glaucoma", "amd", "dr", "mixed"), required=True
    )
    parser.add_argument("--oct-slices", type=int, default=8)
    parser.add_argument("--age-display", choices=("exact", "decade"), default="exact")
    parser.add_argument("--max-image-width", type=int, default=1600)
    return parser.parse_args()


def find_column(columns, candidates, *, required=True):
    folded = {str(column).strip().casefold(): column for column in columns}
    for candidate in candidates:
        match = folded.get(candidate.strip().casefold())
        if match is not None:
            return match
    if required:
        raise KeyError(f"Expected one of {list(candidates)}; found {list(columns)}")
    return None


def clean_text(value, default="Not available"):
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text and text.casefold() not in {"nan", "none"} else default


def normalized_path(value):
    return str(value).strip().replace("\\", "/").casefold()


def safe_case_id(value, fallback):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean_text(value, fallback)).strip("._")
    return text or fallback


def resolve_data_path(value, data_root, manifest_path, audit_path):
    path = Path(str(value)).expanduser()
    if path.is_absolute() and path.is_file():
        return path
    candidates = (
        data_root / path,
        manifest_path.parent / path,
        audit_path.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve image container {value!r}; tried "
        + ", ".join(str(item) for item in candidates)
    )


def normalize_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip(255 * (array - low) / (high - low), 0, 255).astype(np.uint8)


def as_rgb(array):
    return Image.fromarray(normalize_uint8(array)).convert("RGB")


def label_panel(image, label, width, height):
    contained = ImageOps.contain(image.convert("RGB"), (width, height - 34))
    panel = Image.new("RGB", (width, height), "black")
    x = (width - contained.width) // 2
    y = 34 + (height - 34 - contained.height) // 2
    panel.paste(contained, (x, y))
    ImageDraw.Draw(panel).text((10, 9), label, fill="white", font=ImageFont.load_default())
    return panel


def make_montage(npz_path, oct_slices, max_width):
    with np.load(npz_path) as container:
        if "slo_fundus" not in container or "oct_bscans" not in container:
            raise KeyError(
                f"{npz_path} must contain slo_fundus and oct_bscans; found {list(container.files)}"
            )
        fundus = as_rgb(container["slo_fundus"])
        volume = np.asarray(container["oct_bscans"])
    if volume.ndim != 3 or len(volume) == 0:
        raise ValueError(f"Expected OCT [slices,H,W], got {volume.shape}: {npz_path}")
    count = max(1, min(oct_slices, len(volume)))
    indices = np.linspace(0, len(volume) - 1, count, dtype=int)
    oct_images = [as_rgb(volume[index]) for index in indices]

    cell_width, cell_height = 360, 210
    fundus_panel = label_panel(fundus, "SLO / Fundus", 480, 420)
    columns = 4
    rows = int(np.ceil(count / columns))
    oct_grid = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
    for position, (index, image) in enumerate(zip(indices, oct_images)):
        panel = label_panel(image, f"OCT slice {int(index)}", cell_width, cell_height)
        oct_grid.paste(panel, ((position % columns) * cell_width, (position // columns) * cell_height))

    canvas_width = max(fundus_panel.width, oct_grid.width)
    canvas = Image.new("RGB", (canvas_width, fundus_panel.height + oct_grid.height + 12), "white")
    canvas.paste(fundus_panel, ((canvas_width - fundus_panel.width) // 2, 0))
    canvas.paste(oct_grid, ((canvas_width - oct_grid.width) // 2, fundus_panel.height + 12))
    if canvas.width > max_width:
        new_height = round(canvas.height * max_width / canvas.width)
        canvas = canvas.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return canvas, [int(value) for value in indices]


def join_manifest(manifest, audit):
    manifest_file = find_column(manifest.columns, FILE_CANDIDATES)
    audit_file = find_column(audit.columns, FILE_CANDIDATES)
    manifest_case = find_column(manifest.columns, CASE_CANDIDATES, required=False)
    audit_case = find_column(audit.columns, CASE_CANDIDATES, required=False)

    if manifest_case is not None and audit_case is not None:
        manifest_keys = manifest[manifest_case].map(lambda value: clean_text(value, "").casefold())
        audit_keys = audit[audit_case].map(lambda value: clean_text(value, "").casefold())
        if manifest_keys.notna().all() and not audit_keys.duplicated().any():
            key_name = "_join_case"
            left, right = manifest.copy(), audit.copy()
            left[key_name], right[key_name] = manifest_keys, audit_keys
            right["_audit_match"] = True
            merged = left.merge(right, on=key_name, how="left", suffixes=("_manifest", "_audit"), validate="one_to_one")
            if len(merged) == len(manifest) and merged["_audit_match"].fillna(False).all():
                return merged, f"case_id:{manifest_case}->{audit_case}", manifest_file, audit_file

    left, right = manifest.copy(), audit.copy()
    left["_join_file"] = left[manifest_file].map(normalized_path)
    right["_join_file"] = right[audit_file].map(normalized_path)
    right["_audit_match"] = True
    if right["_join_file"].duplicated().any():
        duplicates = right.loc[right["_join_file"].duplicated(), audit_file].head(5).tolist()
        raise ValueError(f"Audit filename is not unique; examples: {duplicates}")
    merged = left.merge(right, on="_join_file", how="left", suffixes=("_manifest", "_audit"), validate="one_to_one")
    if not merged["_audit_match"].fillna(False).all():
        raise ValueError("Could not match every blinded manifest row to the private selected-rows audit")
    return merged, f"filename:{manifest_file}->{audit_file}", manifest_file, audit_file


def merged_column(frame, original, preferred_suffix="_audit"):
    for candidate in (f"{original}{preferred_suffix}", original, f"{original}_manifest"):
        if candidate in frame.columns:
            return candidate
    return None


def demographic_column(frame, candidates):
    for original in candidates:
        column = merged_column(frame, original)
        if column is not None:
            return column
    folded = {column.casefold(): column for column in frame.columns}
    for candidate in candidates:
        for variant in (candidate, f"{candidate}_audit", f"{candidate}_manifest"):
            if variant.casefold() in folded:
                return folded[variant.casefold()]
    return None


def display_age(value, mode):
    text = clean_text(value)
    if mode == "exact" or text == "Not available":
        return text
    try:
        age = int(float(text))
    except ValueError:
        return text
    lower = (age // 10) * 10
    return f"{lower}-{lower + 9}"


def main():
    args = parse_args()
    if args.oct_slices <= 0 or args.max_image_width < 400:
        raise ValueError("oct-slices must be positive and max-image-width must be at least 400")
    manifest = pd.read_csv(args.manifest)
    audit = pd.read_csv(args.selected_rows)
    if manifest.empty or audit.empty:
        raise ValueError("Manifest and selected-rows audit must both be non-empty")
    merged, join_method, manifest_file, audit_file = join_manifest(manifest, audit)

    age_col = demographic_column(merged, AGE_CANDIDATES)
    gender_col = demographic_column(merged, GENDER_CANDIDATES)
    race_col = demographic_column(merged, RACE_CANDIDATES)
    ethnicity_col = demographic_column(merged, ETHNICITY_CANDIDATES)
    truth_col = demographic_column(merged, TRUTH_CANDIDATES)
    if args.disease == "mixed":
        disease_col = demographic_column(merged, ("disease",))
        if disease_col is None:
            raise KeyError("A mixed manifest must contain a disease column")
        merged["_form_disease"] = merged[disease_col].map(
            lambda value: clean_text(value, "").casefold()
        )
        invalid_diseases = sorted(
            set(merged["_form_disease"]) - set(PREDICTION_CANDIDATES)
        )
        if invalid_diseases:
            raise ValueError(f"Unsupported diseases in mixed manifest: {invalid_diseases}")
        disease_counts = merged["_form_disease"].value_counts().to_dict()
        if disease_counts != {"glaucoma": 5, "amd": 5, "dr": 5}:
            raise ValueError(
                "A mixed reviewer manifest must contain exactly 5 glaucoma, "
                f"5 AMD, and 5 DR cases; found {disease_counts}"
            )
        prediction_columns = {
            disease: demographic_column(merged, candidates)
            for disease, candidates in PREDICTION_CANDIDATES.items()
        }
        missing_predictions = [
            disease for disease, column in prediction_columns.items() if column is None
        ]
        if missing_predictions:
            raise KeyError(
                "Selected-rows audit lacks prediction columns for: "
                + ", ".join(missing_predictions)
            )
    else:
        merged["_form_disease"] = args.disease
        prediction_columns = {
            args.disease: demographic_column(
                merged, PREDICTION_CANDIDATES[args.disease]
            )
        }
    missing_demo = [name for name, column in (
        ("Age", age_col), ("Gender", gender_col), ("Race", race_col),
        ("Ethnicity", ethnicity_col),
    ) if column is None]
    if missing_demo:
        raise KeyError(f"Selected-rows audit lacks demographic columns: {missing_demo}")
    if truth_col is None or any(column is None for column in prediction_columns.values()):
        raise KeyError(
            "Selected-rows audit must contain ground truth and the disease prediction "
            "to create the private evaluation key"
        )

    review_column = find_column(manifest.columns, ("review_order",), required=False)
    case_column = find_column(manifest.columns, CASE_CANDIDATES, required=False)
    joined_manifest_file = merged_column(merged, manifest_file, "_manifest")
    joined_audit_file = merged_column(merged, audit_file, "_audit")
    if joined_manifest_file is None or joined_audit_file is None:
        raise KeyError("Could not locate joined filename columns")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    disease_order = {"glaucoma": 0, "amd": 1, "dr": 2}
    if args.disease == "mixed":
        merged["_disease_order"] = merged["_form_disease"].map(disease_order)
        if review_column:
            merged["_original_review_order"] = pd.to_numeric(
                merged[review_column], errors="coerce"
            )
        else:
            merged["_original_review_order"] = np.arange(1, len(merged) + 1)
        merged = merged.sort_values(
            ["_disease_order", "_original_review_order"], kind="stable"
        ).reset_index(drop=True)

    output_rows = []
    private_rows = []
    seen_case_ids = set()
    for position, (_, row) in enumerate(merged.iterrows(), start=1):
        # Mixed bundles receive a new sequential order after grouping diseases.
        review_order = (
            position if args.disease == "mixed"
            else int(row[review_column]) if review_column else position
        )
        row_disease = row["_form_disease"]
        raw_case_id = row[case_column] if case_column else Path(str(row[joined_manifest_file])).stem
        case_id = safe_case_id(raw_case_id, f"case_{review_order:03d}")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id after blinding: {case_id}")
        seen_case_ids.add(case_id)
        npz_path = resolve_data_path(
            row[joined_audit_file], args.data_root, args.manifest, args.selected_rows
        )
        image_filename = f"case_{review_order:03d}.jpg"
        montage, indices = make_montage(npz_path, args.oct_slices, args.max_image_width)
        montage.save(image_dir / image_filename, format="JPEG", quality=92, optimize=True)
        output_rows.append({
            "review_order": review_order,
            "case_id": case_id,
            "image_filename": image_filename,
            "age": display_age(row[age_col], args.age_display),
            "gender": clean_text(row[gender_col]),
            "race": clean_text(row[race_col]),
            "ethnicity": clean_text(row[ethnicity_col]),
            "disease": row_disease,
            "imaging_modality": "SLO/fundus photograph and OCT B-scans",
            "oct_slice_indices": json.dumps(indices),
        })
        truth = int(float(row[truth_col]))
        prediction = int(float(row[prediction_columns[row_disease]]))
        if truth not in (0, 1) or prediction not in (0, 1):
            raise ValueError(
                f"Case {case_id} has non-binary truth/prediction: {truth}/{prediction}"
            )
        private_rows.append({
            "review_order": review_order,
            "case_id": case_id,
            "source_filename": str(row[joined_audit_file]),
            "ground_truth": truth,
            "model_prediction": prediction,
        })

    output_rows.sort(key=lambda item: item["review_order"])
    output_csv = args.output_dir / "form_cases.csv"
    fieldnames = list(output_rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # This key is required for analysis but must never be uploaded with the
    # Google Form bundle or shared with reviewers.
    private_dir = args.output_dir / "_private_do_not_upload"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_key = private_dir / "evaluation_key.csv"
    pd.DataFrame(private_rows).sort_values("review_order").to_csv(private_key, index=False)

    forbidden = [column for column in fieldnames
                 if any(term in column.casefold() for term in FORBIDDEN_OUTPUT_TERMS)]
    if forbidden:
        raise AssertionError(f"Form bundle leaked forbidden columns: {forbidden}")
    if len(output_rows) != len(manifest):
        raise AssertionError("Output row count differs from blinded manifest")
    summary = {
        "manifest": str(args.manifest),
        "selected_rows_private_source": str(args.selected_rows),
        "disease": args.disease,
        "cases": len(output_rows),
        "join_method": join_method,
        "age_display": args.age_display,
        "oct_slices": args.oct_slices,
        "output_csv": str(output_csv),
        "image_directory": str(image_dir),
        "private_evaluation_key_do_not_upload": str(private_key),
        "doctor_bundle_contains_ground_truth_or_predictions": False,
    }
    (args.output_dir / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Upload form_cases.csv and all files inside images/ to one Google Drive folder.")
    print("Do NOT upload _private_do_not_upload/evaluation_key.csv; keep it with the study team.")


if __name__ == "__main__":
    main()
