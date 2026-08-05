"""GAMMA glaucoma LLM baseline: GPT-5.1, GPT-5.6-luna, or Claude Haiku 4.5.

Paired CFP fundus photograph plus eight representative OCT B-scans from the
3D volume. GAMMA carries no demographic metadata, so none is passed. No
RETFound probability, CDR, or other specialist-agent output is supplied,
matching the Drishti-GS and REFUGE2 LLM-baseline protocol used elsewhere in
this repo.

Select the model with the LLM_MODEL environment variable:

    LLM_MODEL=gpt-5.1          python -m evaluate_gamma_llm_baseline
    LLM_MODEL=gpt-5.6-luna     python -m evaluate_gamma_llm_baseline
    LLM_MODEL=claude-haiku-4-5 python -m evaluate_gamma_llm_baseline
"""

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from llm_baseline_utils import (
    ClaudeGlaucomaBaseline,
    GPT51GlaucomaBaseline,
    GPT56GlaucomaBaseline,
    checkpoint_and_report,
    make_oct_grid,
    print_metrics,
)


MANIFEST = Path(os.getenv("GAMMA_MANIFEST", "./data_gamma/manifest.csv"))
DATA_ROOT = Path(os.getenv("GAMMA_DATA_ROOT", "."))
SPLIT = os.getenv("GAMMA_SPLIT", "test")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
OCT_SLICES = int(os.getenv("LLM_OCT_SLICES", "8"))
IMAGE_DESCRIPTION = (
    "First image: color fundus photograph. Second image: eight evenly spaced "
    "OCT B-scans sampled from a 3D volume."
)

EVALUATORS = {
    "gpt-5.1": (GPT51GlaucomaBaseline, "gpt51"),
    "gpt-5.6-luna": (GPT56GlaucomaBaseline, "gpt56"),
    "claude-haiku-4-5": (ClaudeGlaucomaBaseline, "claude45"),
}


def build_evaluator():
    key = os.getenv("LLM_MODEL", "gpt-5.1").strip().lower()
    if key not in EVALUATORS:
        raise ValueError(
            f"Unknown LLM_MODEL={key!r}; expected one of {sorted(EVALUATORS)}"
        )
    evaluator_cls, model_tag = EVALUATORS[key]
    return evaluator_cls(), model_tag


def load_cases():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"GAMMA manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cases = [row for row in rows if row.get("split", "").strip().lower() == SPLIT.lower()]
    if not cases:
        raise ValueError(f"No GAMMA {SPLIT!r} cases found in {MANIFEST}")
    if MAX_CASES > 0:
        cases = cases[:MAX_CASES]
    return cases


def resolve(relative_path):
    path = Path(relative_path)
    return path if path.is_absolute() else DATA_ROOT / path


# --- OCT volume loading, mirrored from scripts/train_retfound_external_glaucoma.py ---
# (kept local rather than imported so this script has no dependency on the
# scripts/ package layout; keep the two in sync if the MHD reader changes.)

def load_mhd_volume(path):
    header = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    required = {"DimSize", "ElementType", "ElementDataFile"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"Missing MHD fields {sorted(missing)}: {path}")
    dimensions = [int(value) for value in header["DimSize"].split()]
    if len(dimensions) != 3:
        raise ValueError(f"Expected three MHD dimensions, got {dimensions}: {path}")
    data_types = {
        "MET_UCHAR": np.uint8, "MET_CHAR": np.int8,
        "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
        "MET_UINT": np.uint32, "MET_INT": np.int32,
        "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64,
    }
    if header["ElementType"] not in data_types:
        raise ValueError(f"Unsupported MHD ElementType {header['ElementType']!r}: {path}")
    data_type = np.dtype(data_types[header["ElementType"]])
    if header.get("ElementByteOrderMSB", "False").lower() == "true":
        data_type = data_type.newbyteorder(">")
    raw_path = path.parent / header["ElementDataFile"]
    values = np.fromfile(raw_path, dtype=data_type)
    expected = int(np.prod(dimensions))
    if values.size != expected:
        raise ValueError(f"Expected {expected} MHD voxels, found {values.size}: {raw_path}")
    # MetaImage dimensions are X Y Z; model input is Z Y X.
    return values.reshape(tuple(reversed(dimensions)))


def load_oct_slices(path, count):
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )
        if not files:
            raise ValueError(f"No B-scan images found in {path}")
        arrays = [np.asarray(Image.open(item).convert("L")) for item in files]
        volume = np.stack(arrays)
    elif path.suffix.lower() == ".mhd":
        volume = load_mhd_volume(path)
    else:
        raise ValueError(f"Unsupported GAMMA OCT input: {path}")
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected OCT volume [slices,height,width], got {volume.shape} at {path}")
    indices = np.linspace(0, volume.shape[0] - 1, count, dtype=int)
    return [volume[index] for index in indices]


def main():
    cases, (evaluator, model_tag) = load_cases(), build_evaluator()
    output_csv = os.getenv("OUTPUT_CSV", f"gamma_{model_tag}_predictions.csv")
    print(f"Evaluating {evaluator.deployment!r} on {len(cases)} GAMMA {SPLIT} cases; output={output_csv}")

    rows = []
    for index, case in enumerate(cases, start=1):
        print("\n" + "=" * 90 + f"\nCASE {index}/{len(cases)} | {case['case_id']}")
        cfp_path = resolve(case["cfp_path"])
        oct_path = resolve(case["oct_path"])
        truth = int(float(case["label"]))
        try:
            with Image.open(cfp_path) as source_image:
                fundus = source_image.convert("RGB")
            oct_slices = load_oct_slices(oct_path, OCT_SLICES)
            oct_grid = make_oct_grid(oct_slices, OCT_SLICES)
            prediction, confidence, reasoning, raw = evaluator.analyze(
                [fundus, oct_grid], {}, IMAGE_DESCRIPTION
            )
            print(f"Ground truth: {truth} | Prediction: {prediction} | Confidence: {confidence}")
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == truth}")
            row = {
                "Case_ID": case["case_id"], "Grade": case.get("grade", ""),
                "CFP_Filename": str(cfp_path), "OCT_Path": str(oct_path), "Model": evaluator.deployment,
                "Ground_Truth": truth, "Pred_GL": prediction, "Confidence": confidence,
                "Reasoning": reasoning, "Raw_Response": raw,
                "Is_Correct": int(prediction == truth),
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            row = {
                "Case_ID": case["case_id"], "Grade": case.get("grade", ""),
                "CFP_Filename": str(cfp_path), "OCT_Path": str(oct_path), "Model": evaluator.deployment,
                "Ground_Truth": truth, "Pred_GL": -1, "Confidence": "", "Reasoning": "",
                "Raw_Response": "", "Is_Correct": -1, "Error": str(exc),
            }
        rows.append(row)
        checkpoint_and_report(rows, output_csv)
        print(f"Checkpoint saved: {output_csv}")

    print_metrics(checkpoint_and_report(rows, output_csv))
    print(f"\nPredictions saved to {output_csv}")


if __name__ == "__main__":
    main()
