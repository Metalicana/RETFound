"""PAPILA glaucoma LLM baseline: GPT-5.1, GPT-5.6-luna, or Claude Haiku 4.5.

Single CFP fundus image plus age/gender demographics. No RETFound
probability, CDR, or other specialist-agent output is supplied, matching the
Drishti-GS and REFUGE2 LLM-baseline protocol used elsewhere in this repo.

Select the model with the LLM_MODEL environment variable:

    LLM_MODEL=gpt-5.1          python -m evaluate_papila_llm_baseline
    LLM_MODEL=gpt-5.6-luna     python -m evaluate_papila_llm_baseline
    LLM_MODEL=claude-haiku-4-5 python -m evaluate_papila_llm_baseline

Deliberately withheld: vertical CDR, pachymetry, axial length, refraction,
and visual-field MD. The PAPILA manifest carries all of these, but they are
structural/functional glaucoma risk factors rather than neutral demographics
-- passing them here would turn this into a leaked-feature baseline rather
than a vision-only one (see the MD/VF label-circularity warning in
update.md). Only age and gender are passed as demographic context.

Only binary-labeled eyes (healthy / glaucoma present) are evaluated. Suspect
eyes are excluded by prepare_papila.py into suspect_manifest.csv and never
reach this script.
"""

import csv
import os
from pathlib import Path

from PIL import Image

from llm_baseline_utils import (
    ClaudeGlaucomaBaseline,
    GPT51GlaucomaBaseline,
    GPT56GlaucomaBaseline,
    checkpoint_and_report,
    print_metrics,
)


MANIFEST = Path(os.getenv("PAPILA_MANIFEST", "./data_papila/manifest.csv"))
DATA_ROOT = Path(os.getenv("PAPILA_DATA_ROOT", "."))
SPLIT = os.getenv("PAPILA_SPLIT", "test")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
IMAGE_DESCRIPTION = "One color fundus photograph (PAPILA)."

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
        raise FileNotFoundError(f"PAPILA manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cases = [
        row for row in rows
        if row.get("split", "").strip().lower() == SPLIT.lower()
        and row.get("label", "").strip() != ""
    ]
    if not cases:
        raise ValueError(f"No PAPILA {SPLIT!r} binary-labeled cases found in {MANIFEST}")
    if MAX_CASES > 0:
        cases = cases[:MAX_CASES]
    return cases


def resolve(relative_path):
    path = Path(relative_path)
    return path if path.is_absolute() else DATA_ROOT / path


def main():
    cases, (evaluator, model_tag) = load_cases(), build_evaluator()
    output_csv = os.getenv("OUTPUT_CSV", f"papila_{model_tag}_predictions.csv")
    print(f"Evaluating {evaluator.deployment!r} on {len(cases)} PAPILA {SPLIT} cases; output={output_csv}")

    rows = []
    for index, case in enumerate(cases, start=1):
        print("\n" + "=" * 90 + f"\nCASE {index}/{len(cases)} | {case['case_id']}")
        image_path = resolve(case["cfp_path"])
        truth = int(float(case["label"]))
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            demographics = {"age": case.get("age", ""), "gender": case.get("gender_code", "")}
            prediction, confidence, reasoning, raw = evaluator.analyze(
                [image], demographics, IMAGE_DESCRIPTION
            )
            print(f"Ground truth: {truth} | Prediction: {prediction} | Confidence: {confidence}")
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == truth}")
            row = {
                "Case_ID": case["case_id"], "Patient_ID": case.get("patient_id", ""),
                "Eye": case.get("eye", ""), "Filename": str(image_path), "Model": evaluator.deployment,
                "Age": case.get("age", ""), "Gender": case.get("gender_code", ""),
                "Ground_Truth": truth, "Pred_GL": prediction, "Confidence": confidence,
                "Reasoning": reasoning, "Raw_Response": raw,
                "Is_Correct": int(prediction == truth),
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            row = {
                "Case_ID": case["case_id"], "Patient_ID": case.get("patient_id", ""),
                "Eye": case.get("eye", ""), "Filename": str(image_path), "Model": evaluator.deployment,
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
