"""Single-call GPT-5.1 CFP baseline on Drishti Normal/Glaucoma folders."""

import os
from pathlib import Path

from PIL import Image

from llm_baseline_utils import GPT51GlaucomaBaseline, checkpoint_and_report, print_metrics


DATA_ROOT = Path(os.getenv("DRISHTI_DATA_ROOT", "./data_drishti"))
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "drishti_gpt51_baseline_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_folder(name):
    match = next((path for path in DATA_ROOT.iterdir() if path.is_dir() and path.name.lower() == name), None)
    if match is None: raise FileNotFoundError(f"Missing {name} folder under {DATA_ROOT}")
    return match


def cases():
    found = []
    for name, label in (("normal", 0), ("glaucoma", 1)):
        found.extend({"path": path, "ground_truth": label} for path in sorted(find_folder(name).rglob("*"))
                     if path.is_file() and path.suffix.lower() in EXTENSIONS)
    return found[:MAX_CASES] if MAX_CASES > 0 else found


def main():
    dataset, evaluator, rows = cases(), GPT51GlaucomaBaseline(), []
    print(f"Evaluating GPT-5.1 baseline on {len(dataset)} Drishti CFPs")
    for index, case in enumerate(dataset, start=1):
        print("\n" + "=" * 90 + f"\nCASE {index}/{len(dataset)} | {case['path']}")
        try:
            prediction, confidence, reasoning, raw = evaluator.analyze(
                [Image.open(case["path"]).convert("RGB")], {"status": "not available in Drishti folders"},
                "One color fundus photograph. Focus on optic-disc cupping, rim thinning/notching, vessel changes, disc hemorrhage, and image quality.",
            )
            print(f"Ground truth: {case['ground_truth']} | Prediction: {prediction} | Confidence: {confidence}")
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == case['ground_truth']}")
            row = {"Filename": str(case["path"]), "Ground_Truth": case["ground_truth"],
                   "Pred_GL": prediction, "Confidence": confidence, "Reasoning": reasoning,
                   "Raw_Response": raw, "Is_Correct": int(prediction == case["ground_truth"])}
        except Exception as exc:
            print(f"!!! Error: {exc}")
            row = {"Filename": str(case["path"]), "Ground_Truth": case["ground_truth"],
                   "Pred_GL": -1, "Is_Correct": -1}
        rows.append(row); checkpoint_and_report(rows, OUTPUT_CSV)
    results = checkpoint_and_report(rows, OUTPUT_CSV); print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__": main()
