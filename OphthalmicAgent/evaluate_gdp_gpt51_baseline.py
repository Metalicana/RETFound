"""Single-call GPT-5.1 baseline on GDP Test: OCT + RNFLT + demographics."""

import os

from data.gdp_loader import GDPTestLoader
from llm_baseline_utils import (
    GPT51GlaucomaBaseline, checkpoint_and_report, make_oct_grid, print_metrics, render_rnflt,
)


CSV_PATH = os.getenv("GDP_CSV", "./data_gdp/data_summary.csv")
BSCAN_DIR = os.getenv("GDP_BSCAN_DIR", "./data_gdp/BScan")
RNFLT_DIR = os.getenv("GDP_RNFLT_DIR", "./data_gdp/RNFLT")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "gdp_test_gpt51_baseline_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
OCT_SLICES = int(os.getenv("OCT_SLICES", "8"))


def main():
    loader = GDPTestLoader(CSV_PATH, BSCAN_DIR, RNFLT_DIR, OCT_SLICES, MAX_CASES, require_rnflt=True)
    evaluator, rows = GPT51GlaucomaBaseline(), []
    print(f"Evaluating GPT-5.1 baseline on {len(loader)} GDP Test cases")
    for index in range(len(loader)):
        case = None
        print("\n" + "=" * 90 + f"\nCASE {index + 1}/{len(loader)}")
        try:
            case = loader.load(index)
            oct_grid = make_oct_grid(case["oct_slices"], OCT_SLICES)
            rnflt_image, rnflt_stats = render_rnflt(case["rnflt"])
            demographics = {"age": case["age"], "gender": case["gender"], "race": case["race"]}
            prediction, confidence, reasoning, raw = evaluator.analyze(
                [oct_grid, rnflt_image], demographics,
                "First image: eight evenly spaced OCT B-scans. Second image: RNFL thickness map.",
                f"RNFLT case-scaled colorbar statistics: {rnflt_stats}",
            )
            print(f"Patient: {case['patient_id']} | Demographics: {demographics}")
            print(f"Ground truth: {case['ground_truth']} | Prediction: {prediction} | Confidence: {confidence}")
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == case['ground_truth']}")
            row = {"Patient_ID": case["patient_id"], "Age": case["age"], "Gender": case["gender"],
                   "Race": case["race"], "Ground_Truth": case["ground_truth"], "Pred_GL": prediction,
                   "Confidence": confidence, "Reasoning": reasoning, "Raw_Response": raw,
                   "Is_Correct": int(prediction == case["ground_truth"])}
        except Exception as exc:
            print(f"!!! Error: {exc}")
            row = {"Patient_ID": case["patient_id"] if case else f"index_{index}",
                   "Ground_Truth": case["ground_truth"] if case else -1, "Pred_GL": -1, "Is_Correct": -1}
        rows.append(row); checkpoint_and_report(rows, OUTPUT_CSV)
    results = checkpoint_and_report(rows, OUTPUT_CSV); print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__": main()
