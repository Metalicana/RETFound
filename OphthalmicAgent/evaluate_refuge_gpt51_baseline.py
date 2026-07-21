"""Single-call GPT-5.1 CFP baseline on the REFUGE Test split."""

import os
from pathlib import Path

import pandas as pd
from PIL import Image

from llm_baseline_utils import GPT51GlaucomaBaseline, checkpoint_and_report, print_metrics


DATA_ROOT = Path(os.getenv("REFUGE_DATA_ROOT", "./"))
CSV_PATH = Path(os.getenv("REFUGE_CSV", "./data_refuge/data.csv"))
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "refuge_test_gpt51_baseline_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))


def cases():
    frame = pd.read_csv(CSV_PATH)
    frame = frame[frame["filename"].astype(str).str.contains(r"[\\/]Test[\\/]", case=False, regex=True)]
    if MAX_CASES > 0: frame = frame.head(MAX_CASES)
    return frame.reset_index(drop=True)


def main():
    frame, evaluator, rows = cases(), GPT51GlaucomaBaseline(), []
    print(f"Evaluating GPT-5.1 baseline on {len(frame)} REFUGE Test CFPs")
    for index, row in frame.iterrows():
        relative = Path(str(row["filename"])); path = relative if relative.is_absolute() else DATA_ROOT / relative
        print("\n" + "=" * 90 + f"\nCASE {index + 1}/{len(frame)} | {path}")
        try:
            prediction, confidence, reasoning, raw = evaluator.analyze(
                [Image.open(path).convert("RGB")], {"status": "not available in REFUGE manifest"},
                "One color fundus photograph. Focus on optic-disc cupping, rim thinning/notching, vessel changes, disc hemorrhage, and image quality.",
            )
            truth = int(row["Ground_Truth"])
            print(f"Ground truth: {truth} | Prediction: {prediction} | Confidence: {confidence}")
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == truth}")
            result = {"Filename": str(path), "Ground_Truth": truth, "Pred_GL": prediction,
                      "Confidence": confidence, "Reasoning": reasoning, "Raw_Response": raw,
                      "Is_Correct": int(prediction == truth)}
        except Exception as exc:
            print(f"!!! Error: {exc}")
            result = {"Filename": str(path), "Ground_Truth": int(row["Ground_Truth"]),
                      "Pred_GL": -1, "Is_Correct": -1}
        rows.append(result); checkpoint_and_report(rows, OUTPUT_CSV)
    results = checkpoint_and_report(rows, OUTPUT_CSV); print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__": main()
