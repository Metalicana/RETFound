"""Evaluate GDP Test with RETFound-OCT, OCT GPT, RNFLT GPT, and final GPT-5.1."""

import os
import re

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from data.gdp_loader import GDPTestLoader
from Orchestrator.gdp import GDPOrchestrator
from VisionAgent.vision_oct import VisionSpecialistOct
from VisionAgent.vision_rnflt import RNFLTSpecialist


CSV_PATH = os.getenv("GDP_CSV", "./data_gdp/data_summary.csv")
BSCAN_DIR = os.getenv("GDP_BSCAN_DIR", "./data_gdp/BScan")
RNFLT_DIR = os.getenv("GDP_RNFLT_DIR", "./data_gdp/RNFLT")
OCT_WEIGHTS = os.getenv("OCT_WEIGHTS", "./weights/oct_model_8_slices_not_center.pth")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "gdp_test_agentic_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
OCT_SLICES = int(os.getenv("OCT_SLICES", "8"))


oct_agent = VisionSpecialistOct(OCT_WEIGHTS)
rnflt_agent = RNFLTSpecialist()
orchestrator = GDPOrchestrator()


def parse_label(result):
    match = re.search(
        r"GLAUCOMA_DETECTED:\s*(-?\d+)", result.get("labels", ""), re.IGNORECASE
    )
    return int(match.group(1)) if match else -1


def print_metrics(results):
    valid = results[results["Pred_GL"].isin([0, 1])]
    failed = len(results) - len(valid)
    if valid.empty:
        print("No valid predictions were produced.")
        return
    y_true = valid["Ground_Truth"].astype(int).to_numpy()
    y_pred = valid["Pred_GL"].astype(int).to_numpy()
    print("\nPer-class classification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Normal", "Glaucoma"],
        digits=4, zero_division=0,
    ))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(f"Valid predictions: {len(valid)} | Failed/invalid predictions: {failed}")


def main():
    loader = GDPTestLoader(
        CSV_PATH,
        BSCAN_DIR,
        RNFLT_DIR,
        OCT_SLICES,
        MAX_CASES,
        require_rnflt=True,
    )
    rows = []
    print(f"Evaluating GDP agentic pipeline on {len(loader)} Test cases")
    for index in range(len(loader)):
        case = None
        print("\n" + "=" * 90)
        print(f"CASE {index + 1}/{len(loader)}")
        try:
            case = loader.load(index)
            demographics = {
                "age": case["age"],
                "gender": case["gender"],
                "race": case["race"],
            }
            state = {"oct_diagnosis": None}
            retfound_scores, oct_report = oct_agent.analyze(
                case["oct_slices"], case["middle_oct"], state
            )
            probability = state["oct_diagnosis"]["Glaucoma"]["Prob_Pct"]
            rnflt_report, rnflt_statistics = rnflt_agent.analyze(case["rnflt"])
            final_result = orchestrator.analyze(
                demographics, probability, oct_report, rnflt_report, rnflt_statistics
            )
            prediction = parse_label(final_result)

            print(f"Patient ID: {case['patient_id']}")
            print(f"Demographics: {demographics}")
            print(f"Ground truth: {case['ground_truth']}")
            print(f"\n{retfound_scores}")
            print(f"OCT specialist report:\n{oct_report}")
            print(f"\nRNFLT statistics: {rnflt_statistics}")
            print(f"RNFLT specialist report:\n{rnflt_report}")
            print(f"\nFinal orchestrator output:\n{final_result['decision']}")
            print(f"Parsed prediction: {prediction}")
            print(f"Correct: {prediction == case['ground_truth']}")

            row = {
                "Patient_ID": case["patient_id"],
                "BScan_Path": str(case["bscan_path"]),
                "RNFLT_Path": str(case["rnflt_path"]),
                "Age": case["age"],
                "Gender": case["gender"],
                "Race": case["race"],
                "Ground_Truth": case["ground_truth"],
                "RETFound_Probability_GL": probability / 100.0,
                "OCT_Report": oct_report,
                "RNFLT_Report": rnflt_report,
                "RNFLT_Statistics": str(rnflt_statistics),
                "Final_Decision": final_result["decision"],
                "Pred_GL": prediction,
                "Is_Correct": int(prediction == case["ground_truth"]) if prediction != -1 else -1,
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            row = {
                "Patient_ID": case["patient_id"] if case else f"index_{index}",
                "Ground_Truth": case["ground_truth"] if case else -1,
                "Pred_GL": -1,
                "Is_Correct": -1,
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        print(f"Checkpoint saved: {OUTPUT_CSV}")

    results = pd.DataFrame(rows)
    print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
