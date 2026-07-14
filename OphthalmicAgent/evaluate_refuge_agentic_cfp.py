"""Evaluate the CFP agentic glaucoma pipeline on the REFUGE Test split."""

import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from CounterfactualAgent.counterfactual_cfp import CounterfactualCFPAgent
from Orchestrator.drishti import DrishtiOrchestrator
from VisionAgent.vision_cfp import VisionSpecialistCFP


DATA_ROOT = Path(os.getenv("REFUGE_DATA_ROOT", "./"))
CSV_PATH = Path(os.getenv("REFUGE_CSV", "./data_refuge/data.csv"))
CFP_WEIGHTS = os.getenv("CFP_WEIGHTS", "./weights/cfp_glaucoma_best.pth")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "refuge_test_agentic_cfp_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))

os.environ.setdefault("CDR_OUTPUT_DIR", "outputs/refuge_cfp/cdr_segmentations")

vision_agent = VisionSpecialistCFP(CFP_WEIGHTS)
counterfactual_agent = CounterfactualCFPAgent(
    cache_path=os.getenv(
        "COUNTERFACTUAL_CACHE_PATH",
        "outputs/refuge_cfp/counterfactual_traces.jsonl",
    )
)
orchestrator = DrishtiOrchestrator()


def load_test_cases():
    frame = pd.read_csv(CSV_PATH)
    required = {"filename", "Ground_Truth"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"REFUGE CSV is missing columns: {sorted(missing)}")
    frame = frame[
        frame["filename"].astype(str).str.contains(
            r"[\\/]Test[\\/]", case=False, regex=True
        )
    ].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No /Test/ rows found in {CSV_PATH}")
    if MAX_CASES > 0:
        frame = frame.head(MAX_CASES)
    cases = []
    for _, row in frame.iterrows():
        relative_path = Path(str(row["filename"]))
        image_path = relative_path if relative_path.is_absolute() else DATA_ROOT / relative_path
        cases.append({
            "path": image_path,
            "ground_truth": int(row["Ground_Truth"]),
        })
    return cases


def initialize_state(case):
    return {
        "patient_id": case["path"].name,
        "image_path": str(case["path"]),
        "cfp_diagnosis": None,
        "retfound_scores": "",
        "vision_opinion_cfp": "",
        "vertical_cdr": None,
        "cdr_segmentation_path": None,
        "counterfactual_trace": {},
        "final_diagnosis": {},
    }


def parse_label(state):
    match = re.search(
        r"GLAUCOMA_DETECTED:\s*(-?\d+)",
        state["final_diagnosis"].get("labels", ""),
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else -1


def run_case(case):
    state = initialize_state(case)
    (
        state["retfound_scores"],
        state["vision_opinion_cfp"],
        probability,
        state["vertical_cdr"],
    ) = vision_agent.analyze(case["path"], state)

    audit = counterfactual_agent.analyze(
        case_id=state["patient_id"],
        retfound_probability=probability,
        cfp_report=state["vision_opinion_cfp"],
        cdr=state["vertical_cdr"],
    )
    state["counterfactual_trace"] = counterfactual_agent.concise_trace(audit)
    state["final_diagnosis"] = orchestrator.analyze(
        probability,
        state["vision_opinion_cfp"],
        state["vertical_cdr"],
        state["counterfactual_trace"],
    )
    return state, probability, parse_label(state)


def print_metrics(results):
    valid = results[results["Pred_GL"].isin([0, 1])]
    failed = len(results) - len(valid)
    if valid.empty:
        print("No valid agentic predictions were produced.")
        return
    y_true = valid["Ground_Truth"].to_numpy()
    y_pred = valid["Pred_GL"].to_numpy()
    print("\nPer-class classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Glaucoma"],
            digits=4,
            zero_division=0,
        )
    )
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(f"Valid predictions: {len(valid)} | Failed/invalid predictions: {failed}")


def main():
    cases = load_test_cases()
    print(f"Evaluating agentic CFP pipeline on {len(cases)} REFUGE Test images")
    rows = []
    for index, case in enumerate(cases, start=1):
        try:
            state, probability, prediction = run_case(case)
            row = {
                "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"],
                "RETFound_Probability_GL": probability / 100.0,
                "Vertical_CDR": state["vertical_cdr"],
                "CFP_Report": state["vision_opinion_cfp"],
                "Agentic_Decision": state["final_diagnosis"].get("decision", ""),
                "Pred_GL": prediction,
                "Is_Correct": int(prediction == case["ground_truth"]) if prediction != -1 else -1,
            }
        except Exception as exc:
            print(f"!!! Error processing {case['path']}: {exc}")
            row = {
                "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"],
                "RETFound_Probability_GL": None,
                "Vertical_CDR": None,
                "CFP_Report": "",
                "Agentic_Decision": "",
                "Pred_GL": -1,
                "Is_Correct": -1,
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        print(f"Completed {index}/{len(cases)}")

    results = pd.DataFrame(rows)
    print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
