"""Evaluate the Drishti Test split with RETFound-CFP and the GPT-5.1 pipeline."""

import json
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from CounterfactualAgent.counterfactual_cfp import CounterfactualCFPAgent
from Orchestrator.drishti import DrishtiOrchestrator
from VisionAgent.vision_cfp import VisionSpecialistCFP


DATA_ROOT = Path(os.getenv("DRISHTI_DATA_ROOT", "./data_drishti"))
MANIFEST = Path(os.getenv("DRISHTI_MANIFEST", DATA_ROOT / "manifest.csv"))
CFP_WEIGHTS = os.getenv(
    "CFP_WEIGHTS", "./weights/drishti_cfp_glaucoma_best.pth"
)
TRAINING_METADATA = Path(os.getenv(
    "DRISHTI_CFP_TRAINING_METADATA", "drishti_cfp_training_metadata.json"
))
OUTPUT_CSV = os.getenv(
    "OUTPUT_CSV", "drishti_test_agentic_cfp_predictions.csv"
)
MAX_CASES = int(os.getenv("MAX_CASES", "0"))

# CDR is calculated and used, but overlay images are not saved during testing.
os.environ.setdefault("SAVE_CDR_SEGMENTATIONS", "0")
# Vision specialist, counterfactual agent, and final orchestrator all read this.
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")


def load_threshold():
    if "CFP_THRESHOLD" in os.environ:
        threshold = float(os.environ["CFP_THRESHOLD"])
    elif TRAINING_METADATA.is_file():
        metadata = json.loads(TRAINING_METADATA.read_text(encoding="utf-8"))
        threshold = metadata.get(
            "selected_validation_threshold",
            metadata.get("fixed_threshold", 0.5),
        )
    else:
        print(f"Training metadata not found at {TRAINING_METADATA}; using 0.5")
        threshold = 0.5
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"CFP threshold must be in [0, 1], got {threshold}")
    return threshold


def resolve_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    repo_candidate = MANIFEST.parent.parent / path
    return repo_candidate if repo_candidate.exists() else MANIFEST.parent / path


def load_test_cases():
    frame = pd.read_csv(MANIFEST)
    required = {"case_id", "split", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Drishti manifest is missing columns: {sorted(missing)}")
    frame = frame[frame["split"].astype(str).str.strip().str.lower() == "test"].copy()
    if frame.empty:
        raise ValueError(f"No Test cases found in {MANIFEST}")
    if MAX_CASES > 0:
        frame = frame.head(MAX_CASES)
    cases = []
    for _, row in frame.iterrows():
        value = str(row.get("cfp_path", "")).strip()
        if value and value.lower() != "nan":
            image_path = resolve_path(value)
        else:
            folder = "Glaucoma" if int(row["label"]) == 1 else "Normal"
            image_path = DATA_ROOT / folder / f"{row['case_id']}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing CFP for {row['case_id']}: {image_path}")
        cases.append({
            "case_id": str(row["case_id"]),
            "path": image_path,
            "ground_truth": int(row["label"]),
        })
    return cases


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
    truth = valid["Ground_Truth"].astype(int).to_numpy()
    predictions = valid["Pred_GL"].astype(int).to_numpy()
    print("\nPer-class metrics (precision, recall, F1, support):")
    print(classification_report(
        truth, predictions, labels=[0, 1], target_names=["Normal", "Glaucoma"],
        digits=4, zero_division=0,
    ))
    print(f"Accuracy: {accuracy_score(truth, predictions):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(truth, predictions, labels=[0, 1]))
    print(f"Valid predictions: {len(valid)} | Failed/invalid: {failed}")


def main():
    threshold = load_threshold()
    cases = load_test_cases()
    vision_agent = VisionSpecialistCFP(CFP_WEIGHTS, threshold=threshold)
    counterfactual_agent = CounterfactualCFPAgent(cache_path=os.getenv(
        "COUNTERFACTUAL_CACHE_PATH",
        "outputs/drishti_cfp/test_counterfactual_traces.jsonl",
    ))
    orchestrator = DrishtiOrchestrator()
    print(
        f"Evaluating {len(cases)} Drishti Test cases with {CFP_WEIGHTS}\n"
        f"GPT deployment: {os.environ['AZURE_OPENAI_DEPLOYMENT']} | "
        f"RETFound threshold: {threshold:.6f}"
    )
    rows = []
    for index, case in enumerate(cases, start=1):
        print("\n" + "=" * 90)
        print(f"CASE {index}/{len(cases)} | {case['case_id']} | {case['path']}")
        print(f"Ground truth: {case['ground_truth']}")
        try:
            state = {"patient_id": case["case_id"], "cfp_diagnosis": None}
            scores, cfp_report, probability, vertical_cdr = vision_agent.analyze(
                case["path"], state
            )
            audit = counterfactual_agent.analyze(
                case_id=case["case_id"], retfound_probability=probability,
                cfp_report=cfp_report, cdr=vertical_cdr,
            )
            trace = counterfactual_agent.concise_trace(audit)
            final = orchestrator.analyze(
                probability, cfp_report, vertical_cdr, trace, threshold=threshold
            )
            prediction = parse_label(final)
            print(f"\n{scores}")
            print(f"Model status at threshold: {state['cfp_diagnosis']['Glaucoma']['Status']}")
            print(f"Vertical CDR: {vertical_cdr if vertical_cdr is not None else 'Not Available'}")
            print(f"\nCFP specialist report:\n{cfp_report}")
            print("\nCounterfactual evidence-ablation trace:")
            print(json.dumps(trace, indent=2, default=str))
            print(f"\nFinal GPT-5.1 output:\n{final.get('decision', '')}")
            print(f"Parsed prediction: {prediction} | Correct: {prediction == case['ground_truth']}")
            row = {
                "case_id": case["case_id"], "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"],
                "RETFound_Probability_GL": probability / 100.0,
                "RETFound_Threshold": threshold, "Vertical_CDR": vertical_cdr,
                "CFP_Report": cfp_report,
                "Counterfactual_Trace": json.dumps(trace, default=str),
                "Agentic_Decision": final.get("decision", ""),
                "Pred_GL": prediction,
                "Is_Correct": int(prediction == case["ground_truth"]) if prediction in (0, 1) else -1,
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            row = {
                "case_id": case["case_id"], "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"], "Pred_GL": -1,
                "Is_Correct": -1, "Error": str(exc),
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        print(f"Completed {index}/{len(cases)} | Checkpoint: {OUTPUT_CSV}")

    results = pd.DataFrame(rows)
    print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
