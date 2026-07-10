"""CFP-only glaucoma pipeline for a Drishti normal/glaucoma folder dataset."""

import os
import re
from pathlib import Path
import pandas as pd

from CounterfactualAgent.counterfactual_cfp import CounterfactualCFPAgent
from Orchestrator.drishti import DrishtiOrchestrator
from VisionAgent.vision_cfp import VisionSpecialistCFP


DATA_ROOT = Path(os.getenv("DRISHTI_DATA_ROOT", "./data_drishti"))
CFP_WEIGHTS = os.getenv("CFP_WEIGHTS", "./cfp_glaucoma_best.pth")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "ophthalmic_performance_results_drishti_cfp.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

vision_agent = VisionSpecialistCFP(CFP_WEIGHTS)
counterfactual_agent = CounterfactualCFPAgent(
    cache_path=os.getenv(
        "COUNTERFACTUAL_CACHE_PATH",
        "outputs/drishti_cfp/counterfactual_traces.jsonl",
    )
)
ophthalmic_agent = DrishtiOrchestrator()


def discover_cases(root):
    """Return images from ``normal`` and ``glaucoma`` folders recursively."""
    cases = []
    for folder_name, label in (("normal", 0), ("glaucoma", 1)):
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Required Drishti folder not found: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                cases.append({"path": path, "ground_truth": label})
    return cases[:MAX_CASES] if MAX_CASES > 0 else cases


def initialize_state(case):
    return {
        "patient_id": case["path"].name,
        "image_path": str(case["path"]),
        "cfp_diagnosis": None,
        "retfound_scores": "",
        "vision_opinion_cfp": "",
        "counterfactual_trace": {},
        "final_diagnosis": {},
    }


def run_diagnostic_pipeline(case, state):
    state["retfound_scores"], state["vision_opinion_cfp"], probability = vision_agent.analyze(
        case["path"], state
    )
    print(f"\n{state['retfound_scores']}")
    print(f"\nCFP Specialist Report:\n{state['vision_opinion_cfp']}")

    if probability >= 90:
        state["final_diagnosis"] = {"labels": "GLAUCOMA_DETECTED: 1"}
        return
    if probability <= 10:
        state["final_diagnosis"] = {"labels": "GLAUCOMA_DETECTED: 0"}
        return

    audit = counterfactual_agent.analyze(
        case_id=state["patient_id"],
        retfound_probability=probability,
        cfp_report=state["vision_opinion_cfp"],
    )
    state["counterfactual_trace"] = counterfactual_agent.concise_trace(audit)
    state["final_diagnosis"] = ophthalmic_agent.analyze(
        probability,
        state["vision_opinion_cfp"],
        state["counterfactual_trace"],
    )
    print(f"\nFinal Diagnosis:\n{state['final_diagnosis']['decision']}")


def parse_glaucoma_label(state):
    match = re.search(
        r"GLAUCOMA_DETECTED:\s*(-?\d+)",
        state["final_diagnosis"].get("labels", ""),
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else -1


def main():
    cases = discover_cases(DATA_ROOT)
    print(f"Found {len(cases)} Drishti CFP images")
    results = []

    for index, case in enumerate(cases):
        try:
            state = initialize_state(case)
            run_diagnostic_pipeline(case, state)
            prediction = parse_glaucoma_label(state)
            result = {
                "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"],
                "Pred_GL": prediction,
                "Is_Correct": int(prediction == case["ground_truth"]) if prediction != -1 else -1,
            }
        except Exception as exc:
            print(f"!!! Error processing {case['path']}: {exc}")
            result = {
                "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"],
                "Pred_GL": -1,
                "Is_Correct": -1,
            }
        results.append(result)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"Completed {index + 1}/{len(cases)}\n{'-' * 30}")


if __name__ == "__main__":
    main()
