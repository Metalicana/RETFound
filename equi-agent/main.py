# OphthalmicAgent/main.py

from Orchestrator.state import AgentState
from BioProfilerAgent.bio_profiler import BioProfiler
from EquityAgent.equity_agent import EquityAgent
from VisionAgent.vision import VisionSpecialist
from VisionAgent.linear_probing_oct3 import FairVisionNPZ
from FunctionalInterpretationAgent.function_interpreter import FunctionalSpecialist
from GuidelinesAgent.guidelines_agent import GuidelinesAgent
from SafetyAgent.safety_agent import SafetyAgent
from Orchestrator.ophthalmic_agent import Orchestrator

from data.loader import GenericEyeLoader

import numpy as np
from PIL import Image
import pandas as pd
import json
import re
import os
import traceback
import csv
from pathlib import Path

OUTPUT_CSV = os.environ.get("EQUI_AGENT_OUTPUT_CSV", "ophthalmic_performance_results_apr30.csv")
TRACE_JSONL = os.environ.get("EQUI_AGENT_TRACE_JSONL", "")
MAX_CASES_PER_DISEASE = int(os.environ.get("EQUI_AGENT_MAX_CASES_PER_DISEASE", "100"))
AMD_BINARY_SCORING = os.environ.get("EQUI_AGENT_AMD_BINARY_SCORING", "0") == "1"
OMIT_MD = os.environ.get("EQUI_AGENT_OMIT_MD", "1") == "1"
ALLOW_LABEL_METADATA = os.environ.get("EQUI_AGENT_ALLOW_LABEL_METADATA", "0") == "1"
DISEASES_ENV = os.environ.get("EQUI_AGENT_DISEASES", "")
OUTPUTS_ROOT = Path(os.environ.get("EQUI_AGENT_OUTPUTS_ROOT", "outputs"))

MD_METADATA_KEYS = {
    "md",
    "mean_deviation",
    "mean deviation",
    "visual_field_md",
    "vf_md",
}

LABEL_METADATA_KEYS = {
    "glaucoma",
    "amd",
    "dr",
    "amd_condition",
    "dr_subtype",
    "stage",
    "label",
    "ground_truth",
    "y",
    "target",
}

PREDICTION_FILES = {
    "retfound_oct": OUTPUTS_ROOT / "predictions" / "fairvision_oct_retfound_test_thresholded.csv",
    "mirage_slo": OUTPUTS_ROOT / "predictions" / "fairvision_slo_mirage_test_thresholded.csv",
}

THRESHOLD_FILES = {
    "retfound_oct": OUTPUTS_ROOT / "metrics" / "thresholds_fairvision_oct_retfound.csv",
    "mirage_slo": OUTPUTS_ROOT / "metrics" / "thresholds_fairvision_slo_mirage.csv",
}

RELIABILITY_TO_CALIBRATED_MODEL = {
    "retfound": "retfound_oct",
    "mirage": "mirage_slo",
    "retfound_oct": "retfound_oct",
    "mirage_slo": "mirage_slo",
}


def age_to_group(age):
    try:
        age_value = float(age)
    except (TypeError, ValueError):
        return "missing"
    if age_value < 50:
        return "younger"
    if age_value < 70:
        return "middle-aged"
    return "older"


def normalize_race(value, ethnicity=None):
    ethnicity_text = str(ethnicity or "").strip().lower()
    if ethnicity_text == "hispanic":
        return "Hispanic"
    race_text = str(value or "").strip().lower()
    if race_text in {"white", "caucasian"}:
        return "Caucasian"
    if race_text == "black":
        return "Black"
    if race_text == "asian":
        return "Asian"
    if race_text == "hispanic":
        return "Hispanic"
    return None


def normalize_gender(value):
    gender_text = str(value or "").strip().lower()
    if gender_text in {"male", "m"}:
        return "male"
    if gender_text in {"female", "f"}:
        return "female"
    return None


def build_structured_reliability(metadata):
    calibration = getattr(equity_agent, "calibration_summary", {}).get("models", {})
    subgroup_context = {
        "race": normalize_race(metadata.get("race"), metadata.get("ethnicity")),
        "gender": normalize_gender(metadata.get("gender")),
        "age_group": age_to_group(metadata.get("age")),
    }
    task_map = {
        "AMD": "amd",
        "DR": "diabetic_retinopathy",
        "Glaucoma": "glaucoma",
    }
    report = {
        "subgroup_context": subgroup_context,
        "tasks": {},
    }

    for display_task, calibration_task in task_map.items():
        task_models = {}
        for model_name, model_calibration in calibration.items():
            fp_values = []
            fn_values = []
            matched = []
            for attr, subgroup in subgroup_context.items():
                if not subgroup or subgroup == "missing":
                    continue
                fp = (
                    model_calibration.get("false_positive", {})
                    .get(calibration_task, {})
                    .get(attr, {})
                    .get(subgroup)
                )
                fn = (
                    model_calibration.get("false_negative", {})
                    .get(calibration_task, {})
                    .get(attr, {})
                    .get(subgroup)
                )
                if fp is not None:
                    fp_values.append(float(fp))
                if fn is not None:
                    fn_values.append(float(fn))
                if fp is not None or fn is not None:
                    matched.append({"attribute": attr, "subgroup": subgroup, "fpr": fp, "fnr": fn})

            fpr_mean = sum(fp_values) / len(fp_values) if fp_values else None
            fnr_mean = sum(fn_values) / len(fn_values) if fn_values else None
            error_sum = (
                (fpr_mean if fpr_mean is not None else 0.0)
                + (fnr_mean if fnr_mean is not None else 0.0)
            )
            task_models[model_name] = {
                "matched_priors": matched,
                "mean_fpr": fpr_mean,
                "mean_fnr": fnr_mean,
                "mean_fpr_plus_fnr": error_sum if matched else None,
                "high_fp": fpr_mean is not None and fpr_mean > 0.15,
                "high_fn": fnr_mean is not None and fnr_mean > 0.15,
            }

        ranked = sorted(
            (
                (name, values)
                for name, values in task_models.items()
                if values["mean_fpr_plus_fnr"] is not None
            ),
            key=lambda item: item[1]["mean_fpr_plus_fnr"],
        )
        report["tasks"][display_task] = {
            "models": task_models,
            "lowest_total_error_model": ranked[0][0] if ranked else None,
            "lowest_total_error": ranked[0][1]["mean_fpr_plus_fnr"] if ranked else None,
            "lowest_fpr_model": min(
                task_models,
                key=lambda name: task_models[name]["mean_fpr"]
                if task_models[name]["mean_fpr"] is not None
                else 999,
            )
            if task_models
            else None,
            "lowest_fnr_model": min(
                task_models,
                key=lambda name: task_models[name]["mean_fnr"]
                if task_models[name]["mean_fnr"] is not None
                else 999,
            )
            if task_models
            else None,
        }

    return report


def load_thresholds():
    thresholds = {}
    for model_name, path in THRESHOLD_FILES.items():
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                thresholds[(model_name, row["task"])] = float(row["threshold"])
    return thresholds


def load_prediction_index():
    index = {}
    for model_name, path in PREDICTION_FILES.items():
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "test":
                    continue
                key = (str(row.get("image_id", "")), str(row.get("task", "")), model_name)
                index[key] = {
                    "model_name": model_name,
                    "task": row.get("task", ""),
                    "image_id": row.get("image_id", ""),
                    "y_prob": float(row["y_prob"]) if row.get("y_prob") not in {"", None} else None,
                    "y_pred": int(float(row["y_pred"])) if row.get("y_pred") not in {"", None} else None,
                }
    return index


CALIBRATED_THRESHOLDS = load_thresholds()
CALIBRATED_PREDICTIONS = load_prediction_index()


def build_calibrated_model_decisions(patient_record):
    image_id = Path(str(patient_record.get("directory", ""))).name
    tasks = ["amd", "dr", "glaucoma"]
    models = ["retfound_oct", "mirage_slo"]
    packet = {
        "image_id": image_id,
        "models": {},
    }
    for model_name in models:
        task_rows = {}
        for task in tasks:
            row = CALIBRATED_PREDICTIONS.get((image_id, task, model_name))
            threshold = CALIBRATED_THRESHOLDS.get((model_name, task))
            if row is None:
                continue
            margin = None
            if row["y_prob"] is not None and threshold is not None:
                margin = row["y_prob"] - threshold
            task_rows[task] = {
                "y_prob": row["y_prob"],
                "validation_threshold": threshold,
                "calibrated_y_pred": row["y_pred"],
                "threshold_margin": margin,
            }
        packet["models"][model_name] = task_rows
    return packet


def build_reliability_recommendations(calibrated_packet, reliability_report):
    recommendations = {}
    task_name_map = {
        "amd": "AMD",
        "dr": "DR",
        "glaucoma": "Glaucoma",
    }
    for task_key, display_task in task_name_map.items():
        task_reliability = reliability_report.get("tasks", {}).get(display_task, {})
        preferred_model = task_reliability.get("lowest_total_error_model")
        lowest_fpr_model = task_reliability.get("lowest_fpr_model")
        lowest_fnr_model = task_reliability.get("lowest_fnr_model")
        preferred_model_key = RELIABILITY_TO_CALIBRATED_MODEL.get(preferred_model, preferred_model)
        lowest_fpr_model_key = RELIABILITY_TO_CALIBRATED_MODEL.get(lowest_fpr_model, lowest_fpr_model)
        lowest_fnr_model_key = RELIABILITY_TO_CALIBRATED_MODEL.get(lowest_fnr_model, lowest_fnr_model)

        task_model_rows = {
            model_name: task_rows.get(task_key)
            for model_name, task_rows in calibrated_packet.get("models", {}).items()
            if task_rows.get(task_key) is not None
        }
        positive_models = [
            name for name, values in task_model_rows.items()
            if values.get("calibrated_y_pred") == 1
        ]
        negative_models = [
            name for name, values in task_model_rows.items()
            if values.get("calibrated_y_pred") == 0
        ]

        preferred_values = task_model_rows.get(preferred_model_key)
        recommended_label = preferred_values.get("calibrated_y_pred") if preferred_values else None
        rationale = "preferred_total_error_model"

        if task_key == "glaucoma":
            if positive_models == [] and negative_models:
                recommended_label = 0
                rationale = "glaucoma_all_models_negative"
            elif len(positive_models) == len(task_model_rows) and positive_models:
                recommended_label = 1
                rationale = "glaucoma_all_models_positive"
            else:
                lower_fpr_values = task_model_rows.get(lowest_fpr_model_key)
                lower_fnr_values = task_model_rows.get(lowest_fnr_model_key)
                positive_margins = [
                    values.get("threshold_margin")
                    for values in task_model_rows.values()
                    if values.get("calibrated_y_pred") == 1 and values.get("threshold_margin") is not None
                ]
                max_positive_margin = max(positive_margins) if positive_margins else None
                if (
                    lower_fpr_values is not None
                    and lower_fpr_values.get("calibrated_y_pred") == 1
                    and max_positive_margin is not None
                    and max_positive_margin >= 0.15
                ):
                    recommended_label = 1
                    rationale = "glaucoma_lowest_fpr_positive_with_margin"
                elif (
                    lower_fnr_values is not None
                    and lower_fpr_values is not None
                    and lower_fnr_values.get("calibrated_y_pred") == 1
                    and lower_fnr_values.get("threshold_margin") is not None
                    and lower_fnr_values.get("threshold_margin") >= 0.20
                    and lower_fpr_values.get("calibrated_y_pred") == 0
                    and lower_fpr_values.get("threshold_margin") is not None
                    and lower_fpr_values.get("threshold_margin") > -0.10
                ):
                    recommended_label = 1
                    rationale = "glaucoma_lowest_fnr_strong_positive_lowest_fpr_borderline_negative"
                else:
                    recommended_label = 0
                    rationale = "glaucoma_conflict_fp_control"
        elif task_key == "amd":
            lower_fnr_values = task_model_rows.get(lowest_fnr_model_key)
            if lower_fnr_values is not None and lower_fnr_values.get("calibrated_y_pred") == 1:
                recommended_label = 1
                rationale = "amd_fn_control_lowest_fnr_model_positive"
            elif len(positive_models) >= 2:
                recommended_label = 1
                rationale = "amd_model_consensus_positive"
            elif positive_models:
                recommended_label = 1
                rationale = "amd_single_model_positive_fn_control"
            elif negative_models:
                recommended_label = 0
                rationale = "amd_all_models_negative"
        else:
            if preferred_values is not None:
                recommended_label = preferred_values.get("calibrated_y_pred")
                rationale = "preferred_total_error_model"
            elif len(positive_models) == len(task_model_rows) and positive_models:
                recommended_label = 1
                rationale = "all_models_positive"
            elif negative_models:
                recommended_label = 0
                rationale = "default_negative"

        recommendations[display_task] = {
            "recommended_binary_label": recommended_label,
            "rationale": rationale,
            "preferred_total_error_model": preferred_model,
            "preferred_total_error_model_key": preferred_model_key,
            "lowest_fpr_model": lowest_fpr_model,
            "lowest_fpr_model_key": lowest_fpr_model_key,
            "lowest_fnr_model": lowest_fnr_model,
            "lowest_fnr_model_key": lowest_fnr_model_key,
            "positive_models": positive_models,
            "negative_models": negative_models,
            "model_decisions": task_model_rows,
        }
    return recommendations


def format_calibrated_model_decisions(packet):
    lines = ["CALIBRATED_MODEL_DECISIONS"]
    lines.append(f"image_id={packet.get('image_id')}")
    for model_name, task_rows in packet.get("models", {}).items():
        lines.append(f"{model_name}:")
        for task, values in task_rows.items():
            lines.append(
                f"  - {task}: y_prob={values.get('y_prob')}, "
                f"validation_threshold={values.get('validation_threshold')}, "
                f"calibrated_y_pred={values.get('calibrated_y_pred')}, "
                f"threshold_margin={values.get('threshold_margin')}"
            )
    return "\n".join(lines)


def format_reliability_recommendations(recommendations):
    lines = ["RELIABILITY_WEIGHTED_RECOMMENDATIONS"]
    for task, values in recommendations.items():
        lines.append(
            f"{task}: recommended_binary_label={values.get('recommended_binary_label')}, "
            f"rationale={values.get('rationale')}, "
            f"preferred_total_error_model={values.get('preferred_total_error_model')}, "
            f"lowest_fpr_model={values.get('lowest_fpr_model')}, "
            f"lowest_fnr_model={values.get('lowest_fnr_model')}, "
            f"positive_models={values.get('positive_models')}, "
            f"negative_models={values.get('negative_models')}"
        )
    return "\n".join(lines)


def format_structured_reliability(report):
    lines = ["STRUCTURED_RELIABILITY_PRIORS"]
    context = report.get("subgroup_context", {})
    lines.append(
        "Subgroup context: "
        + ", ".join(f"{key}={value}" for key, value in context.items())
    )
    for task, task_report in report.get("tasks", {}).items():
        lines.append(f"{task}:")
        for model_name, values in task_report.get("models", {}).items():
            fpr = values.get("mean_fpr")
            fnr = values.get("mean_fnr")
            total = values.get("mean_fpr_plus_fnr")
            lines.append(
                f"  - {model_name}: mean_fpr={fpr if fpr is not None else 'NA'}, "
                f"mean_fnr={fnr if fnr is not None else 'NA'}, "
                f"fpr_plus_fnr={total if total is not None else 'NA'}, "
                f"high_fp={values.get('high_fp')}, high_fn={values.get('high_fn')}"
            )
        lines.append(
            f"  Recommended by priors: lowest_total_error={task_report.get('lowest_total_error_model')}, "
            f"lowest_fpr={task_report.get('lowest_fpr_model')}, "
            f"lowest_fnr={task_report.get('lowest_fnr_model')}"
        )
    return "\n".join(lines)


def append_trace(row):
    if not TRACE_JSONL:
        return
    trace_dir = os.path.dirname(TRACE_JSONL)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    with open(TRACE_JSONL, "a") as handle:
        handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def case_trace(disease, index, patient_record, row, state, error=None):
    metadata = patient_record.get("metadata", {}) if isinstance(patient_record, dict) else {}
    vision = state.get("vision_opinion", {}) if isinstance(state, dict) else {}
    return {
        "disease": disease,
        "case_index": index,
        "filename": patient_record.get("directory", "") if isinstance(patient_record, dict) else "",
        "metadata": metadata,
        "result_row": row,
        "error": error,
        "agents": {
            "bio_profiler": state.get("clinical_narrative", "") if isinstance(state, dict) else "",
            "vision_agent": {
                "summary": vision.get("summary", ""),
                "retfound_scores": vision.get("retfound_scores", ""),
                "mirage_scores": vision.get("mirage_scores", ""),
                "full_report": vision.get("full_report", ""),
            },
            "functional_agent": state.get("functional_opinion", {}) if isinstance(state, dict) else {},
            "calibrated_model_decisions": state.get("calibrated_model_decisions", {}) if isinstance(state, dict) else {},
            "structured_reliability": state.get("structured_reliability", {}) if isinstance(state, dict) else {},
            "reliability_recommendations": state.get("reliability_recommendations", {}) if isinstance(state, dict) else {},
            "equity_agent": state.get("equity_opinion", "") if isinstance(state, dict) else "",
            "guidelines_agent": state.get("guidelines", "") if isinstance(state, dict) else "",
            "orchestrator": state.get("final_diagnosis", {}) if isinstance(state, dict) else {},
            "safety_agent": state.get("safety_output", "") if isinstance(state, dict) else "",
        },
    }


def sanitize_patient_record_for_agents(patient_record):
    masked_record = dict(patient_record)
    metadata = dict(masked_record.get("metadata", {}))
    for key in list(metadata):
        normalized_key = str(key).strip().lower()
        if OMIT_MD and normalized_key in MD_METADATA_KEYS:
            metadata[key] = None
        if not ALLOW_LABEL_METADATA and normalized_key in LABEL_METADATA_KEYS:
            metadata.pop(key, None)
    masked_record["metadata"] = metadata
    return masked_record

#Intiializing agents
profiler = BioProfiler()
equity_agent = EquityAgent()
vision_agent = VisionSpecialist(
    os.environ.get("EQUI_AGENT_OCT_MODEL_WEIGHTS", "./weights/oct_model_best.pth"),
    os.environ.get("EQUI_AGENT_SLO_MODEL_WEIGHTS", "./weights/slo_model_best.pth"),
)
functional_agent = FunctionalSpecialist()
ophthalmic_agent = Orchestrator()
safety_agent = SafetyAgent()
guidelines_agent = GuidelinesAgent()

def run_diagnostic_pipeline(patient_data):

    #BIO PROFILER AGENT
    print("\n\n--- Sending Meta Data to Bio Profiler Agent ---")
    final_state["clinical_narrative"] = profiler.generate_narrative(final_state["metadata"])
    print("\n\nNarrative Ready: ")
    print(f"{final_state['clinical_narrative']}")
    print("\n" + "-"*30)

    #VISION AGENT
    print("\n\n--- Sending Visual Data to Vision Specialist ---")
    final_state["calibrated_model_decisions"] = build_calibrated_model_decisions(patient_data)
    final_state["calibrated_model_decisions_text"] = format_calibrated_model_decisions(
        final_state["calibrated_model_decisions"]
    )
    print("\n\nCalibrated Model Decisions Ready: ")
    print(final_state["calibrated_model_decisions_text"])

    final_state["vision_opinion"] = vision_agent.analyze(patient_data['oct_img'], patient_data['fundus_img'], final_state)
    print("\n\nProbabilities Ready: ")
    print("\n\nOCT DIAGNOSIS USING FINETUNED RETFOUND")

    oct_diagnosis = final_state['oct_diagnosis']

    retfound_scores = final_state["vision_opinion"]["retfound_scores"]

    print(retfound_scores)

    print("\nSLO DIAGNOSIS USING FINETUNED MIRAGE")
    slo_diagnosis = final_state['slo_diagnosis']

    mirage_scores = final_state["vision_opinion"]["mirage_scores"]

    print(mirage_scores)

    print("\n\nVision Specialist's Output Ready: ")
    print(f"{final_state['vision_opinion']['summary']}")

    print("\n" + "-"*30)

    #FUNCTIONAL INTERPRETATION AGENT
    print("\n\n--- Sending Visual Field Data to Functional Interpretation Agent ---")
    final_state["functional_opinion"] = functional_agent.analyze(final_state)
    print("\n\nFunctional Vision Interpreter's Output Ready: ")
    print(f"{final_state['functional_opinion']['summary']}")

    print("\n" + "-"*30)

    #EQUITY AGENT
    print("\n\n--- Sending Narrative and Visual Findings to Equity Agent ---")
    final_state["structured_reliability"] = build_structured_reliability(final_state["metadata"])
    final_state["structured_reliability_text"] = format_structured_reliability(final_state["structured_reliability"])
    print("\n\nStructured Reliability Priors Ready: ")
    print(final_state["structured_reliability_text"])
    final_state["reliability_recommendations"] = build_reliability_recommendations(
        final_state["calibrated_model_decisions"],
        final_state["structured_reliability"],
    )
    final_state["reliability_recommendations_text"] = format_reliability_recommendations(
        final_state["reliability_recommendations"]
    )
    print("\n\nReliability Weighted Recommendations Ready: ")
    print(final_state["reliability_recommendations_text"])

    equity_input = f"""
    ### PATIENT DATA
    Patient narrative: {final_state['clinical_narrative']}

    ### AI MODEL OUTPUTS
    {retfound_scores}


    {mirage_scores}"""

    final_state["equity_opinion"] = equity_agent.analyze_patients(equity_input, output_format="text")
    print("\n\nEquity Agent's Output Ready: ")
    print(f"{final_state['equity_opinion']}")
    print("\n" + "-"*30)

    #GUIDELINES AGENT
    print("\n\n--- Sending Query to Guidelines Agent ---")
    note = f"""
    Patient Narrative: {final_state['clinical_narrative']}
    Functional Interpretation Agent output: {final_state['functional_opinion']['summary']}"""

    try:
        final_state['guidelines'] = guidelines_agent.consult_note(note, max_results=5, diagnosis_only=True,)
    except Exception as exc:
        final_state['guidelines'] = (
            "Guidelines retrieval unavailable for this run; proceed using patient metadata, visual findings, "
            f"functional interpretation, and equity audit only. Retrieval error: {type(exc).__name__}: {exc}"
        )
    print("\n\n Guidelines Agent's Output   Ready: ")
    print(f"{final_state['guidelines']}")

    print("\n" + "-"*30)

    #ORCHESTRATOR
    print("\n\n--- Sending Case to Orchestrator ---")
    final_state["final_diagnosis"] = ophthalmic_agent.analyze(final_state, retfound_scores, mirage_scores)
    print("\n\n Ophthalmic Agent's Final Diagnosis Ready: ")
    print(f"{final_state['final_diagnosis']['decision']}")
    print("\n" + "-"*30)

    #SAFETY AGENT
    print("\n\n--- Sending Case to Safety Agent ---")
    final_state["safety_output"] = safety_agent.run(final_state)
    print("\n\n Safety Agent's Output Ready: ")
    print(f"{final_state['safety_output']}")
    print("\n" + "-"*30)

def parse_agent_labels(final_state):
    label_text = final_state['final_diagnosis']['labels']

    patterns = {
        "AMD": r"AMD_STAGE:\s*(-?\d+)",
        "DR": r"DR_DETECTED:\s*(-?\d+)",
        "GL": r"GLAUCOMA_DETECTED:\s*(-?\d+)"
    }

    results = {}

    try:
        for key, pattern in patterns.items():
            match = re.search(pattern, label_text, re.IGNORECASE)
            if match:
                results[key] = int(match.group(1))
            else:
                print(f"Warning: Could not find {key} in label text!")
                results[key] = -1 # Default fallback
        return results
    except Exception as e:
        print(f"Parsing Error: {e}")
        return None

def initialize_state(patient_data):
    # Initialize the State
    state: AgentState = {
        "patient_id": patient_data['metadata']['filename'],
        "metadata": patient_data['metadata'],
        "fundus_img": np.array(patient_data['fundus_img']),
        "oct_img": np.array(patient_data['oct_img']),
        "clinical_narrative": "",
        "oct_diagnosis": None,
        "slo_diagnosis": None,
        "vision_opinion": {},
        "functional_opinion": {},
        "calibrated_model_decisions": {},
        "calibrated_model_decisions_text": "",
        "structured_reliability": {},
        "structured_reliability_text": "",
        "reliability_recommendations": {},
        "reliability_recommendations_text": "",
        "final_diagnosis": {},
        "fairness_flag": False,
        "safety_output": "",
        "guidelines": "",
        "equity_opinion:": ""
    }

#    Image.fromarray(state['oct_img']).save("check_my_work.png")
#    Image.fromarray(state['fundus_img']).save("check_my_work2.png")
    return state

if __name__ == "__main__":

    results = []

    BASE_PATH = os.environ.get("EQUI_AGENT_DATA_ROOT", "../Datasets/FairVision")
    disease_name_map = {
        "glaucoma": "Glaucoma",
        "amd": "AMD",
        "dr": "DR",
    }
    if DISEASES_ENV.strip():
        diseases = [
            disease_name_map[name.strip().lower()]
            for name in DISEASES_ENV.split(",")
            if name.strip().lower() in disease_name_map
        ]
    else:
        diseases = ['Glaucoma', 'AMD', 'DR']
#    diseases = ['AMD', 'Glaucoma', 'DR']
#    diseases = ['DR', 'AMD', 'Glaucoma']

    for disease in diseases:
      loader = GenericEyeLoader(BASE_PATH)
      df = loader.get_metadata(disease)

      test_rows = df[df['use'] == 'test']

      if not test_rows.empty:
          for i in range(min(MAX_CASES_PER_DISEASE, len(test_rows))):

            try:
              patient_record = loader.load_patient(disease, test_rows.iloc[i])
              patient_record = sanitize_patient_record_for_agents(patient_record)

              age = patient_record['metadata']['age']
              gender = patient_record['metadata']['gender']
              race = patient_record['metadata']['race']
              ethnicity = patient_record['metadata']['ethnicity']

              final_state = initialize_state(patient_record)
              run_diagnostic_pipeline(patient_record)

              pred_labels = parse_agent_labels(final_state)
              ground_truth = patient_record['stage']

              row = {
                  "Filename": patient_record['directory'],
                  "Task_Folder": disease,
                  "Age": age,
                  "Gender": gender,
                  "Race": race,
                  "Ethnicity": ethnicity,
                  "Ground_Truth": ground_truth,
                  "Pred_AMD": pred_labels.get("AMD", -1) if pred_labels else -1,
                  "Pred_DR": pred_labels.get("DR", -1) if pred_labels else -1,
                  "Pred_GL": pred_labels.get("GL", -1) if pred_labels else -1,
                  "MD_Omitted": int(OMIT_MD),
              }

              # Select correct prediction
              if "AMD" in disease:
                  pred = row["Pred_AMD"]
              elif "DR" in disease:
                  pred = row["Pred_DR"]
              elif "Glaucoma" in disease:
                  pred = row["Pred_GL"]
              else:
                  pred = -1

              # Compute correctness
              if AMD_BINARY_SCORING and "AMD" in disease and pred != -1 and ground_truth != -1:
                  pred_for_score = int(float(pred) > 0)
                  truth_for_score = int(float(ground_truth) > 0)
              else:
                  pred_for_score = pred
                  truth_for_score = ground_truth

              if pred_for_score == -1 or truth_for_score == -1:
                  row["Is_Correct"] = -1
              else:
                  row["Is_Correct"] = int(pred_for_score == truth_for_score)

              results.append(row)
              append_trace(case_trace(disease, i, patient_record, row, final_state))

              df = pd.DataFrame(results)
              df.to_csv(OUTPUT_CSV, index=False)

              print(f"\nDisease folder is {disease} and ground truth is {ground_truth}, example number is {i}")
              print("\nEND OF EXAMPLE")
              print("\n" + "-"*30)

            except Exception as e:
              # Catching the content filter specifically
              print(f"!!! Error at Index {i}. Skipping...")
              print(traceback.format_exc())
              patient_record = locals().get("patient_record", {})
              metadata = patient_record.get("metadata", {}) if isinstance(patient_record, dict) else {}
              row = {
                    "Filename": patient_record.get("directory", "") if isinstance(patient_record, dict) else "",
                    "Task_Folder": disease,
                    "Age": metadata.get("age", ""),
                    "Gender": metadata.get("gender", ""),
                    "Race": metadata.get("race", ""),
                    "Ethnicity": metadata.get("ethnicity", ""),
                    "Ground_Truth": -1,
                    "Pred_AMD": -1,
                    "Pred_DR": -1,
                    "Pred_GL": -1,
                    "MD_Omitted": int(OMIT_MD),
                    "Is_Correct": -1
                  }
              results.append(row)
              append_trace(
                  case_trace(
                      disease,
                      i,
                      patient_record,
                      row,
                      locals().get("final_state", {}),
                      error=traceback.format_exc(),
                  )
              )

              df = pd.DataFrame(results)
              df.to_csv(OUTPUT_CSV, index=False)

      else:
          print("No test data found to process.")
