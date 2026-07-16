"""Evaluate FairVision glaucoma using one standalone GPT-5.1 call per case.

Inputs to GPT: patient demographics, SLO/fundus image, and representative OCT
slices. No RETFound, CDR, counterfactual, or other agent is used.
"""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


load_dotenv()

INPUT_CSV = Path(os.getenv("FAIRVISION_CSV", "./data/fairvision_250each.csv"))
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "fairvision_glaucoma_gpt51_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-luna")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
OCT_SLICES = int(os.getenv("GPT_OCT_SLICES", "8"))


def normalize_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip(255.0 * (array - low) / (high - low), 0, 255).astype(np.uint8)


def as_rgb(array):
    return Image.fromarray(normalize_uint8(array)).convert("RGB")


def make_oct_grid(oct_volume, number_of_slices):
    """Create one 4-column image containing evenly spaced OCT slices."""
    count = max(1, min(number_of_slices, len(oct_volume)))
    indices = np.linspace(0, len(oct_volume) - 1, count, dtype=int)
    slices = [ImageOps.contain(as_rgb(oct_volume[index]), (448, 224)) for index in indices]
    cell_width = max(image.width for image in slices)
    cell_height = max(image.height for image in slices)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
    for position, image in enumerate(slices):
        x = (position % columns) * cell_width + (cell_width - image.width) // 2
        y = (position // columns) * cell_height + (cell_height - image.height) // 2
        grid.paste(image, (x, y))
    return grid


def data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def find_column(columns, candidates, required=True):
    lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(f"Expected one of these columns: {candidates}")
    return None


def load_glaucoma_cases():
    frame = pd.read_csv(INPUT_CSV)
    disease_column = find_column(
        frame.columns, ("Task_Folder", "disease_folder", "disease")
    )
    frame = frame[
        frame[disease_column].astype(str).str.strip().str.lower() == "glaucoma"
    ].reset_index(drop=True)
    if MAX_CASES > 0:
        frame = frame.head(MAX_CASES)
    if frame.empty:
        raise ValueError(f"No glaucoma cases found in {INPUT_CSV}")
    return frame


def row_value(row, names, default="Unknown"):
    column = find_column(row.index, names, required=False)
    if column is None or pd.isna(row[column]):
        return default
    return row[column]


def load_case(row):
    path_column = find_column(
        row.index, ("filepath", "file_path", "path", "npz_path", "filename")
    )
    truth_column = find_column(
        row.index, ("Ground_Truth", "groundtruth", "gt", "label")
    )
    npz_path = Path(str(row[path_column]))
    container = np.load(npz_path)
    oct_volume = container["oct_bscans"]
    fundus = container["slo_fundus"]
    return {
        "path": npz_path,
        "ground_truth": int(float(row[truth_column])),
        "age": row_value(row, ("Age",)),
        "gender": row_value(row, ("Gender", "Sex")),
        "race": row_value(row, ("Race",)),
        "ethnicity": row_value(row, ("Ethnicity",)),
        "fundus": as_rgb(fundus),
        "oct_grid": make_oct_grid(oct_volume, OCT_SLICES),
    }


class StandaloneGPTGlaucomaEvaluator:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=API_VERSION,
        )

    def analyze(self, case):
        response = self.client.chat.completions.create(
            model=DEPLOYMENT,
#            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a standalone ophthalmic image evaluator. Determine whether glaucoma is "
                        "present using only the supplied SLO/fundus image, representative OCT B-scans, "
                        "and basic demographics. Demographics are contextual metadata only and must never "
                        "be treated as anatomical evidence or as proof of disease. Examine the optic nerve "
                        "head, cupping and neuroretinal rim when visible, retinal nerve fiber layer or other "
                        "glaucoma-related structural patterns, image quality, and agreement between the two "
                        "modalities. Do not invent measurements, history, symptoms, intraocular pressure, "
                        "visual fields, CDR values, or model scores. Return JSON only with exactly these "
                        "fields: glaucoma_detected (integer 0 or 1), confidence (low, moderate, or high), "
                        "reasoning (brief evidence-based explanation). You must choose 0 or 1 even when "
                        "uncertain, and express uncertainty in confidence and reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Demographics:\n"
                                f"Age: {case['age']}\nGender: {case['gender']}\n"
                                f"Race: {case['race']}\nEthnicity: {case['ethnicity']}\n\n"
                                "First image: SLO/fundus. Second image: evenly spaced OCT B-scans. "
                                "Assess this case for glaucoma."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url(case["fundus"])}},
                        {"type": "image_url", "image_url": {"url": data_url(case["oct_grid"])}},
                    ],
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        prediction = int(parsed["glaucoma_detected"])
        if prediction not in (0, 1):
            raise ValueError(f"GPT returned invalid glaucoma_detected value: {prediction}")
        return prediction, str(parsed.get("confidence", "")), str(parsed.get("reasoning", "")), raw


def print_metrics(results):
    valid = results[results["Pred_GL"].isin([0, 1])]
    failed = len(results) - len(valid)
    if valid.empty:
        print("No valid predictions were produced.")
        return
    y_true = valid["Ground_Truth"].astype(int).to_numpy()
    y_pred = valid["Pred_GL"].astype(int).to_numpy()
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
    frame = load_glaucoma_cases()
    evaluator = StandaloneGPTGlaucomaEvaluator()
    rows = []
    print(f"Evaluating standalone GPT-5.1 on {len(frame)} FairVision glaucoma cases")

    for position, (_, source_row) in enumerate(frame.iterrows(), start=1):
        case = None
        print("\n" + "=" * 90)
        print(f"CASE {position}/{len(frame)}")
        try:
            case = load_case(source_row)
            print(f"Filename: {case['path']}")
            print(
                f"Demographics: age={case['age']}, gender={case['gender']}, "
                f"race={case['race']}, ethnicity={case['ethnicity']}"
            )
            prediction, confidence, reasoning, raw = evaluator.analyze(case)
            print(f"Ground truth: {case['ground_truth']}")
            print(f"GPT prediction: {prediction}")
            print(f"Confidence: {confidence}")
            print(f"Reasoning: {reasoning}")
            print(f"Correct: {prediction == case['ground_truth']}")
            result = {
                "Filename": str(case["path"]),
                "Age": case["age"],
                "Gender": case["gender"],
                "Race": case["race"],
                "Ethnicity": case["ethnicity"],
                "Ground_Truth": case["ground_truth"],
                "Pred_GL": prediction,
                "Confidence": confidence,
                "Reasoning": reasoning,
                "Raw_Response": raw,
                "Is_Correct": int(prediction == case["ground_truth"]),
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            result = {
                "Filename": str(case["path"]) if case else str(row_value(source_row, ("filepath", "filename"), "Error")),
                "Age": case["age"] if case else row_value(source_row, ("Age",)),
                "Gender": case["gender"] if case else row_value(source_row, ("Gender", "Sex")),
                "Race": case["race"] if case else row_value(source_row, ("Race",)),
                "Ethnicity": case["ethnicity"] if case else row_value(source_row, ("Ethnicity",)),
                "Ground_Truth": case["ground_truth"] if case else row_value(source_row, ("Ground_Truth",), -1),
                "Pred_GL": -1,
                "Confidence": "",
                "Reasoning": "",
                "Raw_Response": "",
                "Is_Correct": -1,
            }
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        print(f"Checkpoint saved: {OUTPUT_CSV}")

    results = pd.DataFrame(rows)
    print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
