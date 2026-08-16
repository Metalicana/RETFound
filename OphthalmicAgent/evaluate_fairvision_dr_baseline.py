"""FairVision DR LLM baseline: GPT-5.1, GPT-5.6, or Claude 4.5.

In ``build_evaluator``, leave exactly one return line uncommented. Inputs are
demographics, the SLO/fundus image, and eight representative OCT B-scans. No
RETFound or specialist-model output is supplied.
"""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from anthropic import AnthropicFoundry
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


load_dotenv()

INPUT_CSV = Path(os.getenv("FAIRVISION_CSV", "./data/fairvision_250each.csv"))
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
OCT_SLICES = int(os.getenv("LLM_OCT_SLICES", "8"))
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

SYSTEM_PROMPT = (
    "You are a standalone ophthalmic image evaluator. Determine the binary diabetic retinopathy "
    "(DR) label using only the supplied SLO/fundus image, representative OCT B-scans, and basic "
    "demographics. Demographics are contextual metadata only and must never be treated as anatomical "
    "proof of disease. Examine visible retinal findings including microaneurysms, dot-blot or flame "
    "hemorrhages, hard exudates, cotton-wool spots, venous beading, IRMA, neovascularization, vitreous "
    "hemorrhage, and tractional changes. On OCT, assess intraretinal cysts or fluid, subretinal fluid, "
    "retinal thickening, and other diabetic macular changes. Consider image quality and agreement "
    "between modalities. Do not invent laboratory values, diabetes duration, visual acuity, treatment "
    "history, angiography findings, or foundation-model scores. Return JSON only with exactly these "
    "fields: dr_detected (integer 0 or 1), confidence (low, moderate, or high), and reasoning (brief "
    "evidence-based explanation). You must choose 0 or 1 even when uncertain."
)


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
    return np.clip(255 * (array - low) / (high - low), 0, 255).astype(np.uint8)


def as_rgb(array):
    return Image.fromarray(normalize_uint8(array)).convert("RGB")


def make_oct_grid(volume, count):
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(volume) == 0:
        raise ValueError(f"Expected OCT volume [slices,H,W], got {volume.shape}")
    count = max(1, min(count, len(volume)))
    indices = np.linspace(0, len(volume) - 1, count, dtype=int)
    images = [ImageOps.contain(as_rgb(volume[index]), (448, 224)) for index in indices]
    width, height = max(x.width for x in images), max(x.height for x in images)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    grid = Image.new("RGB", (columns * width, rows * height), "black")
    for position, image in enumerate(images):
        x = (position % columns) * width + (width - image.width) // 2
        y = (position // columns) * height + (height - image.height) // 2
        grid.paste(image, (x, y))
    return grid, indices.tolist()


def jpeg_base64(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError(f"Model did not return JSON: {text[:300]!r}")
        try:
            result, _ = json.JSONDecoder().raw_decode(text[start:])
            return result
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse model response: {text[:500]!r}") from exc


def validate(raw):
    result = extract_json(raw)
    prediction = int(result["dr_detected"])
    if prediction not in (0, 1):
        raise ValueError(f"Invalid dr_detected value: {prediction}")
    return prediction, str(result.get("confidence", "")), str(result.get("reasoning", "")), raw


def user_text(case):
    return (
        f"Demographics:\nAge: {case['age']}\nGender: {case['gender']}\n"
        f"Race: {case['race']}\nEthnicity: {case['ethnicity']}\n\n"
        "First image: SLO/fundus. Second image: evenly spaced OCT B-scans "
        f"from slice indices {case['oct_indices']}. Assess the binary DR label."
    )


class AzureGPTDREvaluator:
    def __init__(self, deployment, model_tag):
        self.deployment, self.model_tag = deployment, model_tag
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=AZURE_API_VERSION,
        )

    def analyze(self, case):
        response = self.client.chat.completions.create(
            model=self.deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text(case)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_base64(case['fundus'])}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_base64(case['oct_grid'])}"}},
                ]},
            ],
        )
        return validate(response.choices[0].message.content or "")


class GPT51DREvaluator(AzureGPTDREvaluator):
    def __init__(self):
        super().__init__(os.getenv("GPT51_DEPLOYMENT", "gpt-5.1"), "gpt51")


class GPT56DREvaluator(AzureGPTDREvaluator):
    def __init__(self):
        super().__init__(os.getenv("GPT56_DEPLOYMENT", "gpt-5.6-luna"), "gpt56")


class Claude45DREvaluator:
    model_tag = "claude45"

    def __init__(self):
        self.deployment = os.getenv("ANTHROPIC_DEPLOYMENT", "claude-haiku-4-5")
        base_url = (os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
                    or os.getenv("AZURE_AI_ANTHROPIC_ENDPOINT")
                    or os.getenv("AZURE_OPENAI_ENDPOINT"))
        api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if not base_url:
            raise ValueError("Set ANTHROPIC_FOUNDRY_BASE_URL to https://<resource>.services.ai.azure.com/anthropic")
        if not api_key:
            raise ValueError("Set AZURE_API_KEY (or AZURE_OPENAI_API_KEY)")
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1/messages"):
            base_url = base_url.removesuffix("/v1/messages")
        if not base_url.endswith("/anthropic"):
            raise ValueError(f"Claude Foundry base URL must end in '/anthropic'; received {base_url}")
        self.client = AnthropicFoundry(api_key=api_key, base_url=base_url)

    def analyze(self, case):
        content = [{"type": "text", "text": user_text(case)}]
        for image in (case["fundus"], case["oct_grid"]):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": jpeg_base64(image)
            }})
        response = self.client.messages.create(
            model=self.deployment, max_tokens=1024, temperature=0.2,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": content}],
        )
        raw = "\n".join(block.text for block in response.content
                         if getattr(block, "type", None) == "text").strip()
        return validate(raw)


def build_evaluator():
    """COMMENT/UNCOMMENT exactly one of these three lines."""
#    return GPT51DREvaluator()
#    return GPT56DREvaluator()
    return Claude45DREvaluator()


def find_column(columns, candidates, required=True):
    lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(f"Expected one of these columns: {candidates}")
    return None


def row_value(row, names, default="Unknown"):
    column = find_column(row.index, names, required=False)
    return default if column is None or pd.isna(row[column]) else row[column]


def load_rows():
    frame = pd.read_csv(INPUT_CSV)
    disease_column = find_column(frame.columns, ("Task_Folder", "disease_folder", "disease"))
    frame = frame[frame[disease_column].astype(str).str.strip().str.lower() == "dr"].reset_index(drop=True)
    if MAX_CASES > 0:
        frame = frame.head(MAX_CASES)
    if frame.empty:
        raise ValueError(f"No DR cases found in {INPUT_CSV}")
    return frame


def load_case(row):
    path_column = find_column(row.index, ("filepath", "file_path", "path", "npz_path", "filename"))
    truth_column = find_column(row.index, ("Ground_Truth", "groundtruth", "gt", "label"))
    path = Path(str(row[path_column]))
    with np.load(path) as container:
        fundus = as_rgb(container["slo_fundus"])
        oct_grid, indices = make_oct_grid(container["oct_bscans"], OCT_SLICES)
    truth = int(float(row[truth_column]))
    if truth not in (0, 1):
        raise ValueError(f"DR Ground_Truth must be binary, got {truth} for {path}")
    return {
        "path": path, "ground_truth": truth,
        "age": row_value(row, ("Age",)), "gender": row_value(row, ("Gender", "Sex")),
        "race": row_value(row, ("Race",)), "ethnicity": row_value(row, ("Ethnicity",)),
        "fundus": fundus, "oct_grid": oct_grid, "oct_indices": indices,
    }


def print_metrics(results):
    valid = results[results["Pred_DR"].isin([0, 1])]
    failed = len(results) - len(valid)
    if valid.empty:
        print("No valid predictions were produced.")
        return
    truth = valid["Ground_Truth"].astype(int).to_numpy()
    predictions = valid["Pred_DR"].astype(int).to_numpy()
    print("\nPer-class classification report:")
    print(classification_report(truth, predictions, labels=[0, 1],
          target_names=["No DR", "DR"], digits=4, zero_division=0))
    print(f"Accuracy: {accuracy_score(truth, predictions):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(truth, predictions, labels=[0, 1]))
    print(f"Valid predictions: {len(valid)} | Failed/invalid predictions: {failed}")


def main():
    frame, evaluator, rows = load_rows(), build_evaluator(), []
    output_csv = os.getenv("OUTPUT_CSV", f"fairvision_dr_{evaluator.model_tag}_predictions.csv")
    print(f"Evaluating {evaluator.deployment!r} on {len(frame)} FairVision DR cases; output={output_csv}")
    for position, (_, source_row) in enumerate(frame.iterrows(), start=1):
        case = None
        print("\n" + "=" * 90 + f"\nCASE {position}/{len(frame)}")
        try:
            case = load_case(source_row)
            prediction, confidence, reasoning, raw = evaluator.analyze(case)
            print(f"Filename: {case['path']}")
            print(f"Ground truth: {case['ground_truth']} | Prediction: {prediction} | Confidence: {confidence}")
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == case['ground_truth']}")
            result = {
                "Filename": str(case["path"]), "Model": evaluator.deployment,
                "Age": case["age"], "Gender": case["gender"], "Race": case["race"],
                "Ethnicity": case["ethnicity"], "Ground_Truth": case["ground_truth"],
                "Pred_DR": prediction, "Confidence": confidence, "Reasoning": reasoning,
                "Raw_Response": raw, "Is_Correct": int(prediction == case["ground_truth"]),
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            result = {
                "Filename": str(case["path"]) if case else str(row_value(source_row, ("filename",), "Error")),
                "Model": evaluator.deployment,
                "Ground_Truth": case["ground_truth"] if case else row_value(source_row, ("Ground_Truth",), -1),
                "Pred_DR": -1, "Confidence": "", "Reasoning": "", "Raw_Response": "",
                "Is_Correct": -1, "Error": str(exc),
            }
        rows.append(result)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"Checkpoint saved: {output_csv}")
    results = pd.DataFrame(rows)
    print_metrics(results)
    print(f"\nPredictions saved to {output_csv}")


if __name__ == "__main__":
    main()
