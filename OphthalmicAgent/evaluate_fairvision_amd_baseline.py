"""FairVision AMD LLM baseline: GPT-5.1, GPT-5.6, or Claude 4.5.

Select exactly one evaluator in ``build_evaluator`` by commenting and
uncommenting the three marked lines. Each case supplies demographics, the
SLO/fundus image, and a grid of representative OCT B-scans. No RETFound or
other specialist-model output is used.
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
    "You are a standalone ophthalmic image evaluator. Determine whether age-related macular "
    "degeneration (AMD) is present using only the supplied SLO/fundus image, representative OCT "
    "B-scans, and basic demographics. Demographics are contextual metadata only and must never "
    "be treated as anatomical proof of disease. Examine the macula for drusen or drusenoid "
    "elevations, pigmentary abnormalities, RPE disruption, geographic atrophy, subretinal or "
    "intraretinal fluid, pigment epithelial detachment, subretinal hyperreflective material, "
    "fibrosis, and other visible AMD-related changes. Consider image quality and agreement between "
    "the fundus and OCT evidence. Do not invent measurements, history, symptoms, visual acuity, "
    "angiography findings, or foundation-model scores. Return JSON only with exactly these fields: "
    "amd_detected (integer 0 or 1), confidence (low, moderate, or high), and reasoning (brief "
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


def make_oct_grid(volume, number_of_slices):
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(volume) == 0:
        raise ValueError(f"Expected non-empty OCT volume [slices,H,W], got {volume.shape}")
    count = max(1, min(number_of_slices, len(volume)))
    indices = np.linspace(0, len(volume) - 1, count, dtype=int)
    images = [ImageOps.contain(as_rgb(volume[index]), (448, 224)) for index in indices]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
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


def data_url(image):
    return f"data:image/jpeg;base64,{jpeg_base64(image)}"


def extract_json_object(text):
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
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse model response: {text[:500]!r}") from exc


def validate_result(raw):
    parsed = extract_json_object(raw)
    prediction = int(parsed["amd_detected"])
    if prediction not in (0, 1):
        raise ValueError(f"Invalid amd_detected value: {prediction}")
    return (
        prediction,
        str(parsed.get("confidence", "")),
        str(parsed.get("reasoning", "")),
        raw,
    )


def user_text(case):
    return (
        "Demographics:\n"
        f"Age: {case['age']}\n"
        f"Gender: {case['gender']}\n"
        f"Race: {case['race']}\n"
        f"Ethnicity: {case['ethnicity']}\n\n"
        "First image: SLO/fundus image. Second image: a grid of evenly spaced OCT B-scans "
        f"from slice indices {case['oct_indices']}. Assess this case for AMD."
    )


class AzureGPTAMDEvaluator:
    def __init__(self, deployment, model_tag):
        self.deployment = deployment
        self.model_tag = model_tag
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
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text(case)},
                        {"type": "image_url", "image_url": {"url": data_url(case["fundus"])}},
                        {"type": "image_url", "image_url": {"url": data_url(case["oct_grid"])}},
                    ],
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        return validate_result(raw)


class GPT51AMDEvaluator(AzureGPTAMDEvaluator):
    def __init__(self):
        super().__init__(os.getenv("GPT51_DEPLOYMENT", "gpt-5.1"), "gpt51")


class GPT56AMDEvaluator(AzureGPTAMDEvaluator):
    def __init__(self):
        super().__init__(os.getenv("GPT56_DEPLOYMENT", "gpt-5.6-luna"), "gpt56")


class Claude45AMDEvaluator:
    model_tag = "claude45"

    def __init__(self):
        self.deployment = os.getenv("ANTHROPIC_DEPLOYMENT", "claude-haiku-4-5")
        base_url = (
            os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
            or os.getenv("AZURE_AI_ANTHROPIC_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if not base_url:
            raise ValueError(
                "Set ANTHROPIC_FOUNDRY_BASE_URL to "
                "https://<resource>.services.ai.azure.com/anthropic"
            )
        if not api_key:
            raise ValueError("Set AZURE_API_KEY (or AZURE_OPENAI_API_KEY)")
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1/messages"):
            base_url = base_url.removesuffix("/v1/messages")
        if not base_url.endswith("/anthropic"):
            raise ValueError(
                "Claude requires a Foundry base URL ending in '/anthropic'; "
                f"received {base_url}"
            )
        self.client = AnthropicFoundry(api_key=api_key, base_url=base_url)

    def analyze(self, case):
        response = self.client.messages.create(
            model=self.deployment,
            max_tokens=1024,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text(case)},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": jpeg_base64(case["fundus"]),
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": jpeg_base64(case["oct_grid"]),
                        },
                    },
                ],
            }],
        )
        raw = "\n".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        return validate_result(raw)


def build_evaluator():
    """COMMENT/UNCOMMENT exactly one of the following three lines."""
    return GPT51AMDEvaluator()
#    return GPT56AMDEvaluator()
#    return Claude45AMDEvaluator()


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
    if column is None or pd.isna(row[column]):
        return default
    return row[column]


def load_amd_rows():
    frame = pd.read_csv(INPUT_CSV)
    disease_column = find_column(frame.columns, ("Task_Folder", "disease_folder", "disease"))
    frame = frame[
        frame[disease_column].astype(str).str.strip().str.lower() == "amd"
    ].reset_index(drop=True)
    if MAX_CASES > 0:
        frame = frame.head(MAX_CASES)
    if frame.empty:
        raise ValueError(f"No AMD cases found in {INPUT_CSV}")
    return frame


def load_case(row):
    path_column = find_column(row.index, ("filepath", "file_path", "path", "npz_path", "filename"))
    truth_column = find_column(row.index, ("Ground_Truth", "groundtruth", "gt", "label"))
    path = Path(str(row[path_column]))
    with np.load(path) as container:
        fundus = as_rgb(container["slo_fundus"])
        oct_grid, oct_indices = make_oct_grid(container["oct_bscans"], OCT_SLICES)
    truth = int(float(row[truth_column]))
    if truth not in (0, 1):
        raise ValueError(f"AMD Ground_Truth must be binary, got {truth} for {path}")
    return {
        "path": path,
        "ground_truth": truth,
        "age": row_value(row, ("Age",)),
        "gender": row_value(row, ("Gender", "Sex")),
        "race": row_value(row, ("Race",)),
        "ethnicity": row_value(row, ("Ethnicity",)),
        "fundus": fundus,
        "oct_grid": oct_grid,
        "oct_indices": oct_indices,
    }


def print_metrics(results):
    valid = results[results["Pred_AMD"].isin([0, 1])]
    failed = len(results) - len(valid)
    if valid.empty:
        print("No valid predictions were produced.")
        return
    truth = valid["Ground_Truth"].astype(int).to_numpy()
    predictions = valid["Pred_AMD"].astype(int).to_numpy()
    print("\nPer-class classification report:")
    print(classification_report(
        truth, predictions, labels=[0, 1], target_names=["No AMD", "AMD"],
        digits=4, zero_division=0,
    ))
    print(f"Accuracy: {accuracy_score(truth, predictions):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(truth, predictions, labels=[0, 1]))
    print(f"Valid predictions: {len(valid)} | Failed/invalid predictions: {failed}")


def main():
    frame = load_amd_rows()
    evaluator = build_evaluator()
    output_csv = os.getenv(
        "OUTPUT_CSV", f"fairvision_amd_{evaluator.model_tag}_predictions.csv"
    )
    rows = []
    print(
        f"Evaluating {evaluator.deployment!r} on {len(frame)} FairVision AMD cases; "
        f"output={output_csv}"
    )
    for position, (_, source_row) in enumerate(frame.iterrows(), start=1):
        case = None
        print("\n" + "=" * 90 + f"\nCASE {position}/{len(frame)}")
        try:
            case = load_case(source_row)
            prediction, confidence, reasoning, raw = evaluator.analyze(case)
            print(f"Filename: {case['path']}")
            print(
                f"Ground truth: {case['ground_truth']} | Prediction: {prediction} "
                f"| Confidence: {confidence}"
            )
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == case['ground_truth']}")
            result = {
                "Filename": str(case["path"]), "Model": evaluator.deployment,
                "Age": case["age"], "Gender": case["gender"], "Race": case["race"],
                "Ethnicity": case["ethnicity"], "Ground_Truth": case["ground_truth"],
                "Pred_AMD": prediction, "Confidence": confidence,
                "Reasoning": reasoning, "Raw_Response": raw,
                "Is_Correct": int(prediction == case["ground_truth"]),
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            result = {
                "Filename": str(case["path"]) if case else str(row_value(source_row, ("filename",), "Error")),
                "Model": evaluator.deployment,
                "Ground_Truth": case["ground_truth"] if case else row_value(source_row, ("Ground_Truth",), -1),
                "Pred_AMD": -1, "Confidence": "", "Reasoning": "", "Raw_Response": "",
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
