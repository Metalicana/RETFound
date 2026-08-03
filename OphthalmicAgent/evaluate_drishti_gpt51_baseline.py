# Previous GPT baseline (kept commented for reference).
#
# import os
# from pathlib import Path
# from PIL import Image
# from llm_baseline_utils import (
#     GPT56GlaucomaBaseline, checkpoint_and_report, print_metrics,
# )
#
# DATA_ROOT = Path(os.getenv("DRISHTI_DATA_ROOT", "./data_drishti"))
# OUTPUT_CSV = os.getenv("OUTPUT_CSV", "drishti_gpt56_baseline_predictions.csv")
# MAX_CASES = int(os.getenv("MAX_CASES", "0"))
# EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
#
# def find_folder(name):
#     match = next(
#         (path for path in DATA_ROOT.iterdir()
#          if path.is_dir() and path.name.lower() == name),
#         None,
#     )
#     if match is None:
#         raise FileNotFoundError(f"Missing {name} folder under {DATA_ROOT}")
#     return match
#
# def cases():
#     found = []
#     for name, label in (("normal", 0), ("glaucoma", 1)):
#         found.extend(
#             {"path": path, "ground_truth": label}
#             for path in sorted(find_folder(name).rglob("*"))
#             if path.is_file() and path.suffix.lower() in EXTENSIONS
#         )
#     return found[:MAX_CASES] if MAX_CASES > 0 else found
#
# def main():
#     dataset, evaluator, rows = cases(), GPT56GlaucomaBaseline(), []
#     for case in dataset:
#         prediction, confidence, reasoning, raw = evaluator.analyze(
#             [Image.open(case["path"]).convert("RGB")],
#             {"status": "not available in Drishti folders"},
#             "One color fundus photograph.",
#         )
#         rows.append({
#             "Filename": str(case["path"]),
#             "Ground_Truth": case["ground_truth"],
#             "Pred_GL": prediction,
#             "Confidence": confidence,
#             "Reasoning": reasoning,
#             "Raw_Response": raw,
#             "Is_Correct": int(prediction == case["ground_truth"]),
#         })
#         checkpoint_and_report(rows, OUTPUT_CSV)
#     print_metrics(checkpoint_and_report(rows, OUTPUT_CSV))
#
# if __name__ == "__main__":
#     main()


"""Standalone Claude CFP baseline on Drishti Normal/Glaucoma folders."""

import base64
import io
import json
import os
from pathlib import Path

from anthropic import AnthropicFoundry
from dotenv import load_dotenv
from PIL import Image

from llm_baseline_utils import checkpoint_and_report, print_metrics


load_dotenv()

DATA_ROOT = Path(os.getenv("DRISHTI_DATA_ROOT", "./data_drishti"))
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "drishti_claude_baseline_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPLOYMENT = os.getenv(
    "ANTHROPIC_DEPLOYMENT",
    os.getenv("AZURE_OPENAI_DEPLOYMENT", "claude-haiku-4-5"),
)


def image_base64(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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
            raise ValueError(f"Claude did not return JSON: {text[:300]!r}")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse Claude response: {text[:500]!r}") from exc


def find_folder(name):
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Drishti data root not found: {DATA_ROOT}")
    match = next(
        (path for path in DATA_ROOT.iterdir()
         if path.is_dir() and path.name.lower() == name.lower()),
        None,
    )
    if match is None:
        raise FileNotFoundError(f"Missing {name} folder under {DATA_ROOT}")
    return match


def load_cases():
    found = []
    for name, label in (("normal", 0), ("glaucoma", 1)):
        found.extend(
            {"path": path, "ground_truth": label}
            for path in sorted(find_folder(name).rglob("*"))
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
    if not found:
        raise ValueError(f"No Drishti CFP images found under {DATA_ROOT}")
    return found[:MAX_CASES] if MAX_CASES > 0 else found


class StandaloneClaudeGlaucomaEvaluator:
    def __init__(self):
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

    def analyze(self, image):
        response = self.client.messages.create(
            model=DEPLOYMENT,
            max_tokens=1024,
            temperature=0.2,
            system=(
                "You are a standalone ophthalmic image evaluator. Determine whether glaucoma is "
                "present using only the supplied color fundus photograph. No demographics are "
                "available. Examine image quality, optic-disc cupping, neuroretinal rim thinning "
                "or notching, vessel displacement or bayoneting, disc hemorrhage, peripapillary "
                "changes, and other visible glaucoma-related structural evidence. Do not invent "
                "CDR measurements, intraocular pressure, visual fields, symptoms, history, "
                "demographics, OCT findings, or foundation-model scores. Return JSON only with "
                "exactly these fields: glaucoma_detected (integer 0 or 1), confidence (low, "
                "moderate, or high), and reasoning (brief evidence-based explanation). Choose "
                "0 or 1 even when uncertain."
            ),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Assess this color fundus photograph for glaucoma.",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64(image),
                        },
                    },
                ],
            }],
        )
        raw = "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        parsed = extract_json_object(raw)
        prediction = int(parsed["glaucoma_detected"])
        if prediction not in (0, 1):
            raise ValueError(f"Claude returned invalid glaucoma_detected: {prediction}")
        return (
            prediction,
            str(parsed.get("confidence", "")),
            str(parsed.get("reasoning", "")),
            raw,
        )


def main():
    dataset = load_cases()
    evaluator = StandaloneClaudeGlaucomaEvaluator()
    rows = []
    print(f"Evaluating Claude deployment {DEPLOYMENT!r} on {len(dataset)} Drishti CFPs")

    for index, case in enumerate(dataset, start=1):
        print("\n" + "=" * 90 + f"\nCASE {index}/{len(dataset)} | {case['path']}")
        try:
            with Image.open(case["path"]) as source_image:
                image = source_image.convert("RGB")
            prediction, confidence, reasoning, raw = evaluator.analyze(image)
            print(
                f"Ground truth: {case['ground_truth']} | Claude prediction: {prediction} "
                f"| Confidence: {confidence}"
            )
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == case['ground_truth']}")
            row = {
                "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"],
                "Pred_GL": prediction,
                "Confidence": confidence,
                "Reasoning": reasoning,
                "Raw_Response": raw,
                "Is_Correct": int(prediction == case["ground_truth"]),
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            row = {
                "Filename": str(case["path"]),
                "Ground_Truth": case["ground_truth"],
                "Pred_GL": -1,
                "Confidence": "",
                "Reasoning": "",
                "Raw_Response": "",
                "Is_Correct": -1,
                "Error": str(exc),
            }
        rows.append(row)
        checkpoint_and_report(rows, OUTPUT_CSV)
        print(f"Checkpoint saved: {OUTPUT_CSV}")

    results = checkpoint_and_report(rows, OUTPUT_CSV)
    print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
