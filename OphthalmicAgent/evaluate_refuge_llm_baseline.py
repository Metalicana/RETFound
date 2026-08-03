# Previous GPT baseline (kept commented for reference).
#
# import os
# from pathlib import Path
# import pandas as pd
# from PIL import Image
# from llm_baseline_utils import (
#     GPT56GlaucomaBaseline, checkpoint_and_report, print_metrics,
# )
#
# DATA_ROOT = Path(os.getenv("REFUGE_DATA_ROOT", "./"))
# CSV_PATH = Path(os.getenv("REFUGE_CSV", "./data_refuge/data.csv"))
# OUTPUT_CSV = os.getenv("OUTPUT_CSV", "refuge_test_gpt56_baseline_predictions.csv")
# MAX_CASES = int(os.getenv("MAX_CASES", "0"))
#
# def cases():
#     frame = pd.read_csv(CSV_PATH)
#     frame = frame[frame["filename"].astype(str).str.contains(
#         r"[\\/]Test[\\/]", case=False, regex=True
#     )]
#     if MAX_CASES > 0:
#         frame = frame.head(MAX_CASES)
#     return frame.reset_index(drop=True)
#
# def main():
#     frame, evaluator, rows = cases(), GPT56GlaucomaBaseline(), []
#     for _, row in frame.iterrows():
#         relative = Path(str(row["filename"]))
#         path = relative if relative.is_absolute() else DATA_ROOT / relative
#         prediction, confidence, reasoning, raw = evaluator.analyze(
#             [Image.open(path).convert("RGB")],
#             {"status": "not available in REFUGE manifest"},
#             "One color fundus photograph.",
#         )
#         truth = int(row["Ground_Truth"])
#         rows.append({
#             "Filename": str(path), "Ground_Truth": truth, "Pred_GL": prediction,
#             "Confidence": confidence, "Reasoning": reasoning, "Raw_Response": raw,
#             "Is_Correct": int(prediction == truth),
#         })
#         checkpoint_and_report(rows, OUTPUT_CSV)
#     print_metrics(checkpoint_and_report(rows, OUTPUT_CSV))
#
# if __name__ == "__main__":
#     main()


"""Standalone Claude CFP baseline on the REFUGE Test split."""

import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
from anthropic import AnthropicFoundry
from dotenv import load_dotenv
from PIL import Image

from llm_baseline_utils import checkpoint_and_report, print_metrics


load_dotenv()

DATA_ROOT = Path(os.getenv("REFUGE_DATA_ROOT", "./"))
CSV_PATH = Path(os.getenv("REFUGE_CSV", "./data_refuge/data.csv"))
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "refuge_test_claude_baseline_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
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


def load_test_cases():
    frame = pd.read_csv(CSV_PATH)
    required = {"filename", "Ground_Truth"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"REFUGE CSV is missing columns: {sorted(missing)}")
    frame = frame[frame["filename"].astype(str).str.contains(
        r"[\\/]Test[\\/]", case=False, regex=True
    )].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No /Test/ rows found in {CSV_PATH}")
    if MAX_CASES > 0:
        frame = frame.head(MAX_CASES)
    return frame


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
                "available for this dataset. Examine image quality, optic-disc cupping, neuroretinal "
                "rim thinning or notching, vessel displacement or bayoneting, disc hemorrhage, "
                "peripapillary changes, and other visible glaucoma-related structural evidence. "
                "Do not invent CDR measurements, intraocular pressure, visual fields, symptoms, "
                "history, demographics, OCT findings, or foundation-model scores. Return JSON only "
                "with exactly these fields: glaucoma_detected (integer 0 or 1), confidence (low, "
                "moderate, or high), and reasoning (brief evidence-based explanation). Choose 0 or "
                "1 even when uncertain."
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
    frame = load_test_cases()
    evaluator = StandaloneClaudeGlaucomaEvaluator()
    rows = []
    print(f"Evaluating Claude deployment {DEPLOYMENT!r} on {len(frame)} REFUGE Test CFPs")

    for index, source_row in frame.iterrows():
        relative = Path(str(source_row["filename"]))
        path = relative if relative.is_absolute() else DATA_ROOT / relative
        truth = int(source_row["Ground_Truth"])
        print("\n" + "=" * 90 + f"\nCASE {index + 1}/{len(frame)} | {path}")
        try:
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
            prediction, confidence, reasoning, raw = evaluator.analyze(image)
            print(
                f"Ground truth: {truth} | Claude prediction: {prediction} "
                f"| Confidence: {confidence}"
            )
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == truth}")
            result = {
                "Filename": str(path),
                "Ground_Truth": truth,
                "Pred_GL": prediction,
                "Confidence": confidence,
                "Reasoning": reasoning,
                "Raw_Response": raw,
                "Is_Correct": int(prediction == truth),
            }
        except Exception as exc:
            print(f"!!! Error: {exc}")
            result = {
                "Filename": str(path),
                "Ground_Truth": truth,
                "Pred_GL": -1,
                "Confidence": "",
                "Reasoning": "",
                "Raw_Response": "",
                "Is_Correct": -1,
                "Error": str(exc),
            }
        rows.append(result)
        checkpoint_and_report(rows, OUTPUT_CSV)
        print(f"Checkpoint saved: {OUTPUT_CSV}")

    results = checkpoint_and_report(rows, OUTPUT_CSV)
    print_metrics(results)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
