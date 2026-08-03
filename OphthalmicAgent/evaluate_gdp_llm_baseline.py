# Previous GPT baseline (kept commented for reference).
#
# import os
# from data.gdp_loader import GDPTestLoader
# from llm_baseline_utils import (
#     GPT56GlaucomaBaseline, checkpoint_and_report, make_oct_grid,
#     print_metrics, render_rnflt,
# )
#
# CSV_PATH = os.getenv("GDP_CSV", "./data_gdp/data_summary.csv")
# BSCAN_DIR = os.getenv("GDP_BSCAN_DIR", "./data_gdp/BScan")
# RNFLT_DIR = os.getenv("GDP_RNFLT_DIR", "./data_gdp/RNFLT")
# OUTPUT_CSV = os.getenv("OUTPUT_CSV", "gdp_test_gpt56_baseline_predictions.csv")
# MAX_CASES = int(os.getenv("MAX_CASES", "0"))
# OCT_SLICES = int(os.getenv("OCT_SLICES", "8"))
#
# def main():
#     loader = GDPTestLoader(
#         CSV_PATH, BSCAN_DIR, RNFLT_DIR, OCT_SLICES, MAX_CASES,
#         require_rnflt=True,
#     )
#     evaluator, rows = GPT56GlaucomaBaseline(), []
#     for index in range(len(loader)):
#         case = loader.load(index)
#         oct_grid = make_oct_grid(case["oct_slices"], OCT_SLICES)
#         rnflt_image, rnflt_stats = render_rnflt(case["rnflt"])
#         demographics = {
#             "age": case["age"], "gender": case["gender"], "race": case["race"]
#         }
#         prediction, confidence, reasoning, raw = evaluator.analyze(
#             [oct_grid, rnflt_image], demographics,
#             "First image: eight evenly spaced OCT B-scans. Second image: RNFL thickness map.",
#             f"RNFLT case-scaled colorbar statistics: {rnflt_stats}",
#         )
#         rows.append({
#             "Patient_ID": case["patient_id"], "Ground_Truth": case["ground_truth"],
#             "Pred_GL": prediction, "Confidence": confidence, "Reasoning": reasoning,
#             "Raw_Response": raw,
#             "Is_Correct": int(prediction == case["ground_truth"]),
#         })
#         checkpoint_and_report(rows, OUTPUT_CSV)
#     print_metrics(checkpoint_and_report(rows, OUTPUT_CSV))
#
# if __name__ == "__main__":
#     main()


"""Standalone Claude baseline on GDP Test: OCT + RNFLT + demographics."""

import base64
import io
import json
import os

from anthropic import AnthropicFoundry
from dotenv import load_dotenv

from data.gdp_loader import GDPTestLoader
from llm_baseline_utils import (
    checkpoint_and_report,
    make_oct_grid,
    print_metrics,
    render_rnflt,
)


load_dotenv()

CSV_PATH = os.getenv("GDP_CSV", "./data_gdp/data_summary.csv")
BSCAN_DIR = os.getenv("GDP_BSCAN_DIR", "./data_gdp/BScan")
RNFLT_DIR = os.getenv("GDP_RNFLT_DIR", "./data_gdp/RNFLT")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "gdp_test_claude_baseline_predictions.csv")
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
OCT_SLICES = int(os.getenv("OCT_SLICES", "8"))
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

    def analyze(self, images, demographics, rnflt_stats):
        content = [
            {
                "type": "text",
                "text": (
                    f"Available demographics: {json.dumps(demographics, sort_keys=True)}\n"
                    "First image: a grid of eight evenly spaced OCT B-scans.\n"
                    "Second image: an RNFL thickness heatmap.\n"
                    f"RNFL map summary statistics: {json.dumps(rnflt_stats, sort_keys=True)}\n"
                    "Assess this case for glaucoma."
                ),
            }
        ]
        for image in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64(image),
                },
            })
        response = self.client.messages.create(
            model=DEPLOYMENT,
            max_tokens=1024,
            temperature=0.2,
            system=(
                "You are a standalone ophthalmic image evaluator. Diagnose glaucoma using only "
                "the supplied OCT B-scans, RNFL thickness map, RNFL summary statistics, and basic "
                "demographics. Demographics are contextual metadata only and must not be treated "
                "as anatomical evidence. Examine visible optic-nerve and retinal nerve fiber layer "
                "patterns, image quality, and agreement between the OCT and RNFL evidence. Do not "
                "invent SLO, CFP, CDR, intraocular pressure, visual fields, symptoms, history, or "
                "foundation-model scores. Return JSON only with exactly these fields: "
                "glaucoma_detected (integer 0 or 1), confidence (low, moderate, or high), and "
                "reasoning (brief evidence-based explanation). Choose 0 or 1 even when uncertain."
            ),
            messages=[{"role": "user", "content": content}],
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
    loader = GDPTestLoader(
        CSV_PATH,
        BSCAN_DIR,
        RNFLT_DIR,
        OCT_SLICES,
        MAX_CASES,
        require_rnflt=True,
    )
    evaluator = StandaloneClaudeGlaucomaEvaluator()
    rows = []
    print(f"Evaluating Claude deployment {DEPLOYMENT!r} on {len(loader)} GDP Test cases")

    for index in range(len(loader)):
        case = None
        print("\n" + "=" * 90 + f"\nCASE {index + 1}/{len(loader)}")
        try:
            case = loader.load(index)
            oct_grid = make_oct_grid(case["oct_slices"], OCT_SLICES)
            rnflt_image, rnflt_stats = render_rnflt(case["rnflt"])
            demographics = {
                "age": case["age"],
                "gender": case["gender"],
                "race": case["race"],
            }
            prediction, confidence, reasoning, raw = evaluator.analyze(
                [oct_grid, rnflt_image], demographics, rnflt_stats
            )
            print(f"Patient: {case['patient_id']} | Demographics: {demographics}")
            print(
                f"Ground truth: {case['ground_truth']} | Claude prediction: {prediction} "
                f"| Confidence: {confidence}"
            )
            print(f"Reasoning: {reasoning}\nCorrect: {prediction == case['ground_truth']}")
            row = {
                "Patient_ID": case["patient_id"],
                "Age": case["age"],
                "Gender": case["gender"],
                "Race": case["race"],
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
                "Patient_ID": case["patient_id"] if case else f"index_{index}",
                "Ground_Truth": case["ground_truth"] if case else -1,
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
