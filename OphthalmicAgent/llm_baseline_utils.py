"""Shared utilities for single-call GPT-5.1 glaucoma baselines."""

import base64
import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


load_dotenv()


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


def as_rgb(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.fromarray(normalize_uint8(value)).convert("RGB")


def image_data_url(image, image_format="JPEG"):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, quality=95 if image_format == "JPEG" else None)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if image_format == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def make_oct_grid(volume, number_of_slices=8):
    count = max(1, min(number_of_slices, len(volume)))
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
    return grid


def render_rnflt(rnflt):
    finite = rnflt[np.isfinite(rnflt)]
    if finite.size == 0:
        raise ValueError("RNFLT map contains no finite values")
    low, high = np.percentile(finite, [1, 99])
    figure, axis = plt.subplots(figsize=(6, 6), dpi=140)
    plot = axis.imshow(rnflt, cmap="turbo", vmin=low, vmax=high)
    axis.set_title("RNFL Thickness Map")
    axis.axis("off")
    figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04, label="Thickness value")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return Image.open(buffer).convert("RGB"), {
        "minimum": float(np.min(finite)), "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)), "p95": float(np.percentile(finite, 95)),
    }


class GPT51GlaucomaBaseline:
    def __init__(self):
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    def analyze(self, images, demographics, image_description, extra_context=""):
        content = [{
            "type": "text",
            "text": (
                f"Available demographics: {json.dumps(demographics, sort_keys=True)}\n"
                f"Visual inputs: {image_description}\n{extra_context}\n"
                "Assess this case for glaucoma."
            ),
        }]
        content.extend({"type": "image_url", "image_url": {"url": image_data_url(image)}} for image in images)
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a standalone ophthalmic image evaluator. Diagnose glaucoma using only the "
                        "provided visual data and available demographics. Demographics are contextual metadata "
                        "only and must never be treated as anatomical proof of disease. Evaluate image quality "
                        "and visible glaucoma-related structural evidence, integrate the supplied modalities, "
                        "and describe agreement, discordance, and uncertainty. Do not invent unavailable "
                        "measurements, symptoms, history, intraocular pressure, visual fields, CDR, or AI model "
                        "scores. Return JSON only with exactly: glaucoma_detected (integer 0 or 1), confidence "
                        "(low, moderate, or high), reasoning (brief evidence-based explanation). You must choose "
                        "0 or 1 even when uncertain."
                    ),
                },
                {"role": "user", "content": content},
            ],
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        prediction = int(parsed["glaucoma_detected"])
        if prediction not in (0, 1):
            raise ValueError(f"Invalid glaucoma_detected value: {prediction}")
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
    print(classification_report(y_true, y_pred, labels=[0, 1],
        target_names=["Normal", "Glaucoma"], digits=4, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(f"Valid predictions: {len(valid)} | Failed/invalid predictions: {failed}")


def checkpoint_and_report(rows, output_csv):
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    return frame
