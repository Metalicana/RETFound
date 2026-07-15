"""GPT-5.1 specialist for GDP RNFL thickness maps."""

import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()


class RNFLTSpecialist:
    def __init__(self, model_client=None):
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
        self.client = model_client or AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    @staticmethod
    def _render(rnflt):
        finite = rnflt[np.isfinite(rnflt)]
        if finite.size == 0:
            raise ValueError("RNFLT map contains no finite values")
        low, high = np.percentile(finite, [1, 99])
        figure, axis = plt.subplots(figsize=(6, 6), dpi=140)
        image = axis.imshow(rnflt, cmap="turbo", vmin=low, vmax=high)
        axis.set_title("RNFL Thickness Map")
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Thickness value")
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(figure)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        statistics = {
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p05": float(np.percentile(finite, 5)),
            "p95": float(np.percentile(finite, 95)),
        }
        return f"data:image/png;base64,{encoded}", statistics

    def analyze(self, rnflt):
        image_url, statistics = self._render(rnflt)
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ophthalmic imaging specialist reviewing an RNFL thickness map for "
                        "glaucoma-related structural evidence. Describe map quality, global versus focal "
                        "thinning patterns, superior/inferior arcuate or wedge-like defects, asymmetry, and "
                        "preserved regions. The displayed colors are scaled to this case, so use the colorbar "
                        "and supplied summary statistics and do not assume a color has a universal normal or "
                        "abnormal meaning. Do not invent normative percentiles, OCT findings outside this map, "
                        "optic-disc findings, CDR, visual fields, or a final glaucoma probability. Do not give "
                        "the final binary diagnosis. Structure the response as 'RNFLT Glaucoma-Relevant "
                        "Features:' followed by 'Overall RNFLT Impression:'."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this RNFLT map. Summary statistics: {statistics}",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        return response.choices[0].message.content, statistics
