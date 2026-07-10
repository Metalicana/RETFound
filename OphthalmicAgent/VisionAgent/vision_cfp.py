import base64
import io
import os

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from openai import AzureOpenAI
from torchvision import transforms

from VisionAgent.linear_probing_fundus import get_model_cfp


load_dotenv()


class VisionSpecialistCFP:
    """RETFound-CFP glaucoma scoring plus CFP-specific visual interpretation."""

    def __init__(self, weights_path, device=None, model_client=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model_cfp()
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model_client = model_client or AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")

    @staticmethod
    def _image(value):
        if isinstance(value, (str, os.PathLike)):
            return Image.open(value).convert("RGB")
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        array = np.asarray(value)
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            low, high = float(array.min()), float(array.max())
            if high > low:
                array = 255 * (array - low) / (high - low)
            array = array.astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    @staticmethod
    def _data_url(image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def analyze(self, cfp_image, state):
        image = self._image(cfp_image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = float(torch.sigmoid(self.model(tensor)).reshape(-1)[0]) * 100
        probability = round(probability, 2)
        state["cfp_diagnosis"] = {
            "Glaucoma": {
                "Prob_Pct": probability,
                "Status": "Positive" if probability >= 50 else "Negative",
            }
        }

        response = self.model_client.chat.completions.create(
            model=self.deployment,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ophthalmic imaging specialist reviewing one color fundus photograph "
                        "for glaucoma. Describe image quality, optic-disc appearance, neuroretinal rim, "
                        "cupping, disc asymmetry visible in this image, peripapillary changes, hemorrhage, "
                        "and other glaucoma-relevant findings. Do not invent measurements or use the AI "
                        "score as visual evidence. End with IMPRESSION: supports glaucoma, supports normal, "
                        "or indeterminate."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this CFP for glaucoma-related structural findings."},
                        {"type": "image_url", "image_url": {"url": self._data_url(image)}},
                    ],
                },
            ],
        )
        report = response.choices[0].message.content
        scores = f"Glaucoma probability from RETFound-CFP: {probability}%"
        return scores, report, probability
