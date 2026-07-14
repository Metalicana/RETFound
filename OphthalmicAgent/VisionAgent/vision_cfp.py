import base64
import io
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import find_objects
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
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
        self.cdr_model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
        self.cdr_processor = SegformerImageProcessor.from_pretrained(self.cdr_model_name)
        self.cdr_model = SegformerForSemanticSegmentation.from_pretrained(
            self.cdr_model_name
        ).to(self.device).eval()
        self.cdr_output_dir = Path(
            os.getenv("CDR_OUTPUT_DIR", "outputs/drishti_cfp/cdr_segmentations")
        )
        self.save_cdr_segmentations = os.getenv(
            "SAVE_CDR_SEGMENTATIONS", "1"
        ).lower() not in {"0", "false", "no"}

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

    @staticmethod
    def _vertical_cdr(mask):
        disc_mask = (mask == 1) | (mask == 2)
        cup_mask = mask == 2
        if not np.any(disc_mask) or not np.any(cup_mask):
            return None
        disc_box = find_objects(disc_mask)[0]
        cup_box = find_objects(cup_mask)[0]
        disc_height = disc_box[0].stop - disc_box[0].start
        cup_height = cup_box[0].stop - cup_box[0].start
        if disc_height <= 0:
            return None
        return round(cup_height / disc_height, 3)

    def _save_cdr_segmentation(self, image, mask, case_id):
        self.cdr_output_dir.mkdir(parents=True, exist_ok=True)
        safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id)).strip("._")
        safe_case_id = safe_case_id or "cfp"

        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
        overlay = original.copy()
        disc_only = mask == 1
        cup = mask == 2

        # Red = optic disc, blue = optic cup.
        overlay[disc_only] = (
            0.55 * overlay[disc_only] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        overlay[cup] = (
            0.45 * overlay[cup] + 0.55 * np.array([0, 80, 255])
        ).astype(np.uint8)

        # Place the unmodified CFP and segmentation overlay side by side.
        combined = Image.new("RGB", (image.width * 2, image.height))
        combined.paste(image.convert("RGB"), (0, 0))
        combined.paste(Image.fromarray(overlay), (image.width, 0))
        output_path = self.cdr_output_dir / f"{safe_case_id}_cdr_overlay.png"
        combined.save(output_path)
        return output_path

    def calculate_cdr(self, image, case_id):
        equalized = ImageOps.equalize(image.convert("RGB"))
        inputs = self.cdr_processor(images=equalized, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.cdr_model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=(equalized.height, equalized.width),
            mode="bilinear",
            align_corners=False,
        )
        mask = logits.argmax(dim=1)[0].cpu().numpy()
        cdr = self._vertical_cdr(mask)
        output_path = None
        if self.save_cdr_segmentations:
            output_path = self._save_cdr_segmentation(image, mask, case_id)
        return cdr, output_path

    def analyze(self, cfp_image, state):
        image = self._image(cfp_image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = float(torch.sigmoid(self.model(tensor)).reshape(-1)[0]) * 100
        probability = round(probability, 2)
        vertical_cdr, cdr_output_path = self.calculate_cdr(
            image, state.get("patient_id", "cfp")
        )
        state["cdr_segmentation_path"] = (
            str(cdr_output_path) if cdr_output_path is not None else None
        )
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
                        "for glaucoma. Give the optic nerve head and glaucomatous cupping the highest "
                        "attention. First assess whether the disc is sufficiently visible and gradable. "
                        "Then describe vertical cupping, neuroretinal-rim width and focal thinning or "
                        "notching, superior-inferior asymmetry, vessel displacement or bayoneting, laminar "
                        "dot visibility, disc hemorrhage, and peripapillary atrophy. Mention other retinal "
                        "findings only when relevant. Do not invent a numerical cup-to-disc ratio, do not "
                        "infer findings from the AI score or calculated CDR, and do not call an image normal "
                        "merely because obvious advanced cupping is absent. End with IMPRESSION: supports "
                        "glaucoma, supports normal, or indeterminate, and state which optic-disc features "
                        "most influenced that impression."
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
        return scores, report, probability, vertical_cdr
