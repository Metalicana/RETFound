import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_agentic_walkthrough_figure import OUT, validate_case, mask_measurements, overlay_mask


class AgenticWalkthroughTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = json.loads((OUT / "case_data.json").read_text())
        cls.measurements = cls.case["segmentation"]
        cls.root = ET.parse(OUT / "agentic_walkthrough.drawio").getroot()
        cls.labels = [c.get("value", "") for c in cls.root.iter("mxCell")]

    def test_case_is_paired_with_correct_inputs(self):
        validate_case(self.case, self.measurements)
        self.assertEqual(self.case["foundations"]["retfound"]["prediction"], 0)
        self.assertEqual(self.case["agent"]["prediction"], 1)
        self.assertAlmostEqual(self.case["agent"]["input_probability_pct"], 21.90499899301915)

    def test_rejects_changed_model_checkpoint_or_case(self):
        bad = copy.deepcopy(self.case)
        bad["agent"]["raw_probability"] = .28
        with self.assertRaisesRegex(ValueError, "different predictions"):
            validate_case(bad, self.measurements)
        bad = copy.deepcopy(self.case)
        bad["case_id"] = "drishtiGS_054"
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            validate_case(bad, self.measurements)

    def test_rejects_different_segmentation_ratio(self):
        bad = copy.deepcopy(self.measurements)
        bad["vertical_cdr"] = .65
        with self.assertRaisesRegex(ValueError, "historical CDR"):
            validate_case(self.case, bad)

    def test_rejects_manufactured_ablation_label(self):
        bad = copy.deepcopy(self.case)
        bad["agent"]["scenarios"]["without_visual_interpretation"] = 1
        with self.assertRaisesRegex(ValueError, "scenarios changed"):
            validate_case(bad, self.measurements)

    def test_assets_match_export_and_contours_match_mask(self):
        for name in ("fundus.png", "mask.png", "overlay.png"):
            path = OUT / "assets" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.measurements["files"][name])
        mask = np.asarray(Image.open(OUT / "assets/mask.png"))
        derived = mask_measurements(mask)
        self.assertAlmostEqual(derived["vertical_cdr"], .6902654867256637)
        self.assertAlmostEqual(derived["area_cdr"], .42203534094178574)
        image = Image.open(OUT / "assets/fundus.png")
        overlay = Image.open(OUT / "assets/overlay.png")
        np.testing.assert_array_equal(overlay, overlay_mask(image, mask))

    def test_all_agents_and_provenance_are_visible(self):
        text = "\n".join(self.labels)
        for label in ("Read the image", "Check the reasoning", "Explain the assessment",
                      "Summarize\npatient history", "Check reliability\nby patient group",
                      "literature (PubMed)", "visual-field tests", "conflicting data"):
            self.assertIn(label, text)
        modules = {c.get("data-module") for c in self.root.iter("mxCell") if c.get("data-module")}
        self.assertEqual(modules, {"Vision Agent", "Counterfactual Agent", "Orchestrator", "BioProfiler",
                                  "Model tools", "Equity Agent", "Guidelines Agent", "Functional Agent", "Safety Agent"})
        self.assertIn("not used in this case", text)
        self.assertIn("Recorded case: RETFound + AI image report + cup-to-disc ratio", text)
        self.assertIn("AI's image report", text)
        self.assertNotIn("human CFP specialist", text)
        caption = (OUT / "README.md").read_text()
        self.assertIn("did not use the other four", caption)
        self.assertIn("21.905%", caption)
        self.assertIn("not independently rerun", caption)
        self.assertIn("Without RETFound probability", caption)
        self.assertIn("Without CDR", caption)
        for model in ("MIRAGE", "RET-CLIP", "RetiZero", "URFound"):
            self.assertNotIn(model, text)
            self.assertIn(model, caption)

    def test_clinical_story_names_the_conflict_and_human_decision(self):
        text = "\n".join(self.labels)
        for phrase in ("Could this be\nglaucoma?", "Model cutoff: 53%", "Model vote: non-glaucoma",
                       "model vote and image findings disagree", "AI assessment: glaucoma",
                       "Photograph-only evidence", "Doctor reviews evidence", "makes the final decision"):
            self.assertIn(phrase, text)
        for jargon in ("Raw FM scores", "Reliability R", "Counterfactual Agent", "Orchestrator", "vCDR"):
            self.assertNotIn(jargon, text)

    def test_key_messages_use_editable_speech_bubbles(self):
        bubbles = {c.get("data-bubble"): c for c in self.root.iter("mxCell") if c.get("data-bubble")}
        self.assertEqual(set(bubbles), {"clinician-request", "image-report", "evidence-ablation", "recorded-assessment"})
        self.assertTrue(all("rounded=1" in cell.get("style", "") for cell in bubbles.values()))

    def test_native_editable_shapes_and_real_raster_images(self):
        cells = list(self.root.iter("mxCell"))
        image_cells = [c for c in cells if "shape=image;" in c.get("style", "")]
        self.assertEqual(len(image_cells), 2)
        self.assertTrue(all("data:image/png," in c.get("style", "") for c in image_cells))
        self.assertGreater(len([c for c in cells if c.get("edge") == "1"]), 40)
        self.assertGreater(len([c for c in cells if c.get("value")]), 30)


if __name__ == "__main__":
    unittest.main()
