"""Lightweight figure integrity checks; no cluster dependencies or inference."""

import base64
import io
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_motivation_options as figures


def prediction(case="case_a", truth="1", pred="1"):
    return dict(image_id=case, y_true=truth, y_pred=pred)


class CohortTests(unittest.TestCase):
    def test_all_four_paired_transitions(self):
        r = [prediction(str(i), "1", str(a)) for i, a in enumerate((1, 1, 0, 0))]
        v = [prediction(str(i), "1", str(a)) for i, a in enumerate((1, 0, 1, 0))]
        counts, _, _ = figures.paired_foundations(r, v)
        self.assertEqual(counts, dict(n=4, both_correct=1, retfound_only=1, visionfm_only=1, both_wrong=1))

    def test_duplicate_case_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            figures.paired_foundations([prediction(), prediction()], [prediction()])

    def test_different_cohort_fails(self):
        with self.assertRaisesRegex(ValueError, "cohorts differ"):
            figures.paired_foundations([prediction()], [prediction("other")])

    def test_changed_truth_fails(self):
        with self.assertRaisesRegex(ValueError, "Truth mismatch"):
            figures.paired_foundations([prediction()], [prediction(truth="0")])

    def test_nonbinary_prediction_fails(self):
        with self.assertRaisesRegex(ValueError, "Invalid prediction"):
            figures.paired_foundations([prediction(pred="-1")], [prediction()])

    def test_bins_use_probability_not_correctness(self):
        rows = []
        for model in ("retfound_oct", "urfound_oct", "visionfm_oct"):
            for i, (prob, truth) in enumerate(((.89, 1), (.9, 1), (1., 0))):
                rows.append(dict(model_name=model, image_id=str(i), y_prob=prob, y_true=truth,
                                 correct=0, split="val"))
        for row in figures.calibration_rows(rows):
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["observed_positive_fraction"], .5)
            self.assertAlmostEqual(row["mean_probability"], .95)

    def test_calibration_rejects_test_data(self):
        rows = [dict(model_name="retfound_oct", image_id="a", y_prob=.95, y_true=1, split="test")]
        with self.assertRaisesRegex(ValueError, "validation"):
            figures.calibration_rows(rows)

    def test_calibration_rejects_invalid_label(self):
        with self.assertRaisesRegex(ValueError, "Invalid calibration"):
            figures.calibration_rows([dict(y_true=-1, y_prob=.95)])

    def test_calibration_rejects_out_of_range_probability(self):
        with self.assertRaisesRegex(ValueError, "Invalid calibration"):
            figures.calibration_rows([dict(y_true=1, y_prob=95)])


class LayoutTests(unittest.TestCase):
    def tearDown(self):
        figures.plt.close("all")

    def test_five_different_layouts_and_all_text_fits(self):
        evidence = json.loads((figures.OUT / "source_data.json").read_text())
        images = {e["case_id"]: {m: np.zeros((200, 200), np.uint8) for m in ("OCT", "SLO")}
                  for e in evidence["agent"]["examples"] + evidence["foundation_examples"]}
        pages = [builder(evidence, images) for builder in figures.BUILDERS]
        self.assertEqual(len(pages), 5)
        self.assertEqual(len({p.name for p in pages}), 5)
        for page in pages:
            with self.subTest(page=page.name):
                page.validate()
        self.assertEqual([len(p.footprint) for p in pages], [4, 4, 0, 2, 2])

    def test_overlong_text_fails(self):
        p = figures.Canvas("test", "Test", "Subtitle")
        p.text("This text cannot fit", 100, 400, 5, 5, 30)
        with self.assertRaisesRegex(ValueError, "outside its box"):
            p.validate()

    def test_overlapping_text_fails(self):
        p = figures.Canvas("test", "Test", "Subtitle")
        p.text("First", 100, 400, 300, 50)
        p.text("Second", 100, 400, 300, 50)
        with self.assertRaisesRegex(ValueError, "Overlapping text"):
            p.validate()

    def test_native_image_contains_original_pixels(self):
        p = figures.Canvas("test", "Test", "Subtitle")
        raw = np.arange(40000, dtype=np.uint8).reshape(200, 200)
        p.image(raw, 100, 400, 300, "OCT")
        cell = next(c for c in p.page.root if "shape=image;" in c.get("style", ""))
        encoded = cell.get("style").split("image=data:image/png,", 1)[1].split(";", 1)[0]
        decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
        np.testing.assert_array_equal(decoded, raw)

    def test_text_over_image_fails(self):
        p = figures.Canvas("test", "Test", "Subtitle")
        p.image(np.zeros((200, 200), np.uint8), 100, 400, 300, "OCT")
        p.text("Obscured image", 120, 450, 500, 60)
        with self.assertRaisesRegex(ValueError, "obscures image"):
            p.validate()

    def test_pack_has_five_native_pages(self):
        root = ET.parse(figures.OUT / "motivation_options.drawio").getroot()
        self.assertEqual(len(root.findall("diagram")), 5)
        for diagram in root.findall("diagram"):
            self.assertIsNotNone(diagram.find("mxGraphModel/root"))

    def test_supporting_results_retain_adverse_transitions(self):
        source = json.loads((figures.OUT / "source_data.json").read_text())
        self.assertEqual(source["agent"]["cohort"]["introduced"], 18)
        self.assertEqual(source["agent"]["cohort"]["agent"]["fp"], 22)
        self.assertEqual(source["agent"]["manuscript_reported_metrics"]["baseline"]["worst_group_f1"], .6344)
        counts = source["foundations"]
        self.assertEqual(sum(counts[k] for k in ("both_correct", "both_wrong", "retfound_only", "visionfm_only")), counts["n"])


if __name__ == "__main__":
    unittest.main()
