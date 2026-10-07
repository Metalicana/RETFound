"""Offline checks for image-review integrity; no model dependencies."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_glaucoma_failure_images.py"
spec = importlib.util.spec_from_file_location("image_review", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ImageReviewTests(unittest.TestCase):
    def test_rejects_unsafe_wrong_task_and_link_members(self):
        for name in ("/data_07001.npz", "../data_07001.npz",
                     "Datasets/FairVision/AMD/Test/data_07001.npz"):
            m = tarfile.TarInfo(name)
            m.size = 100
            with self.assertRaises(ValueError):
                review.archive_case(m)
        m = tarfile.TarInfo("Datasets/FairVision/Glaucoma/Test/data_07001.npz")
        m.size = 100
        self.assertEqual(review.archive_case(m), "data_07001")
        m.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            review.archive_case(m)

    def test_actual_indices_for_200_slice_volume(self):
        middle, sampled, context = review.indices(200)
        self.assertEqual(middle, 100)
        self.assertEqual(sampled.tolist(), [0, 28, 56, 85, 113, 142, 170, 199])
        self.assertEqual(context.tolist(), [28, 56, 113, 142])

    def test_rgb_conversion_parity_is_dtype_dependent(self):
        uint8 = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
        self.assertTrue(np.array_equal(uint8, review.uint8_slice(uint8)))
        floating = np.array([[0, .25, .5, .75, 1]], dtype=np.float32)
        direct = np.asarray(Image.fromarray(floating).convert("RGB"))[..., 0]
        scaled = review.uint8_slice(floating)
        self.assertEqual(direct.tolist(), [[0, 0, 0, 0, 1]])
        self.assertEqual(scaled.tolist(), [[0, 63, 127, 191, 255]])

    def test_slo_scaling_does_not_modify_source(self):
        array = np.array([[10, 20], [20, 30]], dtype=np.uint8)
        original = array.copy()
        scaled = review.slo_loader(array)
        self.assertEqual(scaled.tolist(), [[0, 127], [127, 255]])
        self.assertTrue(np.array_equal(array, original))
        self.assertTrue(np.array_equal(review.slo_loader(np.zeros((2, 2))), np.zeros((2, 2))))

    def test_payload_inventory_and_label_mismatch_retention(self):
        payload = io.BytesIO()
        np.savez(payload, oct_bscans=np.arange(8 * 16 * 16, dtype=np.uint8).reshape(8, 16, 16),
                 slo_fundus=np.arange(16 * 16, dtype=np.uint8).reshape(16, 16), glaucoma=np.array(1))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "cases").mkdir()
            row, overview = review.inspect_case("data_test", payload.getvalue(), {"truth": "0"}, out)
            self.assertFalse(row["reference_matches"])
            self.assertEqual(row["npz_reference"], 1)
            self.assertEqual(row["saved_reference"], 0)
            self.assertTrue(row["pillow_rgb_train_inference_equal"])
            self.assertEqual(len(list((out / "cases").glob("*.png"))), 3)
            self.assertGreater(np.asarray(overview).std(), 0)

    def test_pickle_payload_is_not_loaded(self):
        payload = io.BytesIO()
        np.savez(payload, oct_bscans=np.array([{"bad": True}], dtype=object),
                 slo_fundus=np.zeros((2, 2), dtype=np.uint8), glaucoma=np.array(0))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                review.inspect_case("data_test", payload.getvalue(), {"truth": "0"}, Path(tmp))


if __name__ == "__main__":
    unittest.main()
