from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_figure_cdr_case import mask_measurements, overlay_mask


class FigureCdrExportTests(unittest.TestCase):
    def test_ratios_use_full_disc_including_cup(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[2:18, 3:27] = 1
        mask[6:14, 9:21] = 2
        result = mask_measurements(mask)
        self.assertEqual(result["vertical_cdr"], .5)
        self.assertEqual(result["horizontal_cdr"], .5)
        self.assertEqual(result["area_cdr"], .25)
        self.assertEqual(result["disc_area_pixels"], 384)
        self.assertEqual(result["cup_area_pixels"], 96)
        self.assertEqual(result["disc_bbox_xyxy"], [3, 2, 27, 18])
        self.assertEqual(result["cup_bbox_xyxy"], [9, 6, 21, 14])

    def test_empty_or_missing_cup_fails(self):
        for mask in (np.zeros((20, 30)), np.ones((20, 30))):
            with self.subTest(mask_value=mask[0, 0]):
                with self.assertRaisesRegex(ValueError, "no usable disc/cup"):
                    mask_measurements(mask)

    def test_invalid_classes_or_dimensions_fail(self):
        for mask in (np.full((20, 30), 3), np.zeros((20, 30, 3))):
            with self.assertRaisesRegex(ValueError, "2D mask"):
                mask_measurements(mask)

    def test_contours_preserve_image_elsewhere(self):
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        mask[20:40, 20:40] = 2
        original = Image.new("RGB", (60, 60), (71, 85, 33))
        result = np.asarray(overlay_mask(original, mask))
        np.testing.assert_array_equal(result[10, 30], [0, 195, 220])
        np.testing.assert_array_equal(result[20, 30], [235, 95, 186])
        np.testing.assert_array_equal(result[0, 0], [71, 85, 33])
        np.testing.assert_array_equal(result[30, 30], [71, 85, 33])
        np.testing.assert_array_equal(np.asarray(original)[10, 30], [71, 85, 33])

    def test_different_image_and_mask_dimensions_fail(self):
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            overlay_mask(Image.new("RGB", (30, 20)), np.zeros((30, 20)))


if __name__ == "__main__":
    unittest.main()
