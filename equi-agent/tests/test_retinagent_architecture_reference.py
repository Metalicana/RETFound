"""Structural checks for the native architecture figure; no inference."""
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("architecture_reference", SCRIPTS / "build_retinagent_architecture_reference.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.page, self.assets = module.build()
        self.cells = {cell.get("id"): cell for cell in self.page.root}

    def text_bounds(self, cell):
        geometry = cell.find("mxGeometry")
        x, y = float(geometry.get("x", 0)), float(geometry.get("y", 0))
        parent = self.cells.get(cell.get("parent"))
        while parent is not None and parent.get("id") not in {"0", "1"}:
            g = parent.find("mxGeometry")
            x += float(g.get("x", 0))
            y += float(g.get("y", 0))
            parent = self.cells.get(parent.get("parent"))
        return x, y, float(geometry.get("width")), float(geometry.get("height"))

    def test_native_groups_and_attached_connectors(self):
        counts = module.validate(self.page)
        self.assertEqual(counts["embedded_images"], 3)
        self.assertEqual(counts["attached_connectors"], 18)
        self.assertGreaterEqual(counts["groups"], 30)
        self.assertTrue(all(path.exists() for path in self.assets))

    def test_no_text_box_overlap_or_page_overflow(self):
        texts = [c for c in self.cells.values() if c.get("value") and "text;" in c.get("style", "")]
        for i, cell in enumerate(texts):
            x, y, w, h = self.text_bounds(cell)
            self.assertTrue(0 <= x and 0 <= y and x + w <= 2400 and y + h <= 1460, cell.get("value"))
            for other in texts[i + 1:]:
                a, b, c, d = self.text_bounds(other)
                overlap = min(x + w, a + c) - max(x, a) > 1 and min(y + h, b + d) - max(y, b) > 1
                self.assertFalse(overlap, (cell.get("value"), other.get("value")))

    def test_scope_is_visible(self):
        text = "\n".join(c.get("value", "") for c in self.cells.values())
        for required in ("No online model updates", "Updated protocol only", "GDP progression only",
                         "not active in task-specific FairVision runs", "not a historical execution receipt"):
            self.assertIn(required, text)

    def test_images_are_embedded_without_external_dependencies(self):
        images, _ = module.embedded_images()
        styles = [c.get("style", "") for c in self.cells.values() if "image=data:" in c.get("style", "")]
        for image in images.values():
            self.assertTrue(any("image=" + image + ";" in style for style in styles))
        self.assertFalse(any("image=http" in c.get("style", "") for c in self.cells.values()))


if __name__ == "__main__":
    unittest.main()
