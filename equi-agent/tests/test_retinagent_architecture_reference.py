"""Structural checks for the native architecture figure; no inference."""
import importlib.util
import hashlib
import json
import sys
import tempfile
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
        return module.bounds(self.page, cell)

    def test_native_groups_and_attached_connectors(self):
        counts = module.validate(self.page)
        self.assertEqual(counts["embedded_images"], 3)
        self.assertEqual(counts["attached_connectors"], 5)
        self.assertGreaterEqual(counts["groups"], 15)
        self.assertLessEqual(counts["label_words"], 40)
        self.assertTrue(all(path.exists() for path in self.assets))

    def test_no_text_box_overlap_or_page_overflow(self):
        texts = [c for c in self.cells.values() if c.get("value") and "text;" in c.get("style", "")]
        for i, cell in enumerate(texts):
            x, y, w, h = self.text_bounds(cell)
            self.assertTrue(0 <= x and 0 <= y and x + w <= module.WIDTH and y + h <= module.HEIGHT,
                            cell.get("value"))
            for other in texts[i + 1:]:
                a, b, c, d = self.text_bounds(other)
                overlap = min(x + w, a + c) - max(x, a) > 1 and min(y + h, b + d) - max(y, b) > 1
                self.assertFalse(overlap, (cell.get("value"), other.get("value")))

    def test_diagnosis_is_not_mixed_with_optional_or_progression_modules(self):
        text = "\n".join(c.get("value", "") for c in self.cells.values())
        for required in ("Diagnosis", "Bio-Profiler", "Vision", "RETFound", "CDR tool", "Reliability",
                         "Counterfactual", "Orchestrator"):
            self.assertIn(required, text)
        for absent in ("GDP", "Optional", "Safety", "Equity", "escalation", "online", "historical"):
            self.assertNotIn(absent, text)

    def test_original_evidence_and_audit_both_reach_orchestrator(self):
        orchestrator = next(c for c in self.cells.values()
                            if c.get("data-name") == "Ophthalmologist Orchestrator")
        port = next(c for c in self.cells.values()
                    if c.get("parent") == orchestrator.get("id") and c.get("data-role") == "agent-port")
        incoming = [c for c in self.cells.values() if c.get("target") == port.get("id")]
        self.assertEqual(len(incoming), 2)
        sources = [self.cells[c.get("source")] for c in incoming]
        self.assertTrue(any(c.get("data-name") == "Case evidence" for c in sources))
        self.assertTrue(any(self.cells[c.get("parent")].get("data-name") == "Counterfactual Agent"
                            for c in sources))

    def test_progression_has_its_own_concise_page(self):
        page, _ = module.build(progression=True)
        counts = module.validate(page)
        self.assertEqual(counts["attached_connectors"], 5)
        self.assertLessEqual(counts["label_words"], 40)
        self.assertNotEqual(page.element.get("id"), self.page.element.get("id"))
        text = "\n".join(c.get("value", "") for c in page.root)
        for value in ("GDP progression", "Baseline imaging", "Baseline visual field", "Functional",
                      "Structural", "Helper models", "Six endpoint"):
            self.assertIn(value, text)
        self.assertNotIn("CDR", text)

    def test_archive_preserves_previous_master_and_refuses_manual_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            master = out / "retinagent_architecture.drawio"
            master.write_text("previous figure")
            (out / "provenance.json").write_text(json.dumps(dict(
                figure_sha256=hashlib.sha256(master.read_bytes()).hexdigest())))
            module.archive_existing(out)
            saved = out / "previous_detailed/retinagent_architecture.drawio"
            self.assertEqual(saved.read_text(), master.read_text())
            master.write_text("manually edited figure")
            with self.assertRaisesRegex(ValueError, "manual edits"):
                module.archive_existing(out)
            self.assertEqual(saved.read_text(), "previous figure")

    def test_images_are_embedded_without_external_dependencies(self):
        images, _ = module.embedded_images()
        styles = [c.get("style", "") for c in self.cells.values() if "image=data:" in c.get("style", "")]
        for image in images.values():
            self.assertTrue(any("image=" + image + ";" in style for style in styles))
        self.assertFalse(any("image=http" in c.get("style", "") for c in self.cells.values()))


if __name__ == "__main__":
    unittest.main()
