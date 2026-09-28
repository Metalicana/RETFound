import copy
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_clinical_decisions_figure import add_panel_borders, caption, compact_source, evidence_example, update_diagram, validate_counts


def source():
    return dict(cohort=dict(
        n=249, raw_agent_rows=250, excluded=[dict(case_id="data_07199.npz")],
        baseline=dict(n=249, tn=108, fp=17, fn=45, tp=79, worst_group_f1=.6394),
        agent=dict(n=249, tn=103, fp=22, fn=35, tp=89),
        corrected=23, introduced=18, both_correct=169, both_wrong=39,
        corrected_positive=17, corrected_negative=6,
        introduced_positive=7, introduced_negative=11,
    ), examples=[dict(case_id="data_07062.npz", truth=0, baseline=1, agent=0,
                      trace_fingerprint="verified-trace",
                      scenarios=dict(full_evidence=0, without_visual_interpretation=-1))],
        sources={}, images_verified=True)


def diagram():
    tree = ET.ElementTree(ET.fromstring('<mxfile><diagram><mxGraphModel><root>'
        '<mxCell id="0"/><mxCell id="1" parent="0"/>'
        '<mxCell id="author" value="Edited heading" vertex="1"><mxGeometry y="10"/></mxCell>'
        '<mxCell id="metric" value="Worst-group F1 (%)" vertex="1"><mxGeometry y="660"/></mxCell>'
        '</root></mxGraphModel></diagram></mxfile>'))
    root = tree.find("diagram/mxGraphModel/root")
    for i in range(4):
        cell = ET.SubElement(root, "mxCell", id=f"image{i}", vertex="1", style="shape=image;image=original;")
        ET.SubElement(cell, "mxGeometry", x=str(i * 450), y="80", width="430", height="430")
    return tree


class ClinicalDecisionTests(unittest.TestCase):
    def test_borders_preserve_author_content_including_footer(self):
        before = diagram()
        after = add_panel_borders(before)
        cells = {c.get("id"): c for c in after.iter("mxCell")}
        for cell in before.iter("mxCell"):
            self.assertEqual(ET.tostring(cell), ET.tostring(cells[cell.get("id")]))
        frames = [c for c in cells.values() if c.get("id").startswith("panel_")]
        self.assertEqual(len(frames), 3)
        self.assertTrue(all("fillColor=none" in c.get("style") for c in frames))

    def test_borders_do_not_restore_author_deletions_or_duplicate(self):
        tree = diagram()
        root = tree.find("diagram/mxGraphModel/root")
        root.remove(root.find("mxCell[@id='metric']"))
        once = add_panel_borders(tree)
        twice = add_panel_borders(once)
        self.assertEqual(ET.tostring(once.getroot()), ET.tostring(twice.getroot()))
        self.assertIsNone(twice.find("diagram/mxGraphModel/root/mxCell[@id='metric']"))

    def test_keeps_reported_metric_separate_from_counts(self):
        result = compact_source(source())
        self.assertEqual(result["manuscript_reported_metrics"]["baseline"]["worst_group_f1"], .6344)
        self.assertNotIn("worst_group_f1", result["cohort"]["baseline"])
        self.assertEqual(result["cohort"]["baseline"]["fn"], 45)
        self.assertEqual(result["cohort"]["agent"]["fp"], 22)

    def test_rejects_mixed_cohorts(self):
        bad = copy.deepcopy(source()["cohort"])
        bad["agent"]["tn"] += 1
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_counts(bad)

    def test_rejects_incorrect_transition_counts(self):
        bad = source()["cohort"]
        bad["corrected_positive"] = 18
        with self.assertRaisesRegex(ValueError, "transitions"):
            validate_counts(bad)

    def test_preserves_edits_and_original_images(self):
        before = diagram()
        after = update_diagram(before, source())
        cells = {c.get("id"): c for c in after.iter("mxCell")}
        for c in before.iter("mxCell"):
            if c.get("id") != "metric":
                self.assertEqual(ET.tostring(c), ET.tostring(cells[c.get("id")]))
        self.assertNotIn("metric", cells)
        labels = [c.get("value") for c in after.iter("mxCell")]
        self.assertIn("Inconclusive", labels)
        self.assertIn("Non-glaucoma", labels)
        self.assertNotIn("10 fewer", labels)
        self.assertNotIn("5 more", labels)
        self.assertNotIn("Worst-group F1 (%)", labels)

    def test_rebuild_does_not_duplicate_strip(self):
        once = update_diagram(diagram(), source())
        twice = update_diagram(once, source())
        self.assertEqual(ET.tostring(once.getroot()), ET.tostring(twice.getroot()))

    def test_missing_images_fail_instead_of_substituting(self):
        tree = diagram()
        root = tree.find("diagram/mxGraphModel/root")
        root.remove(root.find("mxCell[@id='image0']"))
        with self.assertRaisesRegex(ValueError, "four original"):
            update_diagram(tree, source())

    def test_ablation_must_match_saved_scenarios_and_final_label(self):
        original = source()
        for key, value in (("truth", 1), ("agent", 1), ("trace_fingerprint", "")):
            bad = copy.deepcopy(original)
            bad["examples"][0][key] = value
            with self.assertRaises(ValueError):
                evidence_example(bad)
        bad = copy.deepcopy(original)
        bad["examples"][0]["scenarios"]["without_visual_interpretation"] = 0
        with self.assertRaisesRegex(ValueError, "ablation"):
            evidence_example(bad)

    def test_negative_outcomes_remain_in_results_not_motivation_panel(self):
        metadata = compact_source(source())
        self.assertEqual(metadata["cohort"]["agent"]["fp"], 22)
        self.assertEqual(metadata["cohort"]["baseline"]["fp"], 17)
        text = caption(metadata)
        self.assertIn("5 more false alarms", text)
        self.assertIn("not independently", text)


if __name__ == "__main__":
    unittest.main()
