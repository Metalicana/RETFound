import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("gap_audit", Path(__file__).resolve().parents[1] / "scripts/audit_manuscript_result_gaps.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ManuscriptAuditTests(unittest.TestCase):
    def test_f1_definitions_are_distinct(self):
        rows = [dict(truth=0, prediction=0)] * 9 + [dict(truth=1, prediction=0)]
        result = audit.metrics(rows)
        self.assertEqual(result["f1_positive"], 0)
        self.assertAlmostEqual(result["f1_macro"], 9 / 19)
        self.assertAlmostEqual(result["f1_weighted"], .9 * 18 / 19)
        self.assertEqual(result["balanced_accuracy"], .5)

    def test_invalid_and_missing_cases_not_filled(self):
        expected = {"a": {"truth": 1}, "b": {"truth": 0}}
        row = dict(id="a", truth=1, prediction=None)
        result, _ = audit.summarize([row], expected)
        self.assertEqual(result["missing_cases"], 1)
        self.assertEqual(result["invalid_prediction"], 1)
        self.assertFalse(result["complete_locked_cohort"])
        self.assertNotIn("f1_macro", result)

    def test_duplicates_and_label_conflicts_block_completion(self):
        row = dict(id="a", truth=1, prediction=1)
        result, _ = audit.summarize([row, row], {"a": {"truth": 0}})
        self.assertFalse(result["complete_locked_cohort"])
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(result["label_conflicts"], 2)
        self.assertNotIn("f1_macro", result)

    def test_progression_is_not_detection(self):
        self.assertIsNone(audit.dataset_task({"task": "progression_forecasting"}, Path("gdp.csv")))
        self.assertEqual(audit.dataset_task({"Task_Folder": "AMD"}, Path("predictions.csv")), ("fairvision", "amd"))

    def test_standalone_amd_format_without_task_column(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fairvision_amd_gpt51_predictions.csv"
            path.write_text("Filename,Model,Ground_Truth,Pred_AMD\na.npz,gpt-5.1,1,1\n")
            rows, _ = audit.audit_file(path, {("fairvision", "amd"): {"a": {"truth": 1}}})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task"], "amd")
            self.assertTrue(rows[0]["complete_locked_cohort"])

    def test_missing_task_not_inferred_from_ambiguous_columns(self):
        self.assertIsNone(audit.dataset_task({"Pred_AMD": "1", "Pred_GL": "0"}, Path("predictions.csv")))
        self.assertIsNone(audit.dataset_task({"task": "other", "Pred_AMD": "1"}, Path("predictions.csv")))
        self.assertIsNone(audit.dataset_task({"dataset": "external", "Pred_AMD": "1"}, Path("predictions.csv")))
        self.assertIsNone(audit.dataset_task({"Pred_AMD": "1"}, Path("gdp_progression.csv")))

    def test_root_level_evaluator_outputs_are_discovered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "audit_export"
            expected = [root / "gdp_test_agentic_predictions.csv",
                        root / "OphthalmicAgent/fairvision_amd_gpt51_predictions.csv",
                        root / "equi-agent/outputs/predictions/model.csv"]
            skipped = [root / "OphthalmicAgent/weights/ignored.csv",
                       root / "equi-agent/outputs/audits/ignored.csv"]
            for path in expected + skipped:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "linked.csv").symlink_to(expected[0])
            self.assertEqual(set(audit.result_csv_paths(root, out)), set(expected))
            self.assertNotIn(expected[1], set(audit.result_csv_paths(root, root / "OphthalmicAgent")))

    def test_root_gdp_output_matches_locked_detection_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gdp_test_agentic_predictions.csv"
            path.write_text("Patient_ID,Ground_Truth,Pred_GL\n301,1,1\n302,0,-1\n")
            expected = {("gdp", "glaucoma"): {"data_0301": {"truth": 1}, "data_0302": {"truth": 0}}}
            rows, _ = audit.audit_file(path, expected)
            self.assertEqual(rows[0]["missing_cases"], 0)
            self.assertEqual(rows[0]["invalid_prediction"], 1)
            self.assertFalse(rows[0]["complete_locked_cohort"])


if __name__ == "__main__":
    unittest.main()
