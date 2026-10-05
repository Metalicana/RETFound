from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_fairvision_escalation_macro_f1 as plot


def case(truth, prediction, score):
    return dict(truth=truth, prediction=prediction, risk_score=score)


def fixture(root):
    summary = dict(tasks=list(plot.TASKS), test_common_cases=6,
        selective_escalation=dict(risk_coverage_rank_column="risk_score", low_reliability_threshold=.35))
    (root / "selective_arbitration_summary.json").write_text(json.dumps(summary))
    rows = [dict(task=task, split="test", dataset="harvard_fairvision",
        model_name="reliability_shrunk_weighted", image_id=f"case{i}", y_true=i, y_pred=i,
        risk_score=.2, y_prob=.9 if i else .1, disagreement_rate=0., weighted_reliability=.8)
        for task in plot.TASKS for i in (0, 1)]
    plot.write_csv(root / "selective_arbitration_predictions.csv", rows)
    return rows


class EscalationCurveTests(unittest.TestCase):
    def test_macro_not_positive_f1(self):
        cases = [case(0, 0, .2)] * 48 + [case(0, 1, .4)] * 18
        cases += [case(1, 0, .6)] * 8 + [case(1, 1, .8)] * 7
        full = plot.sweep(cases, "glaucoma")[-1]
        expected = f1_score([c["truth"] for c in cases], [c["prediction"] for c in cases],
                            labels=[0, 1], average="macro", zero_division=0)
        self.assertAlmostEqual(full["positive_f1"], .35)
        self.assertAlmostEqual(full["macro_f1"], expected)
        self.assertNotAlmostEqual(full["positive_f1"], full["macro_f1"])

    def test_all_thresholds_match_direct_filtering(self):
        cases = [case(0, 0, .2), case(0, 1, .6), case(1, 0, .6), case(1, 1, .9)]
        for row in plot.sweep(cases, "dr"):
            accepted = [c for c in cases if c["risk_score"] <= row["threshold"]]
            self.assertEqual(row["accepted"], len(accepted))
            self.assertEqual(row["escalated"], len(cases) - len(accepted))
            if accepted:
                expected = f1_score([c["truth"] for c in accepted], [c["prediction"] for c in accepted],
                                    labels=[0, 1], average="macro", zero_division=0)
                self.assertAlmostEqual(row["macro_f1"], expected)
            else:
                self.assertIsNone(row["macro_f1"])

    def test_ties_cannot_be_split(self):
        cases = [case(0, 0, .2), case(1, 1, .4), case(0, 1, .4), case(1, 0, .8)]
        counts = [row["accepted"] for row in plot.sweep(cases, "amd")]
        self.assertEqual(counts, [0, 1, 3, 4, 4])

    def test_empty_and_single_class_subsets_are_flagged(self):
        rows = plot.sweep([case(0, 0, .1), case(1, 1, .9)], "dr")
        self.assertEqual(rows[0]["display_support"], "empty")
        self.assertIsNone(rows[0]["macro_f1"])
        self.assertEqual(rows[1]["display_support"], "one_class")
        self.assertEqual(rows[1]["macro_f1"], .5)
        self.assertFalse(rows[1]["both_classes"])
        self.assertEqual(rows[2]["display_support"], "sparse")
        self.assertEqual(rows[2]["macro_f1"], 1.)
        with self.assertRaises(ValueError):
            plot.sweep([], "dr")

    def test_minimum_support_marker_does_not_filter_cases(self):
        row = plot.point("dr", .5, [100, 0, 1, 18], 200)
        self.assertEqual(row["accepted"], 119)
        self.assertEqual(row["display_support"], "sparse")
        row = plot.point("dr", .5, [100, 0, 1, 19], 200)
        self.assertEqual(row["display_support"], "supported")

    def test_original_predictions_are_unchanged(self):
        cases = [case(1, 0, .1), case(0, 1, .9)]
        before = [dict(row) for row in cases]
        plot.sweep(cases, "glaucoma")
        self.assertEqual(cases, before)

    def test_bad_sources_fail_without_silently_dropping_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = fixture(root)
            cases, _ = plot.load_cases(root, expected_per_task=2)
            self.assertEqual(set(cases), set(plot.TASKS))
            for key, invalid in (("y_true", "unknown"), ("risk_score", "nan"),
                    ("risk_score", .4), ("split", "val"), ("model_name", "live_agent")):
                with self.subTest(field=key, invalid=invalid):
                    changed = [dict(row) for row in rows]
                    changed[0][key] = invalid
                    plot.write_csv(root / "selective_arbitration_predictions.csv", changed)
                    with self.assertRaises(ValueError):
                        plot.load_cases(root, expected_per_task=2)
            for changed in (rows[:-1], rows + [rows[0]]):
                plot.write_csv(root / "selective_arbitration_predictions.csv", changed)
                with self.assertRaises(ValueError):
                    plot.load_cases(root, expected_per_task=2)

    @unittest.skipUnless((plot.SOURCE / "selective_arbitration_predictions.csv").exists(),
                         "Historical result download unavailable")
    def test_saved_cohorts_reproduce_full_coverage_metrics(self):
        cases, _ = plot.load_cases(plot.SOURCE)
        for task in plot.TASKS:
            result = plot.sweep(cases[task], task)[-1]
            expected = f1_score([r["truth"] for r in cases[task]],
                [r["prediction"] for r in cases[task]], labels=[0, 1], average="macro", zero_division=0)
            self.assertEqual(result["accepted"], 3000)
            self.assertEqual(result["escalation_percent"], 0)
            self.assertAlmostEqual(result["macro_f1"], expected)


if __name__ == "__main__":
    unittest.main()
