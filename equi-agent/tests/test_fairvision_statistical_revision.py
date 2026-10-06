"""No API clients, inference, or training: statistical revision regression tests."""
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "equi-agent/scripts"))
import build_fairvision_statistical_revision as revision

OUT = revision.DEFAULT_OUT


def read(name):
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def case(i, truth, pred, race="white"):
    return dict(case_id=str(i), truth=truth, prediction=pred, race=race,
                sex_gender="female", age_group="older")


class StatisticalUnitTests(unittest.TestCase):
    def test_paired_alignment_is_by_id_not_row_order(self):
        a = [case(2, 0, 0), case(1, 1, 1), case(3, 0, 1)]
        b = [case(1, 1, 0), case(2, 0, 1)]
        cases, pa, pb = revision.align(a, b)
        self.assertEqual([c["case_id"] for c in cases], ["1", "2"])
        self.assertEqual(pa, [1, 0])
        self.assertEqual(pb, [0, 1])

    def test_paired_conflicts_and_duplicates_fail(self):
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            revision.align([case(0, 1, 0)], [case(0, 0, 0)])
        with self.assertRaises(ValueError):
            revision.align([case(0, 1, 0)] * 2, [case(0, 1, 0)])
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            revision.align([case(0, 1, 0)], [case(0, 1, 0, "black")])

    def test_worst_group_reselected_per_draw_against_sklearn(self):
        cases = [case(0, 1, 0, "black"), case(1, 1, 1, "black"),
                 case(2, 1, 1), case(3, 0, 0), case(4, 0, 1, "black"),
                 case(5, 0, 0, "unknown")]
        truth, statistic = revision.metric_statistic(cases)
        positive = np.array([[0, 1, 2], [1, 1, 2], [0, 0, 1]])
        negative = np.array([[3, 4, 5], [3, 4, 4], [3, 3, 5]])
        observed = statistic(positive, negative)
        for i, ids in enumerate(np.concatenate((positive, negative), axis=1)):
            predictions = [cases[j]["prediction"] for j in ids]
            self.assertAlmostEqual(observed[0, i], f1_score(
                truth[ids], predictions, labels=[0, 1], average="macro", zero_division=0))
            scores = []
            for attr in revision.U.ATTRIBUTES:
                for group in sorted({c[attr] for c in cases} - revision.U.UNKNOWN):
                    members = [cases[j] for j in ids if cases[j][attr] == group]
                    scores.append(f1_score([c["truth"] for c in members],
                        [c["prediction"] for c in members], labels=[0, 1], average="macro", zero_division=0))
            self.assertAlmostEqual(observed[-1, i], min(scores))

    def test_missing_bootstrap_group_is_not_dropped(self):
        cases = [case(0, 1, 1, "black"), case(1, 1, 1), case(2, 0, 0), case(3, 0, 0)]
        _, statistic = revision.metric_statistic(cases)
        self.assertTrue(np.isnan(statistic(np.array([1, 1]), np.array([2]))[-1]))
        result = revision.method_intervals(cases, resamples=100)
        worst = result[-1]
        self.assertGreater(worst["undefined_replicates"], 0)
        self.assertIsNone(worst["ci_lower"])
        self.assertIsNone(worst["ci_upper"])
        self.assertIn("NOT FOUND", worst["interval_status"])
        self.assertTrue(all(r["undefined_replicates"] == 0 for r in result[:-1]))

    def test_bootstrap_reproducible_and_single_class_group_fixed_label(self):
        cases = [case(0, 1, 1), case(1, 1, 0), case(2, 0, 0, "black"), case(3, 0, 1)]
        _, statistic = revision.metric_statistic(cases)
        self.assertEqual(statistic(np.array([0, 0]), np.array([2, 2]))[-1], .5)
        self.assertEqual(revision.method_intervals(cases, 100), revision.method_intervals(cases, 100))

    def test_invalid_forced_label_rejected(self):
        with self.assertRaises(ValueError):
            revision.method_intervals([case(0, 0, -1), case(1, 1, 1)], 100)

    def test_tex_bounds_do_not_become_line_spacing(self):
        result = revision.tex_interval(dict(estimate=.75, ci_lower=.6, ci_upper=.8))
        self.assertIn(r"\\{[0.600, 0.800]}", result)
        self.assertNotIn(r"\\[", result)


@unittest.skipUnless((OUT / "provenance.json").exists(), "Run saved-output analysis first")
class RevisionArtifactTests(unittest.TestCase):
    def test_all_37_rows_have_all_eight_intervals(self):
        rows = read("method_intervals.csv")
        self.assertEqual(len(rows), 37 * 8)
        expected = set(revision.U.METRICS) | {"worst_group_macro_f1"}
        for task, method in {(r["task"], r["method"]) for r in rows}:
            selected = [r for r in rows if (r["task"], r["method"]) == (task, method)]
            self.assertEqual({r["metric"] for r in selected}, expected)
            for row in selected:
                self.assertEqual(int(row["n_bootstrap"]), 10000)
                self.assertEqual(int(row["seed"]), revision.SEED)
                self.assertEqual(int(row["undefined_replicates"]), 0)
                self.assertLessEqual(float(row["ci_lower"]), float(row["ci_upper"]))

    def test_all_f1_points_against_sklearn_and_subgroup_minimum(self):
        aligned = read("fairvision_aligned_predictions.csv")
        summaries = read("fairvision_metrics.csv")
        intervals = read("method_intervals.csv")
        for summary in summaries:
            key = summary["task"], summary["method"]
            rows = [r for r in aligned if (r["task"], r["method"]) == key]
            truth, pred = ([int(r[k]) for r in rows] for k in ("truth", "prediction"))
            point = {r["metric"]: float(r["estimate"]) for r in intervals if (r["task"], r["method"]) == key}
            for avg in ("macro", "weighted"):
                self.assertAlmostEqual(point[f"f1_{avg}"], f1_score(truth, pred, labels=[0, 1], average=avg, zero_division=0))
            groups = [r for r in read("fairvision_subgroups.csv") if (r["task"], r["method"]) == key]
            self.assertAlmostEqual(point["worst_group_macro_f1"], min(float(r["f1_macro"]) for r in groups))

    def test_paired_points_match_method_intervals_on_complete_cohorts(self):
        methods = {(r["task"], r["method"], r["metric"]): r for r in read("method_intervals.csv")}
        for row in read("paired_intervals.csv"):
            if row["kind"] == "method" and row["n"] == "250":
                other = methods[row["task"], row["method"], row["metric"]]
                for key in ("estimate", "ci_lower", "ci_upper"):
                    self.assertAlmostEqual(float(row[key]), float(other[key]))

    def test_seven_comparison_family_and_exact_mcnemar(self):
        tests = read("paired_tests.csv")
        self.assertEqual([(r["task"], r["comparator"]) for r in tests], list(revision.COMPARISONS))
        pvalues = []
        for row in tests:
            c, r = int(row["corrections"]), int(row["regressions"])
            p = binomtest(c, c+r, .5).pvalue
            self.assertAlmostEqual(float(row["p_value"]), p)
            pvalues.append(p)
        # Independently compute Holm adjusted p values without the shared helper.
        ranked = sorted(enumerate(pvalues), key=lambda x: x[1])
        running = 0.
        for rank, (i, p) in enumerate(ranked):
            running = max(running, min(1., (7-rank)*p))
            self.assertAlmostEqual(float(tests[i]["p_holm_seven_comparisons"]), running)

    def test_exact_binomial_intervals_cover_boundaries(self):
        rows = read("binomial_intervals.csv")
        self.assertEqual(len(rows), 74)
        for row in rows:
            expected = binomtest(int(row["successes"]), int(row["denominator"])).proportion_ci(method="exact")
            self.assertAlmostEqual(float(row["ci_lower"]), expected.low)
            self.assertAlmostEqual(float(row["ci_upper"]), expected.high)
        perfect = next(r for r in rows if r["task"] == "dr" and r["method"] == "GPT-5.1" and r["metric"] == "specificity")
        self.assertLess(float(perfect["ci_lower"]), 1.)
        self.assertEqual(float(perfect["ci_upper"]), 1.)

    def test_bonferroni_intervals_wider_and_amd_stays_partial(self):
        for row in read("paired_intervals.csv"):
            if row["task"] == "amd":
                self.assertEqual(row["n"], "210")
                self.assertIn("NOT full cohort", row["scope"])
            if row["kind"] == "difference":
                self.assertAlmostEqual(float(row["bonferroni_marginal_confidence"]), 1-.05/7)
                self.assertLessEqual(float(row["ci_bonferroni_lower"]), float(row["ci_lower"]))
                self.assertGreaterEqual(float(row["ci_bonferroni_upper"]), float(row["ci_upper"]))
        missing = read("fairvision_missing.csv")
        self.assertEqual(sum(r["task"] == "amd" and r["method"] == "RetinAgent" for r in missing), 40)

    def test_provenance_source_and_output_hashes(self):
        provenance = json.loads((OUT / "provenance.json").read_text())
        self.assertEqual(provenance["api_calls"], 0)
        self.assertEqual(provenance["training_runs"], 0)
        for name, expected in provenance["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)
        for name, expected in provenance["output_sha256"].items():
            self.assertEqual(hashlib.sha256((OUT / name).read_bytes()).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
