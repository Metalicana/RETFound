from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import audit_fairvision_glaucoma_uncertainty as audit


def cases():
    return [dict(case_id=f"case{i}", truth=i % 2, age_group="older", race="white", sex_gender="female")
            for i in range(20)]


def paper_fixture(root):
    manifest, agent, baseline = [], [], []
    for c in cases():
        filename = f"data/Glaucoma/Test/{c['case_id']}.npz"
        common = dict(Ground_Truth=c["truth"], Age=75, Gender="female", Race="white")
        manifest.append(dict(filename=filename, Task_Folder="Glaucoma", **common))
        agent.append(dict(Filename=filename, Task_Folder="Glaucoma", Pred_GL=c["truth"], **common))
        baseline.append(dict(Filename=filename, Disease="Glaucoma", Model="RETFound", Modality="OCT",
                             Prediction=0, **common))
    for name, rows in (("manifest_recovered.csv", manifest), ("agent_predictions_recovered.csv", agent),
                       ("retfound_predictions_recovered.csv", baseline)):
        audit.runner.audit.write_csv(root / name, rows)
    audit.runner.write_json(root / "recovery_report.json", dict(case_id="case0", npz_verified=False,
        limitations=["Worst-group F1 and other model rows are not recalculated here."]))


def ablation_fixture(root):
    cohort = [dict(task="glaucoma", **c) for c in cases()]
    offline = [dict(case_id=c["case_id"], task="glaucoma", variant=v, prediction=c["truth"])
               for v in audit.runner.VARIANTS[:2] for c in cohort]
    config = dict(prepared_sha256=audit.runner.digest(cohort), offline_sha256=audit.runner.digest(offline))
    config["fingerprint"] = audit.runner.digest(config)
    for name, data in (("config.json", config), ("prepared_cases.json", cohort),
                       ("offline_predictions.json", offline), ("live_receipt.json", dict(run=config["fingerprint"]))):
        audit.runner.write_json(root / name, data)
    for variant in audit.runner.VARIANTS[2:]:
        for c in cohort:
            audit.runner.write_json(root / "agent/glaucoma" / variant / f"{c['case_id']}.json",
                dict(case_id=c["case_id"], task="glaucoma", variant=variant, fingerprint=config["fingerprint"],
                     shared_evidence_sha256=f"shared-{c['case_id']}", prediction=c["truth"]))


class UncertaintyTests(unittest.TestCase):
    def test_metrics_match_sklearn_including_vectorized_batches(self):
        truth = np.array([0, 0, 0, 1, 1, 1])
        predictions = np.array([[0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]])
        actual = audit.metric_array(truth, predictions)
        expected = []
        for prediction in predictions:
            expected.append([f1_score(truth, prediction, average="macro", labels=[0, 1], zero_division=0),
                f1_score(truth, prediction, average="weighted", labels=[0, 1], zero_division=0),
                f1_score(truth, prediction, zero_division=0), recall_score(truth, prediction),
                recall_score(truth, prediction, pos_label=0), balanced_accuracy_score(truth, prediction),
                accuracy_score(truth, prediction)])
        np.testing.assert_allclose(actual, expected)
        batches = np.array([[0, 1, 2, 3, 4, 5], [2, 0, 1, 5, 3, 4]])
        vectorized = audit.metric_array(truth[batches], predictions[:, batches])
        np.testing.assert_allclose(vectorized, np.repeat(actual[:, :, None], 2, axis=2))

    def test_identical_predictions_have_zero_paired_intervals(self):
        values = [0] * len(cases())
        intervals, tests, differences, _, _ = audit.analyse(cases(), {"a": values, "b": values}, "b", resamples=200)
        for row in intervals:
            if row["kind"] == "difference":
                self.assertEqual((row["estimate"], row["ci_lower"], row["ci_upper"]), (0., 0., 0.))
        self.assertEqual(tests[0]["p_value"], 1.)
        self.assertEqual(differences, [])

    def test_error_test_uses_paired_discordances_and_is_reproducible(self):
        truth = [c["truth"] for c in cases()]
        baseline, primary = truth.copy(), truth.copy()
        for i in range(6):
            baseline[i] = 1 - baseline[i]
        primary[6] = 1 - primary[6]
        predictions = dict(baseline=baseline, primary=primary)
        first = audit.analyse(cases(), predictions, "primary", resamples=200)
        second = audit.analyse(cases(), predictions, "primary", resamples=200)
        self.assertEqual(first, second)
        self.assertEqual(first[1][0]["corrections"], 6)
        self.assertEqual(first[1][0]["regressions"], 1)
        self.assertAlmostEqual(first[1][0]["p_value"], binomtest(6, 7, .5).pvalue)
        self.assertEqual(len(first[2]), 7)

    def test_bootstrap_resamples_fixed_class_counts_and_entire_prediction_vectors(self):
        actual = audit.metric_array
        baseline = np.array([c["truth"] for c in cases()])

        def checked(truth, predictions):
            self.assertTrue(np.all(np.sum(truth == 1, axis=-1) == 10))
            self.assertTrue(np.all(np.sum(truth == 0, axis=-1) == 10))
            np.testing.assert_array_equal(predictions[0], truth)
            np.testing.assert_array_equal(predictions[1], 1 - truth)
            return actual(truth, predictions)

        with patch.object(audit, "metric_array", side_effect=checked) as spy:
            audit.analyse(cases(), dict(a=baseline, b=1-baseline), "a", resamples=200)
        self.assertGreater(spy.call_count, 1)

    def test_invalid_case_inputs_rejected(self):
        truth = [c["truth"] for c in cases()]
        for cohort, predictions in ((cases() + [cases()[0]], dict(a=truth, b=truth)),
                (cases(), dict(a=truth[:-1], b=truth)), (cases(), dict(a=truth, b=[-1] * 20)),
                ([dict(c, truth=0) for c in cases()], dict(a=truth, b=truth))):
            with self.assertRaises(ValueError):
                audit.analyse(cohort, predictions, "b", resamples=100)

    def test_holm(self):
        np.testing.assert_allclose(audit.holm([.04, .001, .03]), [.06, .003, .06])

    def test_paper_loader_rejects_mismatches_without_intersection(self):
        for fault in ("missing", "duplicate", "truth", "metadata", "prediction", "task"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paper_fixture(root)
                path = root / "agent_predictions_recovered.csv"
                rows = audit.runner.read_csv(path)
                if fault == "missing":
                    rows.pop()
                elif fault == "duplicate":
                    rows.append(rows[0])
                else:
                    field, value = {"truth": ("Ground_Truth", 1), "metadata": ("Race", "asian"),
                                    "prediction": ("Pred_GL", -1), "task": ("Task_Folder", "AMD")}[fault]
                    rows[0][field] = value
                audit.runner.audit.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    audit.load_paper(root, 20)

    def test_ablation_loader_requires_complete_matched_provenance(self):
        for fault in (None, "missing", "extra", "fingerprint", "evidence", "receipt", "repair", "prepared"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ablation_fixture(root)
                path = root / "agent/glaucoma/retinagent_full/case0.json"
                if fault == "missing":
                    path.unlink()
                elif fault == "extra":
                    audit.runner.write_json(path.with_name("extra.json"), audit.read_json(path))
                elif fault in ("fingerprint", "evidence"):
                    row = audit.read_json(path)
                    row["fingerprint" if fault == "fingerprint" else "shared_evidence_sha256"] = "changed"
                    audit.runner.write_json(path, row)
                elif fault == "receipt":
                    audit.runner.write_json(root / "live_receipt.json", dict(run="wrong"))
                elif fault == "repair":
                    audit.runner.write_json(root / audit.runner.CDR_REPAIR / "receipt.json", dict(status="partial"))
                elif fault == "prepared":
                    audit.runner.write_json(root / "prepared_cases.json", [])
                if fault:
                    with self.assertRaises(ValueError):
                        audit.load_ablation(root, 20)
                else:
                    cohort, predictions, primary, _, _ = audit.load_ablation(root, 20)
                    self.assertEqual(len(cohort), 20)
                    self.assertEqual(len(predictions), 4)
                    self.assertEqual(primary, "RetinAgent (paired full control)")

    def test_cli_preserves_sources_and_subgroup_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_fixture(root)
            before = {p: p.read_bytes() for p in root.iterdir()}
            argv = ["audit", "--paper-root", str(root), "--out-dir", str(root / "audit"),
                    "--expected-cases", "20", "--resamples", "200"]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                audit.main()
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            provenance = json.loads((root / "audit/provenance.json").read_text())
            self.assertEqual(provenance["api_calls"], 0)
            self.assertEqual(len(provenance["sources"]), 4)
            self.assertEqual(len(audit.runner.read_csv(root / "audit/worst_groups.csv")), 2)
            self.assertTrue(audit.runner.read_csv(root / "audit/subgroups.csv"))
            report = (root / "audit/report.md").read_text()
            self.assertIn("Recovery-stage note", report)
            self.assertIn("this audit recalculates subgroup metrics", report)
            self.assertIn("not macro-F1", report)


if __name__ == "__main__":
    unittest.main()
