"""Offline audit tests: no patient inference and no API client imports."""
import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "equi-agent/scripts"))
SPEC = importlib.util.spec_from_file_location(
    "manuscript_audit", ROOT / "equi-agent/scripts/audit_manuscript_twenty_questions.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
OUT = ROOT / "equi-agent/outputs/audits/manuscript_20_questions_20261005"


def rows(name):
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


class AuditUnitTests(unittest.TestCase):
    def test_invalid_predictions_are_not_negative(self):
        for value in ("-1", "", "nan", None, "Not Available", "2"):
            self.assertIsNone(AUDIT.binary(value))
        self.assertEqual(AUDIT.binary("0.0"), 0)
        self.assertEqual(AUDIT.binary("1"), 1)

    def test_duplicate_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            AUDIT.index([{"Filename": "data_001.npz"}, {"case_id": "data_001"}])

    def test_fixed_label_macro_includes_absent_class(self):
        self.assertEqual(AUDIT.stat([dict(truth=0, prediction=0)])['f1_macro'], .5)
        result = AUDIT.stat([dict(truth=0, prediction=0)] * 9 + [dict(truth=1, prediction=0)])
        self.assertNotEqual(result['f1_macro'], result['f1_weighted'])

    def test_group_contract_excludes_ethnicity_and_unknowns(self):
        data = [dict(truth=0, prediction=0, age_group="younger", race="black",
                     sex_gender="female", ethnicity="hispanic"),
                dict(truth=1, prediction=1, age_group="older", race="unknown",
                     sex_gender="male", ethnicity="non-hispanic")]
        groups = AUDIT.subgroup_rows(data, "glaucoma", "test")
        self.assertEqual({g['attribute'] for g in groups}, {'race', 'age_group', 'sex_gender'})
        self.assertNotIn('unknown', {g['subgroup'] for g in groups})
        self.assertEqual(len(groups), 5)


@unittest.skipUnless((OUT / "provenance.json").exists(), "Saved audit not available")
class SavedArtifactTests(unittest.TestCase):
    def test_all_fairvision_metrics_against_sklearn(self):
        aligned = rows("fairvision_aligned_predictions.csv")
        for metric in rows("fairvision_metrics.csv"):
            selected = [r for r in aligned if (r['task'], r['method']) ==
                        (metric['task'], metric['method'])]
            truth = [int(r['truth']) for r in selected]
            pred = [int(r['prediction']) for r in selected]
            for average, field in [('macro', 'f1_macro'), ('weighted', 'f1_weighted'), ('binary', 'f1_positive')]:
                with self.subTest(task=metric['task'], model=metric['method'], average=average):
                    self.assertAlmostEqual(float(metric[field]), f1_score(
                        truth, pred, labels=[0, 1], average=average, zero_division=0))
            self.assertEqual(len(selected), int(metric['n']))

    def test_missing_cases_remain_explicit(self):
        metrics = {(r['task'], r['method']): r for r in rows("fairvision_metrics.csv")}
        amd = metrics['amd', 'RetinAgent']
        self.assertEqual([int(amd[k]) for k in ('n', 'tp', 'tn', 'fp', 'fn')], [210, 101, 85, 11, 13])
        missing = [r for r in rows('fairvision_missing.csv') if r['task'] == 'amd' and r['method'] == 'RetinAgent']
        self.assertEqual(len(missing), 40)
        self.assertEqual(sum(int(r['truth']) for r in missing), 11)
        self.assertEqual(int(metrics['glaucoma', 'MIRAGE']['n']), 249)
        external = {r['task']: r for r in rows('external_complete_response_metrics.csv')}
        self.assertEqual(int(external['drishti']['n']), 47)
        self.assertEqual(int(external['refuge2']['n']), 207)
        self.assertTrue(all(r['full_cohort_result'] == 'NOT FOUND' for r in external.values()))

    def test_exact_mcnemar_matches_binomial(self):
        for row in rows('paired_tests.csv'):
            c, r = int(row['corrections']), int(row['regressions'])
            self.assertAlmostEqual(float(row['p_value']), binomtest(c, c+r, .5).pvalue)

    def test_counterfactual_counts_and_directions(self):
        data = rows('counterfactual_directions.csv')
        pooled = next(r for r in data if r['task'] == 'pooled' and r['attribute'] == 'ALL')
        self.assertEqual([int(pooled[k]) for k in ('n_perturbations', 'label_changes', 'increased', 'decreased')],
                         [54000, 202, 2637, 501])
        for r in data:
            self.assertAlmostEqual(float(r['increased_percent']), 100*int(r['increased'])/int(r['n_perturbations']))

    def test_weight_draws_are_seeded_dirichlet(self):
        draws = [r for r in rows('weight_draws.csv') if r['scenario'].startswith('dirichlet_')]
        observed = [[float(r[k]) for k in AUDIT.sensitivity.CANONICAL] for r in draws]
        expected = np.random.default_rng(20261005).dirichlet(np.ones(5), size=200)
        np.testing.assert_allclose(observed, expected)
        for r in rows('weight_patient_ranking.csv'):
            self.assertAlmostEqual(float(r['unchanged_percent']), 100*int(r['unchanged'])/int(r['n']))

    def test_prompt_export_covers_roles_and_baselines(self):
        text = (OUT / 'supplementary_prompts.txt').read_text()
        for path in ('Progression/prompts.py', 'EquityAgent/equity_agent.py',
                     'Orchestrator/fairvision_glaucoma.py', 'Orchestrator/fairvision_amd.py',
                     'Orchestrator/fairvision_dr.py', 'evaluate_fairvision_glaucoma_baseline.py',
                     'evaluate_fairvision_amd_baseline.py', 'evaluate_fairvision_dr_baseline.py',
                     'run_gdp_progression_llm_multitarget_baseline.py'):
            self.assertIn(path, text)
        self.assertIn('not proof of prompts used in historical runs', text)

    def test_provenance_records_no_runs(self):
        provenance = json.loads((OUT / 'provenance.json').read_text())
        self.assertEqual(provenance['api_calls'], 0)
        self.assertEqual(provenance['training_runs'], 0)
        self.assertEqual(provenance['bootstrap_replicates'], 10000)
        self.assertEqual(provenance['dirichlet_draws'], 200)


if __name__ == '__main__':
    unittest.main()
