from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "equi-agent/scripts"
sys.path.insert(0, str(SCRIPTS))
import run_gdp_progression_clean_suite as suite
import run_equi_agent_gdp_progression_multitarget_live as agent

native = suite.native


def cohort(target, dev=300, test=200):
    return [{"patient_id": str(i), "eye_id": "", "image_id": f"data_{i:04d}.npz",
             "dataset": "harvard_gdp", "task": "progression_forecasting",
             "split": "train" if i < dev else "test", "y_true": str(i % 2),
             "progression_target": target, "race": "unknown",
             "sex_gender": "female" if i % 4 < 2 else "male", "age_group": "unknown"}
            for i in range(dev + test)]


def predictions(rows, signature, fold, split):
    return [{**r, "run_fingerprint": signature, "model_name": native.MODEL, "fold": fold,
             "split": split, "y_prob": 0.8 if native.label(r) else 0.2,
             "y_pred": native.label(r), "applied_threshold": 0.5} for r in rows]


def arguments(root):
    return SimpleNamespace(run_root=root / "run", primary_run=root / "previous-primary",
                           native_root=root / "Harvard-GDP", manifests_root=root / "manifests",
                           original_args=root / "args.json", llm_root=root / "llms",
                           device="cuda:0", deployment="gpt-5.1", stage="train")


def write_manifests(args):
    for target in suite.TARGETS:
        native.write_csv(suite.target_manifest(args, target), cohort(target))


def write_bundle(args, target, rows):
    """Synthetic receipts exercise file validation without training a model."""
    dev, test = rows
    directory = suite.target_run(args, target)
    folds = native.make_folds(dev, 5, 3280)
    config = {"manifest_sha256": native.sha256(suite.target_manifest(args, target)),
              "recipe": {"num_epochs": 60, "random_seed": 3280}, "final_seed": 4280,
              "folds": 5, "fold_assignments": folds, "threshold": 0.5,
              "progression_target": target, "checkpoint_rule": "last_epoch_fixed_in_advance",
              "test_ids_excluded_from_oof": [native.case_id(r) for r in test]}
    signature = native.fingerprint(config)
    native.write_json(directory / "resolved_config.json", config)
    all_oof = []
    for fold in range(1, 6):
        train = [r for r in dev if folds[native.case_id(r)] != fold]
        heldout = [r for r in dev if folds[native.case_id(r)] == fold]
        result = predictions(heldout, signature, fold, "oof")
        path = directory / f"fold_{fold}/predictions.csv"
        native.write_csv(path, result, native.FIELDS)
        native.write_json(path.with_name("complete.json"), {
            "run_fingerprint": signature, "epochs": 60, "seed": 3280 + fold,
            "checkpoint_rule": "last_epoch_fixed_in_advance", "predictions_sha256": native.sha256(path),
            "train_ids": [native.case_id(r) for r in train],
            "heldout_ids": [native.case_id(r) for r in heldout]})
        all_oof.extend(result)
    native.write_json(directory / "oof_summary.json", {
        "complete_development_oof": True, "cases": 300, "run_fingerprint": signature,
        "test_used_for_fitting_selection_or_priors": False})
    prefix = f"gdp_progression_forecasting_{target}"
    prior = directory / "priors" / f"exp8_{prefix}_{native.MODEL}" / f"{prefix}_{native.MODEL}_aggregate.csv"
    native.write_csv(prior, [{**native.metrics(all_oof, "oof"), "progression_target": target,
                              "run_fingerprint": signature}])
    result = predictions(test, signature, "final", "test")
    path = directory / "final/predictions.csv"
    native.write_csv(path, result, native.FIELDS)
    checkpoint = path.with_name("model.pt")
    checkpoint.write_bytes(b"test fixture, not a trained checkpoint")
    native.write_json(path.with_name("complete.json"), {
        "run_fingerprint": signature, "epochs": 60, "seed": 4280,
        "checkpoint_rule": "last_epoch_fixed_in_advance", "predictions_sha256": native.sha256(path),
        "checkpoint_sha256": native.sha256(checkpoint),
        "train_ids": [native.case_id(r) for r in dev], "heldout_ids": [native.case_id(r) for r in test]})
    native.write_csv(directory / "predictions" / f"{prefix}_{native.MODEL}.csv", result, native.FIELDS)
    native.write_json(directory / "final_summary.json", {
        "complete_test_cohort": True, "cases": 200, "run_fingerprint": signature})


def write_agent_inputs(root):
    for target in suite.TARGETS:
        prefix = f"gdp_progression_forecasting_{target}"
        rows = predictions(cohort(target, 0, 2), "fingerprint", "final", "test")
        native.write_csv(root / "predictions" / f"{prefix}_{native.MODEL}.csv", rows, native.FIELDS)
        prior = {"model_name": native.MODEL, "split": "oof", "n": 300,
                 "prior_source": "development_oof", "f1_average": "binary",
                 "progression_target": target, "run_fingerprint": "fingerprint",
                 "f1": .6, "auroc": .75, "balanced_accuracy": .7,
                 "ece": .2, "fpr": .1, "fnr": .5, "sensitivity": .5, "specificity": .9}
        native.write_csv(root / "priors" / f"exp8_{prefix}_{native.MODEL}" / f"{prefix}_{native.MODEL}_aggregate.csv", [prior])


def agent_args(root):
    with patch.object(sys, "argv", ["agent", "--predictions-root", str(root / "predictions"),
                                     "--metrics-root", str(root / "priors"), "--out-dir", str(root / "agent"),
                                     "--models", native.MODEL, "--expected-cases", "2",
                                     "--require-oof-priors", "--dry-run"]):
        return agent.parse_args()


class CleanGDPSuiteTest(unittest.TestCase):
    def test_commands_preserve_primary_run_and_separate_other_endpoints(self):
        args = arguments(Path("/tmp/example"))
        for target in suite.TARGETS:
            command = suite.native_command(args, target)
            self.assertEqual(command[command.index("--target") + 1], target)
            self.assertIn("--fit-final", command)
            self.assertIn(suite.target_run(args, target), command)
        self.assertEqual(suite.target_run(args, native.TARGET), args.primary_run)
        self.assertEqual(len({suite.target_run(args, t) for t in suite.TARGETS}), 6)
        command = suite.agent_command(args)
        self.assertEqual(command[command.index("--models") + 1], native.MODEL)
        self.assertIn("--require-oof-priors", command)
        self.assertNotIn("retfound_oct", command)
        self.assertIn("--dry-run", suite.agent_command(args, dry_run=True))

    def test_endpoint_membership_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(Path(temp))
            write_manifests(args)
            self.assertEqual(len(suite.check_manifests(args)), 6)
            rows = cohort("md")
            rows[0]["patient_id"] = "different-patient"
            native.write_csv(suite.target_manifest(args, "md"), rows)
            with self.assertRaisesRegex(ValueError, "membership differ"):
                suite.check_manifests(args)

    def test_native_adapter_changes_only_label_selection(self):
        class Dataset:
            def __init__(self, root, **kwargs):
                self.settings = kwargs
                self.rnflt_data = ["data_0000.npz"]
                self.unlabel_flags = None

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.npz"
            source.touch()
            for index, target in enumerate(suite.TARGETS):
                rows = cohort(target, 1, 0)
                dataset = native.native_dataset(Dataset, rows, {"data_0000": source}, root / target, target)
                self.assertEqual(dataset.progression_index, index)
                self.assertEqual(dataset.progression_type, f"progression_outcome_{target}")
                self.assertEqual(dataset.settings["resolution"], 224)
                self.assertEqual(dataset.settings["modality"], 2)

    def test_bundle_checks_and_exports_all_six_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(Path(temp))
            write_manifests(args)
            cohorts = suite.check_manifests(args)
            for target in suite.TARGETS:
                write_bundle(args, target, cohorts[target])
            suite.stage_inputs(args, cohorts)
            # Stubbed agent outputs are only a fixture for the table collector.
            native.write_json(args.run_root / "agent/summary.json", {"dry_run": False, "complete_live_cohort": True})
            for target in suite.TARGETS:
                native.write_csv(args.run_root / "agent" / f"predictions_{target}.csv",
                                 predictions(cohorts[target][1], "fixture", "final", "test"))
            with redirect_stdout(StringIO()):
                suite.collect(args, cohorts)
            rows = native.read_csv(args.run_root / "results/results.csv")
            self.assertEqual(len(rows), 12)
            self.assertEqual({r["target"] for r in rows}, set(suite.TARGETS))
            latex = (args.run_root / "results/results.tex").read_text()
            self.assertEqual(latex.count(r"\begin{table}"), 6)
            self.assertEqual(latex.count(r"\end{table}"), 6)
            self.assertEqual(latex.count(r"\begin{tabular}{lccccc}"), 6)
            self.assertEqual(len(suite.read_json(args.run_root / "results/unavailable_llm_baselines.json")), 24)
            # A changed OOF prior cannot be staged even if final test outputs exist.
            _, prior = suite.checked_bundle(args, "md", cohorts["md"])
            metrics = native.read_csv(prior)
            metrics[0]["f1"] = "0.1"
            native.write_csv(prior, metrics)
            with self.assertRaisesRegex(ValueError, "differs from OOF"):
                suite.checked_bundle(args, "md", cohorts["md"])

    def test_tables_use_forced_labels_not_probability_threshold(self):
        rows = predictions(cohort("md", 0, 4), "fixture", "final", "test")
        for row in rows:
            row["y_prob"] = .1
        values, _ = suite.table_metrics(rows)
        self.assertEqual(values["f1"], 1)
        self.assertEqual(values["sensitivity"], 1)
        self.assertEqual(values["specificity"], 1)
        self.assertIsNone(values["worst_group_f1"])

    def test_rare_targets_do_not_fabricate_worst_group_f1(self):
        rows = predictions(cohort("md_fast", 0, 200), "fixture", "final", "test")
        for i, row in enumerate(rows):
            row.update(y_true=str(int(i < 4)), y_pred=int(i < 4), y_prob=.8 if i < 4 else .2)
        values, groups = suite.table_metrics(rows)
        self.assertIsNone(values["worst_group_f1"])
        self.assertTrue(all(not r["eligible"] for r in groups))
        dense = predictions(cohort("td_pointwise_no_p_cut", 0, 200), "fixture", "final", "test")
        self.assertEqual(suite.table_metrics(dense)[0]["worst_group_f1"], 1)

    def test_prediction_alignment_rejects_missing_cases_and_wrong_labels(self):
        expected = cohort("md", 0, 4)
        good = predictions(expected, "fixture", "final", "test")
        wrong = copy.deepcopy(good)
        wrong[0]["y_true"] = "1"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.csv"
            for rows in (good[:-1], good + good[:1], wrong):
                native.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    suite.align_predictions(path, expected)

    def test_test_priors_and_invalid_inputs_rejected_before_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_agent_inputs(root)
            args = agent_args(root)
            inputs, keys = agent.load_target_inputs(args, ("image_id", "task"))
            self.assertEqual(len(keys), 2)
            prior = Path(inputs["md"]["loaded_priors"][0]["path"])
            original = native.read_csv(prior)
            for field, value in (("split", "test"), ("n", "200"), ("f1", "nan"),
                                 ("progression_target", "vfi"), ("run_fingerprint", "other")):
                rows = copy.deepcopy(original)
                rows[0][field] = value
                native.write_csv(prior, rows)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    agent.load_target_inputs(args, ("image_id", "task"))
            native.write_csv(prior, original)
            path = Path(inputs["md"]["loaded_files"][0]["path"])
            rows = native.read_csv(path)
            native.write_csv(path, rows + rows[:1])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                agent.load_target_inputs(args, ("image_id", "task"))

    def test_strict_resume_rejects_dry_run_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_agent_inputs(root)
            args = agent_args(root)
            args.out_dir.mkdir()
            inputs, _ = agent.load_target_inputs(args, ("image_id", "task"))
            agent.strict_resume_guard(args, inputs)
            args.max_cases = 1
            agent.strict_resume_guard(args, inputs)
            args.dry_run = False
            with self.assertRaisesRegex(ValueError, "changed"):
                agent.strict_resume_guard(args, inputs)
            args.dry_run = True
            path = Path(inputs["md"]["loaded_priors"][0]["path"])
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                agent.strict_resume_guard(args, inputs)

    def test_one_case_smoke_is_not_a_complete_locked_cohort_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_agent_inputs(root)
            args = agent_args(root)
            command = [sys.executable, str(SCRIPTS / "run_equi_agent_gdp_progression_multitarget_live.py"),
                       "--predictions-root", str(args.predictions_root), "--metrics-root", str(args.metrics_root),
                       "--out-dir", str(args.out_dir), "--models", native.MODEL, "--expected-cases", "2",
                       "--require-oof-priors", "--dry-run"]
            result = subprocess.run(command + ["--max-cases", "1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = suite.read_json(args.out_dir / "summary.json")
            self.assertTrue(summary["complete_requested_cohort"])
            self.assertFalse(summary["complete_locked_cohort"])
            self.assertFalse(summary["complete_live_cohort"])
            self.assertEqual(summary["missing_calls"], 1)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(suite.read_json(args.out_dir / "summary.json")["completed_calls"], 2)
            self.assertEqual(len((args.out_dir / "attempts.jsonl").read_text().splitlines()), 2)

    def test_repeated_api_failure_stops_without_fallback_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_agent_inputs(root)
            args = agent_args(root)
            args.dry_run = False
            args.max_retries = 0
            args.max_consecutive_errors = 1
            with patch.object(agent, "parse_args", return_value=args), \
                    patch.object(agent, "make_client", return_value=("azure", object())), \
                    patch.object(agent, "call_llm", side_effect=ValueError("invalid API response")) as call, \
                    redirect_stdout(StringIO()), self.assertRaises(SystemExit) as error:
                agent.main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(call.call_count, 1)
            summary = suite.read_json(args.out_dir / "summary.json")
            self.assertFalse(summary["complete_live_cohort"])
            self.assertEqual(summary["completed_calls"], 0)
            self.assertFalse(any(args.out_dir.glob("predictions_*.csv")))


if __name__ == "__main__":
    unittest.main()
