from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "equi-agent/scripts/estimate_gdp_native_oof.py"
SPEC = importlib.util.spec_from_file_location("gdp_native_oof", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def cohort(dev_n=40, test_n=10):
    return [
        {"patient_id": f"patient-{i}", "image_id": f"data_{i:04d}.npz",
         "split": "train" if i < dev_n else "test", "y_true": str(i % 2),
         "progression_target": runner.TARGET}
        for i in range(dev_n + test_n)
    ]


def predictions(rows, fold="1", split="oof", signature="signature"):
    return [{**r, "y_prob": 0.8 if runner.label(r) else 0.2,
             "y_pred": runner.label(r), "split": split, "fold": fold,
             "model_name": runner.MODEL, "applied_threshold": 0.5,
             "run_fingerprint": signature} for r in rows]


def saved_settings():
    return {
        "model": "efficientnet", "data_modality": 2, "image_size": 224,
        "loss_type": "bce", "data_type": "label+unlabel",
        "progression_outcome": f"progression_outcome_{runner.TARGET}",
        "num_epochs": 60, "batch_size": 6, "lr_vf": 2e-5, "weight_decay_vf": 0.0,
        "random_seed": 3280, "warmup_steps": 0, "use_fp16": False,
        "resume_checkpoint_vf": "", "resume_checkpoint": "", "data_dir": "/unused",
    }


class GDPNativeOOFTest(unittest.TestCase):
    def test_recipe_rejects_silent_substitution(self):
        self.assertEqual(runner.validate_recipe(saved_settings())["num_epochs"], 60)
        for key, value in [("model", "efficientnet_b1"), ("num_epochs", 10),
                           ("resume_checkpoint_vf", "old_test_selected.pt"),
                           ("progression_outcome", "progression_outcome_md")]:
            settings = saved_settings()
            settings[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_recipe(settings)

    def test_grouped_folds_disjoint_complete_deterministic(self):
        rows = cohort()
        # Two eyes per patient, same label within each patient.
        for i, row in enumerate(rows[:40]):
            row["patient_id"] = f"paired-{i // 2}"
            row["y_true"] = str((i // 2) % 2)
        dev, test = runner.validate_cohort(rows, 40, 10)
        folds = runner.make_folds(dev, 5, 3280)
        self.assertEqual(folds, runner.make_folds(dev, 5, 3280))
        self.assertEqual(set(folds), {runner.case_id(r) for r in dev})
        self.assertFalse(set(folds) & {runner.case_id(r) for r in test})
        for fold in range(1, 6):
            train_groups = {r["patient_id"] for r in dev if folds[runner.case_id(r)] != fold}
            heldout_groups = {r["patient_id"] for r in dev if folds[runner.case_id(r)] == fold}
            self.assertFalse(train_groups & heldout_groups)

    def test_test_labels_do_not_change_development_folds(self):
        rows = cohort()
        dev, _ = runner.validate_cohort(rows, 40, 10)
        original = runner.make_folds(dev, 5, 3280)
        for row in rows[40:]:
            row["y_true"] = str(1 - runner.label(row))
        changed_dev, _ = runner.validate_cohort(rows, 40, 10)
        self.assertEqual(original, runner.make_folds(changed_dev, 5, 3280))

    def test_invalid_cohorts_rejected(self):
        changes = [
            lambda r: r[-1].update(patient_id=r[0]["patient_id"]),
            lambda r: r[1].update(image_id=r[0]["image_id"]),
            lambda r: r[0].update(patient_id=""),
            lambda r: r[0].update(y_true="-1"),
            lambda r: r[0].update(progression_target="md"),
            lambda r: r[0].update(image_id="../case.npz"),
            lambda r: r[0].update(split="unknown"),
        ]
        for change in changes:
            rows = cohort()
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_cohort(rows, 40, 10)

    def test_staged_split_identity_is_checked(self):
        rows = cohort(4, 2)
        dev, test = runner.validate_cohort(rows, 4, 2)
        settings = {"data_subset": "train", "data_val_subset": "val", "data_tst_subset": "test"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("train", "val", "test"):
                (root / name).mkdir()
            for i, row in enumerate(rows):
                subset = "test" if i >= 4 else "val" if i >= 3 else "train"
                (root / subset / row["image_id"]).touch()
            paths = runner.staged_paths(root, settings, dev, test)
            self.assertEqual(len(paths), 6)
            (root / "test" / rows[-1]["image_id"]).rename(root / "train" / rows[-1]["image_id"])
            with self.assertRaisesRegex(ValueError, "Staged NPZs differ"):
                runner.staged_paths(root, settings, dev, test)

    def test_npz_audit_does_not_open_test_inputs(self):
        rows = cohort(2, 2)
        with tempfile.TemporaryDirectory() as temp:
            paths = {runner.case_id(r): Path(temp) / r["image_id"] for r in rows}
            for r in rows[:2]:
                np.savez(paths[runner.case_id(r)], rnflt=np.zeros((225, 225)),
                         tds=np.zeros(52), progression=np.array([0, 0, 0, 0, 0, runner.label(r)]))
            # Test NPZ paths deliberately do not exist.
            hashes = runner.audit_development_npzs(rows[:2], paths)
            self.assertEqual(set(hashes), {runner.case_id(r) for r in rows[:2]})
            wrong = copy.deepcopy(rows[:2])
            wrong[0]["y_true"] = "1"
            with self.assertRaisesRegex(ValueError, "target mismatch"):
                runner.audit_development_npzs(wrong, paths)

    def test_prediction_receipts_reject_tampering_and_incomplete_outputs(self):
        rows = cohort()[:10]
        good = predictions(rows)
        runner.validate_predictions(good, rows, "oof", "1", "signature")
        for field, value in [("y_true", "-1"), ("y_prob", "nan"), ("split", "test"),
                             ("fold", "2"), ("run_fingerprint", "old-run"),
                             ("applied_threshold", 0.3), ("y_pred", 1)]:
            bad = copy.deepcopy(good)
            bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_predictions(bad, rows, "oof", "1", "signature")
        for bad in (good[:-1], good + good[:1]):
            with self.assertRaises(ValueError):
                runner.validate_predictions(bad, rows, "oof", "1", "signature")

    def test_metric_semantics_match_existing_agent(self):
        rows = cohort()[:4]
        result = predictions(rows)
        result[0].update(y_prob=0.8, y_pred=1)
        values = runner.metrics(result, "oof")
        self.assertAlmostEqual(values["f1"], 0.8)
        self.assertEqual(values["balanced_accuracy"], 0.75)
        self.assertEqual(values["fpr"], 0.5)
        self.assertEqual(values["fnr"], 0.0)
        self.assertEqual(values["f1_average"], "binary")
        self.assertEqual(values["prior_source"], "development_oof")
        self.assertAlmostEqual(values["global_trust_weight"], (0.7 * 0.75 + 0.3 * 0.8) * (1 - values["ece"]))

    def test_all_negative_predictions_write_valid_json(self):
        result = predictions(cohort()[:4])
        for row in result:
            row.update(y_prob=0.1, y_pred=0)
        values = runner.metrics(result, "oof")
        self.assertIsNone(values["ppv"])
        json.dumps(values, allow_nan=False)

    def test_complete_partition_resumes_without_model_or_npz_reads(self):
        rows = cohort()
        train, heldout = rows[:30], rows[30:40]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            directory = out / "fold_1"
            csv_path = directory / "predictions.csv"
            result = predictions(heldout)
            runner.write_csv(csv_path, result, runner.FIELDS)
            receipt = {"run_fingerprint": "signature", "seed": 3281,
                       "train_ids": [runner.case_id(r) for r in train],
                       "heldout_ids": [runner.case_id(r) for r in heldout],
                       "epochs": 60, "checkpoint_rule": "last_epoch_fixed_in_advance",
                       "predictions_sha256": runner.sha256(csv_path)}
            runner.write_json(directory / "complete.json", receipt)
            args = runner.argparse.Namespace(out_dir=out)
            with patch.object(runner, "fit_model", side_effect=AssertionError("must not train")):
                loaded = runner.run_partition(args, {"num_epochs": 60}, None, None, {}, train, heldout,
                                              "fold_1", "oof", 3281, "signature")
            self.assertEqual(len(loaded), 10)
            receipt["seed"] = 10
            runner.write_json(directory / "complete.json", receipt)
            with self.assertRaisesRegex(ValueError, "receipt"):
                runner.run_partition(args, {"num_epochs": 60}, None, None, {}, train, heldout,
                                     "fold_1", "oof", 3281, "signature")

    def test_source_changes_cannot_silently_change_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "utils").mkdir()
            for name in runner.NATIVE_HASHES:
                (root / name).write_text("# not the recovered implementation\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                runner.source_hashes(root)

    def test_heldout_loader_created_only_after_fit(self):
        rows = cohort()[:10]
        train, heldout = rows[:6], rows[6:]
        events = []

        def dataset_factory(cls, selected, paths, directory):
            events.append(directory.name)
            return object()

        def fit(*args):
            events.append("fit_complete")
            return object()

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(runner, "native_dataset", side_effect=dataset_factory))
            stack.enter_context(patch.object(runner, "fit_model", side_effect=fit))
            stack.enter_context(patch.object(runner, "predict", return_value=predictions(heldout)))
            args = SimpleNamespace(out_dir=Path(temp), device="cuda:0", num_workers=0)
            runner.run_partition(args, {"num_epochs": 60}, None, None, {}, train, heldout,
                                 "fold_1", "oof", 3281, "signature")
            self.assertEqual(events, ["train", "fit_complete", "oof"])
            self.assertTrue((Path(temp) / "fold_1/complete.json").is_file())

    def test_full_output_flow_keeps_test_metrics_out_of_priors(self):
        # Exercise orchestration/exports with a stub trainer, never a neural model.
        for final in (False, True):
            for incomplete in (False, True):
                with self.subTest(final=final, incomplete=incomplete), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                    out = Path(temp)
                    manifest = out / "manifest.csv"
                    rows = cohort(300, 200)
                    runner.write_csv(manifest, rows)
                    original = out / "args.json"
                    runner.write_json(original, saved_settings())
                    args = SimpleNamespace(native_root=out, original_args=original, manifest=manifest,
                                           data_root=None, out_dir=out, folds=5, num_workers=2,
                                           device="cuda:0", prepare_only=False, fit_final=final)
                    fake_torch = SimpleNamespace(__version__="unit-test", device=lambda x: x,
                                                 cuda=SimpleNamespace(is_available=lambda: True,
                                                                      get_device_properties=lambda _: None))
                    stack.enter_context(patch.dict(sys.modules, {"torch": fake_torch,
                                                                 "torchvision": SimpleNamespace(__version__="unit-test")}))
                    stack.enter_context(patch.object(runner, "source_hashes", return_value={}))
                    stack.enter_context(patch.object(runner, "staged_paths", return_value={}))
                    stack.enter_context(patch.object(runner, "audit_development_npzs", return_value={}))
                    stack.enter_context(patch.object(runner, "load_native", return_value=(None, None)))
                    calls = []

                    def partition(args, recipe, factory, cls, paths, train, heldout, name, split, seed, signature):
                        calls.append((name, train, heldout))
                        self.assertFalse({runner.case_id(r) for r in train} & {runner.case_id(r) for r in heldout})
                        if split == "oof":
                            self.assertTrue(all(r["split"] == "train" for r in train + heldout))
                        else:
                            self.assertEqual(len(train), 300)
                            self.assertTrue(all(r["split"] == "test" for r in heldout))
                            self.assertTrue((out / "oof_summary.json").is_file())
                        result = predictions(heldout, name.removeprefix("fold_"), split, signature)
                        return result[:-1] if incomplete else result

                    stack.enter_context(patch.object(runner, "run_partition", side_effect=partition))
                    stack.enter_context(redirect_stdout(StringIO()))
                    if incomplete:
                        with self.assertRaisesRegex(ValueError, "Incomplete OOF"):
                            runner.execute(args)
                        self.assertFalse((out / "priors").exists())
                        self.assertFalse((out / "final_summary.json").exists())
                        continue
                    runner.execute(args)
                    prior_files = list((out / "priors").rglob("*_aggregate.csv"))
                    self.assertEqual(len(prior_files), 1)
                    prior = runner.read_csv(prior_files[0])[0]
                    self.assertEqual(prior["split"], "oof")
                    self.assertEqual(prior["n"], "300")
                    self.assertEqual(len(calls), 6 if final else 5)
                    self.assertEqual((out / "test_metrics/aggregate.csv").is_file(), final)
                    self.assertEqual((out / "final_summary.json").is_file(), final)

    def test_preparation_never_trains_or_exports_priors(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            out = Path(temp)
            manifest = out / "manifest.csv"
            runner.write_csv(manifest, cohort(300, 200))
            original = out / "args.json"
            runner.write_json(original, saved_settings())
            args = SimpleNamespace(native_root=out, original_args=original, manifest=manifest,
                                   data_root=None, out_dir=out, folds=5, num_workers=2,
                                   device="cuda:0", prepare_only=True, fit_final=True)
            stack.enter_context(patch.dict(sys.modules, {"torch": SimpleNamespace(__version__="unit-test"),
                                                         "torchvision": SimpleNamespace(__version__="unit-test")}))
            stack.enter_context(patch.object(runner, "source_hashes", return_value={}))
            stack.enter_context(patch.object(runner, "staged_paths", return_value={}))
            stack.enter_context(patch.object(runner, "audit_development_npzs", return_value={}))
            stack.enter_context(patch.object(runner, "load_native", return_value=(None, None)))
            partition = stack.enter_context(patch.object(runner, "run_partition", side_effect=AssertionError("must not run")))
            stack.enter_context(redirect_stdout(StringIO()))
            runner.execute(args)
            partition.assert_not_called()
            self.assertTrue((out / "fold_assignments.csv").exists())
            self.assertFalse((out / "oof_summary.json").exists())
            self.assertFalse((out / "priors").exists())
            # A change to the original settings cannot reuse an old run directory.
            settings = saved_settings()
            settings["note"] = "different original configuration"
            runner.write_json(original, settings)
            with self.assertRaisesRegex(ValueError, "configuration/source/data changed"):
                runner.execute(args)


if __name__ == "__main__":
    unittest.main()
