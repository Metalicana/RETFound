# GDP Native Reliability Rerun

Scope: the **single-target** RNFLT+TDS helper for `td_pointwise_no_p_cut`.
This is not the six-output EfficientNet, an LLM baseline, or an agent run.

The runner now accepts `--target` for any of the six endpoints, training a separate
single-output model for each. For the all-endpoint OOF, final-fit and API workflow,
see [Clean Six-Endpoint GDP Rerun](gdp_progression_clean_suite.md).

## Recovered Implementation

The source exported from CECSL's `~/Harvard-GDP` on 2026-09-28 confirms:

- `scripts/train_progression_pseudo_supervisor.py` constructs train, validation
  and test loaders, but its epoch evaluation iterates over **tst_dataloader**.
  It logs those results under `val_*` and tracks the best test AUC. This establishes
  what that log means, not which checkpoint was subsequently exported.
- `utils/modules.py` constructs an ImageNet-pretrained EfficientNet V2-S with a
  fresh two-channel input convolution, a one-output head and sigmoid.
- `utils/image_datasets.py` removes the first RNFLT row and column, clips to
  [-2, 350] and adds 2. It does **not** divide the RNFLT channel by 352.
  TDS values are transformed with `(tds + 38) / 64 * 2` and placed on the original
  8x9 visual-field grid, expanded and cropped to 224x224.
- Although settings say `label+unlabel`, this loader sets `unlabel_flags=None`.
  For these binary-labeled development cases, training follows the supervised
  branch. Its scheduler steps only in the inactive unlabeled branch. The policy
  network and EMA weights do not contribute to this predictor's supervised updates.

The new runner imports the original model factory and dataset directly. It checks
their source hashes against the downloaded bundle and records helper-source hashes.
It does not edit that repository or any prompts.

## Prespecified Repair

1. Match the manifest's 300 development / 200 test IDs to the old staged NPZ
   directories (240 train + 60 validation = 300 development).
2. Make five stratified development folds, keeping manifest patient IDs together.
   Each development case receives exactly one prediction from a model that did
   not train on it. With one case per patient, each fit uses 240 cases and holds out 60.
3. Train **60 fixed epochs**, batch size 6, BCE, AdamW at 2e-5 with betas (0, 0.1),
   zero weight decay, constant learning rate and no AMP. Each fold starts fresh
   from ImageNet weights. Use seed 3280 for partitioning, 3280+fold for fits.
4. Keep the threshold fixed at 0.5. Do not select epochs, thresholds, seeds,
   coefficients or models on OOF performance or test results.
5. Compute pooled OOF positive-class F1, AUROC, balanced accuracy, FPR, FNR and
   ten-bin ECE using the existing repository metrics. The global trust weight is
   the existing agent formula `(0.70*BA + 0.30*F1)*(1-min(ECE,0.8))`, floored at 0.05.
6. Optional `--fit-final`: train a fresh model on all 300 development cases for
   the same 60 fixed epochs (seed 4280). Save its checkpoint before evaluating the
   200 test cases. Keep test metrics separate from the OOF priors.

This changes the invalid test-informed epoch selection, training partition and
random draws. It is **not** a bitwise reproduction or a way to retrospectively
validate the old agent results. Similar priors would be a descriptive finding only.
OOF estimates describe 240-case fits; the optional final model uses 300 cases.
Patient separation is as reliable as the manifest identifiers; synthetic image IDs
are not independent proof of real-world patient disjointness.

## CECSL Commands

After pushing locally and pulling on CECSL, in `~/RETFound` with `retfound` active:

```bash
OLD=equi-agent/outputs/gdp_native_progression_td_pointwise_no_p_cut_efficientnet_modality2_auc0.8170
OUT="$HOME/RETFound/equi-agent/outputs/gdp_native_oof_v1"
mkdir -p "$OUT"

python equi-agent/scripts/estimate_gdp_native_oof.py \
  --original-args "$OLD/args_train.txt" --out-dir "$OUT" --prepare-only
```

The preparation step validates development NPZ shapes/labels, source hashes and
runtime imports without constructing a model or reading test NPZ contents.
If it passes, launch the reliability experiment on physical GPU 1:

```bash
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  python equi-agent/scripts/estimate_gdp_native_oof.py \
  --original-args "$OLD/args_train.txt" --out-dir "$OUT" --device cuda:0 \
  > "$OUT/run.log" 2>&1 < /dev/null &
echo $! > "$OUT/run.pid"
tail -n 25 "$OUT/run.log"
```

The command makes no LLM calls. Existing experiments remain untouched. It reuses
completed folds only after checking configuration, file hashes, cohort and labels.
A partially trained fold restarts from scratch. Concurrent runs in the same output
directory are blocked. Changed settings/data require a new output directory.
The six-endpoint extension explicitly accepts the known original primary runner's
hash only when every other configuration field is unchanged. It records both code
versions in `execution_version.json`; other code changes require a new directory.

## Outputs

- `resolved_config.json`, `original_args.json`, `fold_assignments.csv`: provenance.
- `fold_*/predictions.csv`, `fold_*/complete.json`: per-fold outputs and receipts.
- `predictions_oof.csv`: all 300 development predictions, `split=oof`.
- `oof_aggregate.csv`, `fold_metrics.csv`, `oof_summary.json`: reliability results.
- `priors/exp8_gdp_progression_forecasting_td_pointwise_no_p_cut_gdp_native_rnflt_tds_efficientnet/`:
  OOF aggregate in the existing agent loader's naming convention. Do not put test
  aggregates here or mix it with the historical metrics root.
- With `--fit-final`: `final/model.pt`, `predictions/*.csv`, `test_metrics/aggregate.csv`
  and `final_summary.json`. No agent is automatically launched.

Next, use the clean helper predictions and OOF prior root for an explicitly
configured, unchanged-prompt agent rerun in a new directory. Check the saved request
and resolved settings against the historical run before attributing any difference
to reliability priors alone. Keep an all-200-case paired comparison and report failures.
