# Foundation Model Integration Notes

This document tracks the external foundation models extracted under
`../Foundation_Models/` and how each should be adapted to the Equi-Agent
FairVision/GDP experiment pipeline.

Every integrated model must ultimately emit standard prediction CSVs with the
same schema used by the existing RETFound/MIRAGE experiments:

```text
patient_id, eye_id, visit_id, image_id, dataset, task, model_name,
y_true, y_prob, y_pred, split, race, ethnicity, sex_gender, age, age_group,
metadata_missing_flag
```

The first target is FairVision train/validation/test fine-tuning for `amd`,
`dr`, and `glaucoma`. GDP should be added only for models that can consume OCT
or longitudinal inputs in a scientifically defensible way.

## Shared Experiment Contract

1. Train on `outputs/manifests/fairvision_{task}.csv` rows with `split=train`.
2. Tune thresholds only on `split=val`.
3. Report final metrics only on `split=test`.
4. Save checkpoints to `outputs/checkpoints/`.
5. Save validation and test predictions to `outputs/predictions/`.
6. Run `scripts/tune_thresholds.py` and `scripts/evaluate_predictions.py`.
7. Rebuild tables with `scripts/build_manuscript_tables.py`.

If manifest `image_path` values point to another machine, use:

```bash
PATH_PREFIX_FROM=/home/ab575577/RETFound \
PATH_PREFIX_TO=/Users/metalicana/projects_spring_2026/RETFound \
bash equi-agent/scripts/run_fairvision_supervised_baseline.sh
```

## VisionFM

Source: `../Foundation_Models/VisionFM-main`

Modality fit:
- Fundus/SLO: yes, via Fundus encoder.
- OCT: yes, via OCT encoder, but FairVision OCT volumes must be reduced to a
  compatible 2D slice or adapted to the model's expected OCT input.

Existing entrypoints:
- `finetune_visionfm_for_multiclass_classification.py`
- `inference_visionfm_for_multiclass_classification.py`
- `evaluation/extract_features.py`
- `evaluation/train_cls_decoder.py`
- `evaluation/train_cls_multi_decoder.py`

Adjustment needed:
- Add a FairVision dataset wrapper that reads NPZ files from the manifest and
  returns either `slo_fundus` or the center OCT B-scan as a PIL/RGB image.
- Prefer linear-probe or classifier-head fine-tuning first, because it is easier
  to keep comparable with RETFound/MIRAGE.
- Requires downloaded pretrained weights:
  `VFM_Fundus_weights.pth` and/or `VFM_OCT_weights.pth`.

Initial model names:
- `visionfm_fundus`
- `visionfm_oct`

## UrFound

Source: `../Foundation_Models/UrFound-main`

Modality fit:
- Fundus/SLO: yes.
- OCT: yes, based on its fundus/OCT universal-retina design.

Existing entrypoint:
- `main_finetune.py`

Adjustment needed:
- Its native fine-tuning loader expects an ImageFolder-style downstream dataset.
  Either create FairVision symlink/export folders per task or add a manifest
  dataset class modeled after `finetune/datasets_finetune.py`.
- For our pipeline, the cleaner route is a small Equi-Agent wrapper that imports
  `finetune.models_vit`, loads the UrFound checkpoint, replaces the head with a
  binary classifier, and writes standard prediction CSVs.
- Requires pretrained checkpoint from Hugging Face (`yyyyk/UrFound`) or a local
  path supplied by the user.

Initial model names:
- `urfound_oct`
- `urfound_slo`

## FLAIR

Source: `../Foundation_Models/FLAIR-main`

Modality fit:
- Fundus/SLO: yes.
- OCT: no clear support from the provided repository.

Existing entrypoints:
- `main_transferability.py`
- library usage via `from flair import FLAIRModel`

Adjustment needed:
- Use zero-shot as an optional exploratory baseline, but manuscript comparison
  should use linear probing or a fine-tuned classifier head for fairness with
  other foundation models.
- Extract image embeddings from SLO/fundus images, train task-specific logistic
  heads on FairVision train, tune thresholds on val, test on held-out test.
  Initial runner: `scripts/train_fairvision_flair.py`.
- Requires FLAIR model weights, likely from Hugging Face (`jusiro2/FLAIR`), so
  network/model-cache setup must be confirmed.

Initial model name:
- `flair_slo`

Run notes:
- Full run wrapper: `scripts/run_fairvision_flair.sh`.
- Command reference: `docs/flair_fairvision_commands.md`.

## RetiZero

Source: `../Foundation_Models/RetiZero-main`

Modality fit:
- Fundus/SLO: yes.
- OCT: no clear support from the provided repository.

Existing entrypoints:
- `Finetuning.py`
- `Zeroshot.py`
- `ImageRetrieval.py`

Adjustment needed:
- `Finetuning.py` contains hardcoded placeholder paths; it needs a manifest
  dataset wrapper and CLI arguments for train/val/test CSVs.
- Use SLO/fundus only first.
- Requires downloaded pretrained RetiZero weights from the Google Drive link in
  the README.

Initial model name:
- `retizero_slo`

## RET-CLIP

Source: `../Foundation_Models/RET-CLIP-main`

Modality fit:
- Fundus/SLO: yes.
- OCT: no clear support from the provided repository.

Existing entrypoints:
- `RET_CLIP/training/main.py`
- `RET_CLIP/eval/zeroshot_evaluation.py`
- `RET_CLIP/eval/extract_features.py`

Adjustment needed:
- Use image-embedding extraction plus task-specific linear heads first.
- The released model is Chinese-text CLIP; zero-shot English prompts may be
  inappropriate. Fine-tuned/linear-probe image embeddings are safer for the
  manuscript. Initial runner: `scripts/train_fairvision_ret_clip.py`.
- Requires pretrained RET-CLIP weights from the Google Drive link in the README.

Initial model name:
- `ret_clip_slo`

Run notes:
- Full run wrapper: `scripts/run_fairvision_ret_clip.sh`.
- Command reference: `docs/ret_clip_fairvision_commands.md`.

## Current Unknowns

- Local raw FairVision NPZ files are not present under `Datasets/FairVision` in
  this workspace, even though standard prediction outputs already exist.
- The active Python environment is missing several dependencies (`pandas`,
  `openai`, `torchvision`, etc.).
- The pretrained weights for VisionFM, UrFound, FLAIR, RetiZero, and RET-CLIP
  are not included in the zip archives.
- We need to decide whether SLO-only foundation models should be included in
  agent arbitration alongside OCT models, or reported as separate fundus-only
  baselines before arbitration.
