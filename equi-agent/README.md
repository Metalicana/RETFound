# Equi-Agent

Inference-time trust calibration utilities for fair ophthalmic foundation model experiments.

This scaffold intentionally separates reusable experiment infrastructure from the older
`OphthalmicAgent/` prototype. It does not contain computed paper results. Tables and metrics
should stay empty or `TBD` in manuscript templates until real experiments produce prediction
files in the standard schema.

## Minimum Doable Experiments From Current Codebase

1. Experiment 1: standalone reliability gaps, if saved validation/test predictions can be
   converted into the standard prediction schema.
2. Experiment 4: static fusion baselines, once multiple base-model prediction files share
   common samples.
3. Experiment 7: calibration summaries and reliability data, using the same prediction files.

Experiments 2, 5, and 6 become doable after validation-derived subgroup priors and a structured
Equi-Agent arbitration output are available. Experiment 8 needs Harvard-GDP longitudinal
manifests and patient/eye/visit labels before claims can be made.

## Core Utilities

- `src/config.py`: JSON/YAML config loading.
- `src/data/manifest.py`: FairVision-style manifest normalization.
- `src/data/splits.py`: patient/eye split leakage validator.
- `src/data/predictions.py`: standard prediction schema validator.
- `src/metrics/classification.py`: binary diagnostic and calibration metrics.
- `src/metrics/bootstrap.py`: patient-level bootstrap confidence intervals.
- `src/fairness/subgroup.py`: subgroup metrics and disparity summaries.
- `src/ensembles/baselines.py`: majority vote, mean probability, confidence-weighted, and stacking.
- `src/calibration/priors.py`: validation-only subgroup reliability prior builder.
- `src/tables/export.py`: CSV/JSON/LaTeX export helpers.

## Legacy Agent Mirror

The legacy `OphthalmicAgent/` agent architecture has been mirrored into this folder for
local experiments:

`BioProfilerAgent/`, `EquityAgent/`, `FunctionalInterpretationAgent/`, `GuidelinesAgent/`,
`Orchestrator/`, `SafetyAgent/`, `VisionAgent/`, `main.py`, and `main_baseline.py`.

The intentional difference is data loading. `data/loader.py` expects the local mirror layout:

```text
RETFound/
  Datasets/
    FairVision/
      HarvardFairVision30k/{AMD,DR,Glaucoma}/ReadMe/data_summary_*.csv
      Training/data_*.npz
      Validation/data_*.npz
      Test/data_*.npz
      Training/slo_fundus_*.jpg
      Validation/slo_fundus_*.jpg
      Test/slo_fundus_*.jpg
    GDP/
      ReadMe/data_summary.csv
      Bscan/
      RNFLT/
```

Run the mirrored agent from this directory so relative weights and scripts resolve:

```bash
cd equi-agent
python -u -m main
```

Inspect local NPZ formats before wiring model inputs:

```bash
python equi-agent/scripts/smoke_npz_formats.py --max-per-group 1
```

## RETFound and MIRAGE FairVision Training

The refactor trains against the new dataset layout by default:

```text
RETFound/
  Datasets/FairVision/{Training,Validation,Test}/data_*.npz
  Datasets/FairVision/HarvardFairVision30k/{AMD,DR,Glaucoma}/ReadMe/data_summary_*.csv
  equi-agent/weights/{RETFound_mae_natureOCT.pth,oct_model_best.pth,slo_model_best.pth}
  equi-agent/VisionAgent/MIRAGE/MIRAGE-Base.pth
```

From the repo root, run both legacy-equivalent multi-head probes:

```bash
bash equi-agent/scripts/run_fairvision_retfound_mirage_training.sh
```

Or submit/run them separately from `equi-agent`:

```bash
sbatch train_oct.sh
sbatch train_slo.sh
```

Useful overrides:

```bash
FAIRVISION_DATA_ROOT=/home/ab575577/RETFound/Datasets/FairVision
RETFOUND_OCT_BACKBONE_WEIGHTS=/home/ab575577/RETFound/equi-agent/weights/RETFound_mae_natureOCT.pth
RETFOUND_OCT_MODEL_WEIGHTS=/home/ab575577/RETFound/equi-agent/weights/oct_model_best.pth
MIRAGE_DIR=/home/ab575577/RETFound/equi-agent/VisionAgent/MIRAGE
MIRAGE_SLO_MODEL_WEIGHTS=/home/ab575577/RETFound/equi-agent/weights/slo_model_best.pth
```

## Standard Prediction Schema

Every downstream method should emit one row per sample, task, split, and model:

`patient_id, eye_id, visit_id, image_id, dataset, task, model_name, y_true, y_prob, y_pred, split, race, ethnicity, sex_gender, age, age_group, metadata_missing_flag`

Longitudinal experiments add:

`sequence_id, visit_index, visit_date, time_since_baseline, num_visits, progression_label`
