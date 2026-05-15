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
      Training/{AMD,DR,Glaucoma}/data_*.npz
      Validation/{AMD,DR,Glaucoma}/data_*.npz
      Test/{AMD,DR,Glaucoma}/data_*.npz
```

Run the mirrored agent from this directory so relative weights and scripts resolve:

```bash
cd equi-agent
python -u -m main
```

## Standard Prediction Schema

Every downstream method should emit one row per sample, task, split, and model:

`patient_id, eye_id, visit_id, image_id, dataset, task, model_name, y_true, y_prob, y_pred, split, race, ethnicity, sex_gender, age, age_group, metadata_missing_flag`

Longitudinal experiments add:

`sequence_id, visit_index, visit_date, time_since_baseline, num_visits, progression_label`
