# FLAIR FairVision Commands

FLAIR is integrated as an SLO/fundus-only foundation model baseline. The first
experiment trains a balanced logistic linear probe on frozen FLAIR image
embeddings for each FairVision task, tunes thresholds on validation, evaluates
on test, and rebuilds manuscript tables.

## Create Environment

```bash
conda create -n equi-flair python=3.11 -y
conda activate equi-flair

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r equi-agent/requirements.txt
python -m pip install -r Foundation_Models/FLAIR-main/requirements.txt
python -m pip install huggingface_hub transformers safetensors kornia
```

If your machine has CUDA-specific requirements, install the matching PyTorch
build before the remaining packages.

## Required Inputs

- Raw FairVision NPZ files readable from the paths in
  `equi-agent/outputs/manifests/fairvision_{task}.csv`, or provide path rewrite
  variables.
- FLAIR pretrained weights, either:
  - Hugging Face access/cache via `FROM_HF=true`, or
  - a local checkpoint path via `FLAIR_WEIGHTS=/path/to/flair_weights.pth`.

On the cluster, the project base is:

```text
/home/ab575577/RETFound
```

The existing manifests already use this prefix, so do not pass
`--path-prefix-from` or `--path-prefix-to` when running on the cluster.

## Smoke Run

```bash
conda activate equi-flair

python equi-agent/scripts/train_fairvision_flair.py \
  --task amd \
  --from-hf \
  --limit-train 32 \
  --limit-val 16 \
  --limit-test 16 \
  --device cpu \
  --out-val equi-agent/outputs/predictions/fairvision_amd_flair_slo_smoke_val.csv \
  --out-test equi-agent/outputs/predictions/fairvision_amd_flair_slo_smoke_test.csv
```

## Full Run

From the repo root:

```bash
conda activate equi-flair

FROM_HF=true \
TASKS="amd dr glaucoma" \
BATCH_SIZE=16 \
DEVICE=cuda \
bash equi-agent/scripts/run_fairvision_flair.sh
```

With local FLAIR weights instead of Hugging Face:

```bash
FLAIR_WEIGHTS=/path/to/flair_weights.pth \
TASKS="amd dr glaucoma" \
BATCH_SIZE=16 \
DEVICE=cuda \
bash equi-agent/scripts/run_fairvision_flair.sh
```

Outputs:

```text
equi-agent/outputs/predictions/fairvision_{task}_flair_slo_val.csv
equi-agent/outputs/predictions/fairvision_{task}_flair_slo_test.csv
equi-agent/outputs/predictions/fairvision_{task}_flair_slo_test_thresholded.csv
equi-agent/outputs/predictions/fairvision_flair_slo_test_thresholded.csv
equi-agent/outputs/metrics/exp2_flair_slo/
equi-agent/outputs/checkpoints/fairvision_{task}_flair_slo_linear_probe.pkl
```
