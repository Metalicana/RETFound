# RET-CLIP FairVision Commands

RET-CLIP is integrated as an SLO/fundus-only foundation model baseline. The
first experiment freezes the RET-CLIP image encoder, extracts fundus/SLO image
embeddings, trains a balanced logistic linear probe for each FairVision task,
tunes thresholds on validation, evaluates on test, and rebuilds manuscript
tables.

## Create Environment

You can use a separate environment:

```bash
conda create -n equi-retclip python=3.11 pip -y
conda activate equi-retclip

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r equi-agent/requirements.txt
python -m pip install -r Foundation_Models/RET-CLIP-main/requirements.txt
```

Or reuse `equi-flair` if it already has the needed packages:

```bash
conda activate equi-flair
python -m pip install -r Foundation_Models/RET-CLIP-main/requirements.txt
```

## Required Inputs

- Raw FairVision NPZ files readable from the existing manifest paths.
- RET-CLIP pretrained checkpoint from the link in
  `Foundation_Models/RET-CLIP-main/README.md`.

On the cluster, the project base is:

```text
/home/ab575577/RETFound
```

The existing manifests already use this prefix, so do not pass path rewrite
variables when running on the cluster.

## Smoke Run

From `/home/ab575577/RETFound`:

```bash
conda activate equi-retclip

python equi-agent/scripts/train_fairvision_ret_clip.py \
  --task amd \
  --ret-clip-weights /path/to/ret_clip_checkpoint.pt \
  --limit-train 32 \
  --limit-val 16 \
  --limit-test 16 \
  --device cpu \
  --out-val equi-agent/outputs/predictions/fairvision_amd_ret_clip_slo_smoke_val.csv \
  --out-test equi-agent/outputs/predictions/fairvision_amd_ret_clip_slo_smoke_test.csv
```

## Full Run

```bash
conda activate equi-retclip

RET_CLIP_WEIGHTS=/path/to/ret_clip_checkpoint.pt \
TASKS="amd dr glaucoma" \
BATCH_SIZE=32 \
DEVICE=cuda \
bash equi-agent/scripts/run_fairvision_ret_clip.sh
```

Outputs:

```text
equi-agent/outputs/predictions/fairvision_{task}_ret_clip_slo_val.csv
equi-agent/outputs/predictions/fairvision_{task}_ret_clip_slo_test.csv
equi-agent/outputs/predictions/fairvision_{task}_ret_clip_slo_test_thresholded.csv
equi-agent/outputs/predictions/fairvision_ret_clip_slo_test_thresholded.csv
equi-agent/outputs/metrics/exp2_ret_clip_slo/
equi-agent/outputs/checkpoints/fairvision_{task}_ret_clip_slo_linear_probe.pkl
```
