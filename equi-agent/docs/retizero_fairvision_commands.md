# RetiZero FairVision Commands

Run these commands from the cluster repo root:

```bash
cd /home/ab575577/RETFound
conda create -n equi-retizero python=3.11 pip -y
conda activate equi-retizero
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r equi-agent/requirements.txt
python -m pip install scikit-learn pandas tqdm pillow tabulate kornia transformers
```

Download the RetiZero checkpoint:

```bash
python -m pip install gdown
mkdir -p Foundation_Models/RetiZero-main/pretrained

gdown "https://drive.google.com/uc?id=14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM" \
  -O Foundation_Models/RetiZero-main/pretrained/RetiZero.pth
```

Run the SLO/Fundus RetiZero experiments:

```bash
RETIZERO_WEIGHTS=/home/ab575577/RETFound/Foundation_Models/RetiZero-main/pretrained/RetiZero.pth \
TASKS="amd dr glaucoma" \
BATCH_SIZE=32 \
DEVICE=cuda \
THRESHOLD_METRIC=balanced_accuracy \
bash equi-agent/scripts/run_fairvision_retizero.sh
```

Outputs are written to:

- `equi-agent/outputs/checkpoints/fairvision_*_retizero_slo_linear_probe.pkl`
- `equi-agent/outputs/predictions/fairvision_*_retizero_slo_{val,test,test_thresholded}.csv`
- `equi-agent/outputs/metrics/exp2_retizero_slo*/`
- `equi-agent/outputs/tables/`

If manifest paths were created on a different machine, add:

```bash
PATH_PREFIX_FROM=/home/ab575577/RETFound \
PATH_PREFIX_TO=/your/local/repo/root
```
