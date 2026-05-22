# UrFound FairVision Commands

Run these commands from the cluster repo root:

```bash
cd /home/ab575577/RETFound
conda create -n equi-urfound python=3.11 pip -y
conda activate equi-urfound
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r equi-agent/requirements.txt
python -m pip install scikit-learn pandas tqdm pillow tabulate timm huggingface_hub
```

Download the UrFound checkpoint from Hugging Face:

```bash
mkdir -p Foundation_Models/UrFound-main/pretrained

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yyyyk/UrFound",
    local_dir="Foundation_Models/UrFound-main/pretrained",
    local_dir_use_symlinks=False,
)
PY

find Foundation_Models/UrFound-main/pretrained -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \)
```

Set `URFOUND_WEIGHTS` to the downloaded `.pth` checkpoint shown by `find`.
Then run the SLO/Fundus UrFound experiments:

```bash
URFOUND_WEIGHTS=/home/ab575577/RETFound/Foundation_Models/UrFound-main/pretrained/YOUR_CHECKPOINT.pth \
MODALITY=slo \
TASKS="amd dr glaucoma" \
BATCH_SIZE=64 \
DEVICE=cuda \
THRESHOLD_METRIC=balanced_accuracy \
bash equi-agent/scripts/run_fairvision_urfound.sh
```

Run the OCT UrFound experiments:

```bash
URFOUND_WEIGHTS=/home/ab575577/RETFound/Foundation_Models/UrFound-main/pretrained/YOUR_CHECKPOINT.pth \
MODALITY=oct \
TASKS="amd dr glaucoma" \
BATCH_SIZE=64 \
DEVICE=cuda \
THRESHOLD_METRIC=balanced_accuracy \
bash equi-agent/scripts/run_fairvision_urfound.sh
```

Outputs are written to:

- `equi-agent/outputs/checkpoints/fairvision_*_urfound_{slo,oct}_linear_probe.pkl`
- `equi-agent/outputs/predictions/fairvision_*_urfound_{slo,oct}_{val,test,test_thresholded}.csv`
- `equi-agent/outputs/metrics/exp2_urfound_{slo,oct}*/`
- `equi-agent/outputs/tables/`

If manifest paths were created on a different machine, add:

```bash
PATH_PREFIX_FROM=/home/ab575577/RETFound \
PATH_PREFIX_TO=/your/local/repo/root
```
