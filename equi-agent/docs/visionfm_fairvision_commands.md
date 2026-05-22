# VisionFM FairVision Commands

Run these commands from the cluster repo root:

```bash
cd /home/ab575577/RETFound
conda activate equi-visionfm
```

Install dependencies if the environment is new:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r equi-agent/requirements.txt
python -m pip install scikit-learn pandas tqdm pillow tabulate munkres miseval einops pyyaml
```

Do not install `Foundation_Models/VisionFM-main/requirements.txt` into a Python
3.11 environment unless you intentionally want to recreate the original
VisionFM environment. That file pins older CUDA/PyTorch and NumPy versions; the
FairVision adapter only needs the lighter dependency set above.

Download the VisionFM weights into the expected folder:

```bash
python -m pip install gdown
mkdir -p Foundation_Models/VisionFM-main/pretrain_weights

# Fundus/SLO encoder
gdown "https://drive.google.com/uc?id=13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4" \
  -O Foundation_Models/VisionFM-main/pretrain_weights/VFM_Fundus_weights.pth

# OCT encoder
gdown "https://drive.google.com/uc?id=1o6E-ine2QLx2pxap-c77u-SU0FjxwypA" \
  -O Foundation_Models/VisionFM-main/pretrain_weights/VFM_OCT_weights.pth
```

Run the SLO/Fundus VisionFM experiments:

```bash
VISIONFM_WEIGHTS=/home/ab575577/RETFound/Foundation_Models/VisionFM-main/pretrain_weights/VFM_Fundus_weights.pth \
MODALITY=slo \
TASKS="amd dr glaucoma" \
BATCH_SIZE=64 \
DEVICE=cuda \
THRESHOLD_METRIC=balanced_accuracy \
bash equi-agent/scripts/run_fairvision_visionfm.sh
```

Run the OCT VisionFM experiments:

```bash
VISIONFM_WEIGHTS=/home/ab575577/RETFound/Foundation_Models/VisionFM-main/pretrain_weights/VFM_OCT_weights.pth \
MODALITY=oct \
TASKS="amd dr glaucoma" \
BATCH_SIZE=64 \
DEVICE=cuda \
THRESHOLD_METRIC=balanced_accuracy \
bash equi-agent/scripts/run_fairvision_visionfm.sh
```

`THRESHOLD_METRIC=balanced_accuracy` avoids the degenerate F1-selected
threshold where imbalanced AMD/DR validation splits can choose threshold `0.0`
and predict every test case as positive.

Outputs are written to:

- `equi-agent/outputs/checkpoints/fairvision_*_visionfm_{slo,oct}_linear_probe.pkl`
- `equi-agent/outputs/predictions/fairvision_*_visionfm_{slo,oct}_{val,test,test_thresholded}.csv`
- `equi-agent/outputs/metrics/exp2_visionfm_{slo,oct}*/`
- `equi-agent/outputs/tables/`

If manifest paths were created on a different machine, add:

```bash
PATH_PREFIX_FROM=/home/ab575577/RETFound \
PATH_PREFIX_TO=/your/local/repo/root
```
