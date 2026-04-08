#!/bin/bash
#
# SLURM DIRECTIVES: Configure the resources needed for your final job.
#
#SBATCH --job-name=OCT_Linear_Probing    # Name of job for the queue
#SBATCH --output=slurm_logs/slurm-%j.out  # Standard output log file (where prints go)
#SBATCH --error=slurm_logs/slurm-%j.err   # Standard error log file
#SBATCH --time=04:00:00                # Maximum job run time (5 hours)
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks-per-node=1            # Run one main task
#SBATCH --cpus-per-task=16              # Request 16 CPU cores (Matches your NUM_WORKERS=16 setting)
#SBATCH --gres=gpu:1                   # Request 1 GPU (the resource needed)

# --- SETUP ENVIRONMENT ---
echo "--- Starting job on node $(hostname) ---"

source $(conda info --base)/etc/profile.d/conda.sh

# Load the required modules
module load anaconda

# Activate your conda environment (crucial!)
conda activate eyeAgent

# Check GPU status (Optional, but good for logs)
nvidia-smi

cd /lustre/fs1/home/yu395012/RETFound/OphthalmicAgent

# --- EXECUTE PYTHON SCRIPT ---
echo "Running main training script: linear_probing_oct2.py"

python -u -m VisionAgent.linear_probing_oct2

echo "--- Job finished successfully ---"