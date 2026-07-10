#!/bin/bash

#SBATCH --job-name=counterfactual     # Name of job for the queue
#SBATCH --output=slurm_logs/slurm-%j.out  # Standard output log file
#SBATCH --error=slurm_logs/slurm-%j.err   # Standard error log file
#SBATCH --time=4:00:00                 # Maximum job run time 
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks-per-node=1            # Run one main task
#SBATCH --cpus-per-task=16             # Request 16 CPU cores
#SBATCH --gres=gpu:1                   # Request 1 GPU

# --- SETUP ENVIRONMENT ---
echo "--- Starting job on node $(hostname) ---"

# Load the required modules
module load anaconda

# Activate your conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate eyeAgent

# Navigate to working directory
cd /lustre/fs1/home/yu395012/RETFound/OphthalmicAgent

# Create output directory
mkdir -p outputs/glaucoma_counterfactual_250

# --- RUN EXECUTION ---
# Running in the foreground allows Slurm to monitor the process properly.
# Standard output/error will go directly to your Slurm log file.

OUTPUT_CSV="outputs/glaucoma_counterfactual_250/predictions.csv" \
COUNTERFACTUAL_CACHE_PATH="outputs/glaucoma_counterfactual_250/counterfactual_traces.jsonl" \
python -u -m main_new

echo "--- Job finished successfully ---"