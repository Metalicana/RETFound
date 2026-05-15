#!/bin/bash
#
# SLURM DIRECTIVES: Configure the resources needed for your final job.
#
#SBATCH --job-name=Main_Script    # Name of job for the queue
#SBATCH --output=slurm_logs/slurm-%j.out  # Standard output log file (where prints go)
#SBATCH --error=slurm_logs/slurm-%j.err   # Standard error log file
#SBATCH --time=20:00:00                # Maximum job run time 
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks-per-node=1            # Run one main task
#SBATCH --cpus-per-task=16              # Request 16 CPU cores (Matches your NUM_WORKERS=16 setting)
#SBATCH --gres=gpu:1                   # Request 1 GPU (the resource needed)

export SERPAPI_KEY="714e89089441f4ffa25801256a94aaf1c78b4ee60afdf0921512317f4acc6b5b"

# --- SETUP ENVIRONMENT ---
echo "--- Starting job on node $(hostname) ---"

source $(conda info --base)/etc/profile.d/conda.sh

# Load the required modules
module load anaconda

# Activate your conda environment (crucial!)
conda activate eyeAgent

# Check GPU status (Optional, but good for logs)
nvidia-smi

cd "$(dirname "$0")"
mkdir -p slurm_logs

# --- EXECUTE PYTHON SCRIPT ---
echo "Running baseline testing script: main_baseline.py"

python -u -m main_baseline

echo "--- Job finished successfully ---"
