#!/usr/bin/env bash
#SBATCH --job-name=download-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail


# Go to the directory from which sbatch was called
cd "$SLURM_SUBMIT_DIR"


mkdir -p logs

source ./env.sh


# Choose a shared location visible to later training jobs

: "${EXPERIMENT_DATA_DIR:?DATA_DIR environment variable is not set}"

mkdir -p "$EXPERIMENT_DATA_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Data directory: $EXPERIMENT_DATA_DIR"


srun python download_data.py

echo "Download finished successfully."
