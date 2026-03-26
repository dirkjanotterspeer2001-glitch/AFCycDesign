#!/bin/bash
#SBATCH --job-name=afcycdesign
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# -------------------------
# USER SETTINGS
# -------------------------

module load ALICE/default
module load CUDA/12.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

mkdir -p logs

python process_afcycdesign.py
