#!/bin/bash
#SBATCH -J afdesign
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-short
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=42G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err

set -euo pipefail

# -------------------------
# USER SETTINGS
# -------------------------
TARGET="/home/s2528436/prosculpt_afcycdesign/Examples/input_data/DDR2.pdb"
PEPTIDES_DIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_200/01/2_mpnn/rosetta/afcyc_ready"
OUTDIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_200/01/3_afcycdesign/result_AFCycDesign"
SCORE="${OUTDIR}/score.sc"

SCRIPT="/home/s2528436/ColabDesign/af/examples/afdesign_tools_v2.py"

# Use $HOME (do not use "~" inside variables)
ENVIRONMENT="$HOME/miniconda3/envs/colabdesign"

# Params dir (you had AF_PARAMS but passed PARAMS_DIR)
PARAMS_DIR="/home/s2528436/ColabDesign/af/examples/params"

# Optional (kept from your script though not used below)
WRAPPER="/home/s2528436/prosculpt_afcycdesign/process_afcycdesign.py"

# -------------------------
# Setup environment
# -------------------------
mkdir -p logs
module purge
module load ALICE/default

echo "=== Job context ==="
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Node:  $(hostname)"
echo "CUDA_VISIBLE_DEVICES (outside srun): ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

# -------------------------
# GPU / CUDA proof (run INSIDE srun; many clusters expose GPUs only in job steps)
# -------------------------
srun --ntasks=1 --cpus-per-task=1 bash -lc '
  echo "=== GPU check inside srun ==="
  echo "Node: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo
  echo "which nvidia-smi:"
  command -v nvidia-smi || echo "nvidia-smi not in PATH"
  echo
  echo "nvidia-smi -L:"
  /usr/bin/nvidia-smi -L 2>/dev/null || nvidia-smi -L || true
  echo
  echo "nvidia-smi:"
  /usr/bin/nvidia-smi 2>/dev/null || nvidia-smi || true
  echo
  echo "which nvcc:"
  command -v nvcc || echo "nvcc not in PATH (toolkit may not be loaded; driver can still work)"
  echo
  echo "nvcc --version:"
  nvcc --version || true
'

# -------------------------
# Run AFCycDesign / ColabDesign binder
# -------------------------
srun conda run -p "$ENVIRONMENT" python3 "$SCRIPT" binder \
  --target "$TARGET" \
  --peptides_dir "$PEPTIDES_DIR" \
  --outdir "$OUTDIR" \
  --score "$SCORE" \
  --cyclic \
  --offset_type 2 \
  --data_dir "$PARAMS_DIR" \
  --watch
