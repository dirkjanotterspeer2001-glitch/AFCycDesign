#!/bin/bash
#SBATCH -J afdesign
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --partition=gpu-short
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem 42gb
#SBATCH --time 04:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# -------------------------
# USER SETTINGS
# -------------------------

# TARGET="/home/s2528436/prosculpt_afcycdesign/Examples/input_data/KIT.pdb"
# PEPTIDES_DIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/KIT_test/01/2_mpnn/rosetta/afcyc_ready"
# OUTDIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/KIT_test/01/3_afcycdesign/result_AFCycDesign"
# TARGET="/home/s2528436/prosculpt_afcycdesign/Examples/input_data/DDR2.pdb"
# PEPTIDES_DIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10000/01/2_mpnn/rosetta/afcyc_ready"
# OUTDIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10000/01/3_afcycdesign/result_AFCycDesign"
TARGET="/home/s2528436/prosculpt_afcycdesign/Examples/input_data/DDR2.pdb"
PEPTIDES_DIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10/01/2_mpnn/rosetta/afcyc_ready"
OUTDIR="/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10/01/3_afcycdesign/result_AFCycDesign"
SCORE="${OUTDIR}/score.sc"
SCRIPT="/home/s2528436/ColabDesign/af/examples/afdesign_tools_v2.py"
PARAMS_DIR="/home/s2528436/ColabDesign/af/examples/params"

# -------------------------
# Setup environment
# -------------------------
module purge
module load ALICE/default

# module load CUDA/11.8.0
# module load cuDNN/8.6.0.163-CUDA-11.8.0

module load CUDA/12.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# module load CUDA/12.3.2
# module load cuDNN/8.9.7.29-CUDA-12.3.2

# module load CUDA/12.4.0
# module load cuDNN/9.1.1.17-CUDA-12.4.0


export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu
export JAX_LOG_COMPILES=1


# ENVIRONMENT="~/miniconda3/envs/colabdesign/bin/python"
# ENVIRONMENT="~/miniconda3/envs/afdesign/bin/python"
ENVIRONMENT="~/miniconda3/envs/colabdesign"


echo "Running on GPU: ${CUDA_VISIBLE_DEVICES}"

echo "=== NVIDIA driver + GPU visibility (nvidia-smi) ==="
which nvidia-smi || true
nvidia-smi || true
echo
nvidia-smi -L || true
echo


srun --chdir="$OUTDIR" conda run -p $ENVIRONMENT python3 "${SCRIPT}" binder \
  --target "${TARGET}" \
  --peptides_dir "${PEPTIDES_DIR}" \
  --outdir "${OUTDIR}" \
  --score "${SCORE}" \
  --cyclic \
  --offset_type 2 \
  --data_dir "${PARAMS_DIR}" \
  --watch

