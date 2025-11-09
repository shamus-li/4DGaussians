#!/bin/bash
#SBATCH --job-name=4dgs
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --mem=64G

set -euo pipefail

source /home/wl757/.bashrc
export TMPDIR="${TMPDIR:-/tmp}"

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch run_4dgs.slurm <path-to-dataset>" >&2
    exit 1
fi

DATASET_SOURCE="$1"
if [[ ! -d "${DATASET_SOURCE}" ]]; then
    echo "ERROR: Dataset source '${DATASET_SOURCE}' does not exist or is not a directory." >&2
    exit 1
fi

# Derive name from last two segments: <parent>/<child>
CHILD_NAME="$(basename "${DATASET_SOURCE%/}")"
PARENT_NAME="$(basename "$(dirname "${DATASET_SOURCE%/}")")"
CHILD_NAME="${CHILD_NAME// /_}"
PARENT_NAME="${PARENT_NAME// /_}"
# Use nested naming: <parent>/<child>
DATASET_NAME="${PARENT_NAME}/${CHILD_NAME}"

mkdir -p slurm_logs "data/multipleview/$(dirname -- "${DATASET_NAME}")"

echo ">>> Source dataset: ${DATASET_SOURCE}"
echo ">>> Dataset name: ${DATASET_NAME}"
echo ">>> Working directory: ${PWD}"

set +u
conda activate 4dgaussians
set -u

python video_to_4dgs.py \
    --videos "${DATASET_SOURCE}" \
    --output "data/multipleview/${DATASET_NAME}" \
    --dataset_name "${DATASET_NAME}"

python train.py \
    -s "data/multipleview/${DATASET_NAME}" \
    --expname "multipleview/${DATASET_NAME}"

conda deactivate || true
