#!/bin/bash
#SBATCH --job-name=4dgs_dual
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=monakhova
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02,awang-compute-01,kim-compute-03,kim-compute-04,kim-compute-05
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --array=0-1

set -euo pipefail

source /home/wl757/.bashrc
export TMPDIR="${TMPDIR:-/tmp}"

usage() {
    echo "Usage: sbatch dual_training_4dgs.slurm [--resume] VIDEO_DIR [MATCH_STRING] [INIT_METHOD] [MEGASAM_ROOT]" >&2
}

find_latest_checkpoint() {
    local run_dir="$1"
    if [[ ! -d "${run_dir}" ]]; then
        return 1
    fi
    local prev_shopt
    prev_shopt="$(shopt -p nullglob || true)"
    shopt -s nullglob
    local ckpts=("${run_dir}"/chkpnt_*.pth)
    eval "${prev_shopt:-shopt -u nullglob}"
    if [[ ${#ckpts[@]} -eq 0 ]]; then
        return 1
    fi
    printf "%s\n" "${ckpts[@]}" | sort -V | tail -n 1
}

RESUME_LATEST=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_LATEST=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

VIDEO_DIR="$1"
MATCH_STRING="${2:-}"
INIT_METHOD="${3:-vggt}"
MEGASAM_ROOT="${4:-../mega-sam}"
SKIP_TRAIN_FLAG="${SKIP_TRAIN:-0}"

if [[ "${INIT_METHOD}" != "vggt" && "${INIT_METHOD}" != "megasam" ]]; then
    echo "ERROR: INIT_METHOD must be 'vggt' or 'megasam'" >&2
    exit 1
fi

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: Video directory '${VIDEO_DIR}' does not exist." >&2
    exit 1
fi

# Derive dataset identifier from the last two path segments: <parent>/<child>
CHILD_NAME="$(basename "${VIDEO_DIR%/}")"
PARENT_NAME="$(basename "$(dirname "${VIDEO_DIR%/}")")"
CHILD_NAME="${CHILD_NAME// /_}"
PARENT_NAME="${PARENT_NAME// /_}"

# Use nested naming: <parent>/<child>
DATA_NAME="${PARENT_NAME}/${CHILD_NAME}"
DATASET_DIR="data/multipleview/${DATA_NAME}"

mkdir -p slurm_logs "$(dirname -- "${DATASET_DIR}")"

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    EXPNAME="multipleview/${DATA_NAME}/combined"
    if [[ -n "${MATCH_STRING}" ]]; then
        FILTER_ARG="--match_string \"${MATCH_STRING}\""
        RUN_TYPE="Full training, filtered evaluation (match_string='${MATCH_STRING}')"
    else
        FILTER_ARG=""
        RUN_TYPE="Full (all cameras)"
    fi
else
    EXPNAME="multipleview/${DATA_NAME}/filtered"
    if [[ -n "${MATCH_STRING}" ]]; then
        FILTER_ARG="--match_string \"${MATCH_STRING}\" --filter_training"
        RUN_TYPE="Filtered training and evaluation (match_string='${MATCH_STRING}')"
    else
        echo "ERROR: Array task 1 requires MATCH_STRING argument" >&2
        exit 1
    fi
fi

echo "=========================================="
echo "4DGAUSSIANS DUAL TRAINING - ARRAY JOB"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Run type: ${RUN_TYPE}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Video dir: ${VIDEO_DIR}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Experiment name: ${EXPNAME}"
echo "Initializer: ${INIT_METHOD}"
if [[ -n "${MATCH_STRING}" ]]; then
    echo "Match string: ${MATCH_STRING}"
fi
if [[ "${RESUME_LATEST}" -eq 1 ]]; then
    echo "Resume: enabled (will reuse latest checkpoint if found)"
fi
echo "=========================================="
echo ""

set +u
conda activate 4dgaussians
set -u

echo ">>> Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo ">>> Cache cleared"
echo ""

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    echo ">>> Converting videos to 4DGaussians format..."
    CONVERT_CMD=(
        python
        video_to_4dgs.py
        --videos "${VIDEO_DIR}"
        --output "${DATASET_DIR}"
        --dataset_name "${DATA_NAME}"
    )

    if [[ "${INIT_METHOD}" == "megasam" ]]; then
        CONVERT_CMD+=(--init_method megasam --megasam_root "${MEGASAM_ROOT}" --megasam_env megasam)
    fi

    echo "-> ${CONVERT_CMD[@]}"
    "${CONVERT_CMD[@]}"

    echo ""
    echo ">>> Conversion complete. Starting training..."
    echo ""
fi

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]]; then
    echo ">>> Waiting for dataset conversion to complete..."

    MAX_WAIT=7200
    ELAPSED=0
    while [[ ! -f "${DATASET_DIR}/.conversion_done" ]]; do
        if [[ ${ELAPSED} -ge ${MAX_WAIT} ]]; then
            echo "ERROR: Timeout waiting for dataset conversion sentinel" >&2
            exit 1
        fi
        sleep 10
        ELAPSED=$((ELAPSED + 10))
    done

    echo ">>> Dataset ready. Starting filtered training..."
    echo ""
fi

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 1 && "${SKIP_TRAIN_FLAG}" != "1" && "${RESUME_LATEST}" != "1" ]]; then
    FILTER_OUTPUT_DIR="output/${EXPNAME}"
    if [[ -d "${FILTER_OUTPUT_DIR}" ]]; then
        echo ">>> Removing stale filtered output at ${FILTER_OUTPUT_DIR}"
        rm -rf "${FILTER_OUTPUT_DIR}"
    fi
elif [[ "${SLURM_ARRAY_TASK_ID}" -eq 1 && "${RESUME_LATEST}" == "1" ]]; then
    echo ">>> Resume flag set; keeping existing filtered output directory."
fi

TRAIN_CMD="python train.py \
    -s \"${DATASET_DIR}\" \
    --expname \"${EXPNAME}\""

if [[ -n "${FILTER_ARG}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} ${FILTER_ARG}"
fi

RUN_OUTPUT_DIR="output/${EXPNAME}"
if [[ "${RESUME_LATEST}" -eq 1 ]]; then
    if START_CHECKPOINT="$(find_latest_checkpoint "${RUN_OUTPUT_DIR}")"; then
        echo ">>> Found checkpoint for resume: ${START_CHECKPOINT}"
        TRAIN_CMD="${TRAIN_CMD} --start_checkpoint \"${START_CHECKPOINT}\""
    else
        echo ">>> Resume requested but no checkpoints found under ${RUN_OUTPUT_DIR}; starting from scratch."
    fi
fi

echo "Executing: ${TRAIN_CMD}"
echo ""

# Run the training
eval "${TRAIN_CMD}"

echo ""
echo "=========================================="
echo "ARRAY TASK ${SLURM_ARRAY_TASK_ID} COMPLETE"
echo "Results saved to: output/${EXPNAME}"
echo "=========================================="

conda deactivate || true
