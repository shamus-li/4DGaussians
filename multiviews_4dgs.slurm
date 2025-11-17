#!/bin/bash
# SLURM array to generate Blender multiview data with gs7, convert to 4DGS, then train here
# Submit from 4DGaussians repo root: sbatch multiviews_4dgs.slurm

#SBATCH --job-name=4dgs_multiview
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --array=0-8

set -eo pipefail

source /home/wl757/.bashrc

# Keep training-data generation inside gs7, but run conversion + training inside this repo
WORKDIR="/home/wl757/multiplexed-pixels/gs7"
FOURD_REPO="$(pwd)"  # this 4DGaussians repo

# Move to gs7 for the sim + blender stages
cd "${WORKDIR}"

if [[ -z "${ADDR2LINE:-}" ]]; then
    if command -v addr2line >/dev/null 2>&1; then
        export ADDR2LINE="$(command -v addr2line)"
    else
        export ADDR2LINE=""
    fi
fi

CAMERA_TYPES=(monocular stereo iphone)
VIEW_COUNTS=(1 3 5)
TOTAL_TYPES=${#CAMERA_TYPES[@]}
TOTAL_VIEWS=${#VIEW_COUNTS[@]}
TOTAL_JOBS=$((TOTAL_TYPES * TOTAL_VIEWS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID ${TASK_ID} out of range (expected 0-$((TOTAL_JOBS - 1)))." >&2
    exit 1
fi

CAMERA_INDEX=$((TASK_ID / TOTAL_VIEWS))
VIEW_INDEX=$((TASK_ID % TOTAL_VIEWS))

CAMERA_MODE="${CAMERA_TYPES[CAMERA_INDEX]}"
N_TRAIN_VIEWS="${VIEW_COUNTS[VIEW_INDEX]}"
ANIMATION_VIEWS=60
ADJACENT_EVAL_FRAMES=0

# ---------------------------------------------------------------------------- #
# User-set configuration shared by all jobs
# ---------------------------------------------------------------------------- #
DATASET_BASE_NAME="demo"
DATASET_SOURCE="/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12"

# ---------------------------------------------------------------------------- #
# Derived configuration for the selected array task
# ---------------------------------------------------------------------------- #
printf -v DATASET_NAME "%s_%s_%dv" "${DATASET_BASE_NAME}" "${CAMERA_MODE}" "${N_TRAIN_VIEWS}"
MODEL_DIR="./output/${DATASET_NAME}"
TRANSFORMS_JSON="${MODEL_DIR}/transforms.json"
BLENDER_BLEND="blender/lego.blend"
BLENDER_SCRIPT="blender/render_blender.py"
RENDER_DIR="./renders/${DATASET_NAME}"
FOURD_DATASET="${FOURD_REPO}/data/multipleview/${DATASET_NAME}"
FOURD_EXPERIMENT="multipleview/${DATASET_NAME}"
FOURD_MODEL_PATH_REL="output/${FOURD_EXPERIMENT}"
FOURD_MODEL_DIR="${FOURD_REPO}/${FOURD_MODEL_PATH_REL}"

TRAIN_ARGS=()
case "${CAMERA_MODE}" in
    monocular)
        CAMERA_DESC="Monocular"
        if (( N_TRAIN_VIEWS == 1 )); then
            ADJACENT_EVAL_FRAMES=6
        fi
        ;;
    stereo)
        CAMERA_DESC="Stereo"
        TRAIN_ARGS+=(--use_stereo --angle_deg 10)
        ;;
    iphone)
        CAMERA_DESC="iPhone"
        TRAIN_ARGS+=(--use_iphone --angle_deg 10)
        ;;
    *)
        echo "ERROR: Unsupported camera mode '${CAMERA_MODE}'." >&2
        exit 1
        ;;
esac

BLENDER_ARGS=(
    --trajectory random_static
    --num-cams "${N_TRAIN_VIEWS}"
    --loc-upper
    --views "${ANIMATION_VIEWS}"
    --arm-mode range
    --arm-z-start 11.0
    --arm-z-end 3.0
)

FOURD_CONVERT_ARGS=(
    --test_blender "/share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego"
    --test_transforms "transforms_val.json"
    --video_blender "/share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego"
    --video_transforms "transforms_val.json"
    --enable_bundle_adjust
    --adjacent_eval_frames "${ADJACENT_EVAL_FRAMES}"
)

FOURD_TRAIN_ARGS=()

# Prefer dataset-specific config if available, otherwise fall back to default.
if [[ -d "${FOURD_MODEL_DIR}" ]]; then
    echo ">>> Removing existing 4DGaussians output at ${FOURD_MODEL_DIR}..."
    rm -rf "${FOURD_MODEL_DIR}"
fi

if [[ -d "${FOURD_DATASET}" ]]; then
    echo ">>> Removing existing converted dataset at ${FOURD_DATASET}..."
    rm -rf "${FOURD_DATASET}"
fi

if [[ -d "${RENDER_DIR}" ]]; then
    echo ">>> Clearing previous Blender renders at ${RENDER_DIR}..."
    rm -rf "${RENDER_DIR}"
fi

mkdir -p "${WORKDIR}/slurm_logs" "${MODEL_DIR}" "${RENDER_DIR}" "${FOURD_REPO}/data/multipleview"

echo "=========================================="
echo "4DGAUSSIANS MULTI-VIEW PIPELINE"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID: ${TASK_ID}"
echo "Camera mode: ${CAMERA_DESC} (${CAMERA_MODE})"
echo "Train views: ${N_TRAIN_VIEWS}"
echo "Dataset base: ${DATASET_SOURCE}"
echo "Dataset name: ${DATASET_NAME}"
echo "Model dir: ${MODEL_DIR}"
echo "Render dir: ${RENDER_DIR}"
echo "4D dataset: ${FOURD_DATASET}"
echo "Experiment: ${FOURD_EXPERIMENT}"
echo "Config: auto (dataset config preferred after conversion)"
echo "=========================================="
echo ""

TRAIN_BASE_ARGS=(
    -s "${DATASET_SOURCE}"
    -m "${MODEL_DIR}"
    --skip_train
    --n_train_images "${N_TRAIN_VIEWS}"
)

BLENDER_BASE_ARGS=(
    --transforms-json "${TRANSFORMS_JSON}"
    --results "${RENDER_DIR}"
)

FOURD_CONVERT_BASE_ARGS=(
    --blender "${RENDER_DIR}"
    --output "${FOURD_DATASET}"
    --dataset_name "${DATASET_NAME}"
)

FOURD_TRAIN_BASE_ARGS=(
    -s "${FOURD_DATASET}"
    --expname "${FOURD_EXPERIMENT}"
)

# ---------------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------------- #
echo ">>> Generating camera trajectories (${CAMERA_DESC}, ${N_TRAIN_VIEWS} view[s])..."
conda run -n gaussian_splatting python train_sim_multiviews.py \
    "${TRAIN_BASE_ARGS[@]}" \
    "${TRAIN_ARGS[@]}"
echo ""

echo ">>> Rendering Blender frames..."
TOTAL_RENDER_FRAMES=$((ANIMATION_VIEWS * N_TRAIN_VIEWS))
if (( TOTAL_RENDER_FRAMES <= 0 )); then
    TOTAL_RENDER_FRAMES=1
fi

CACHE_ROOT="/share/monakhova/shamus_data/multiplexed_pixels/4dgs_cache"
mkdir -p "${CACHE_ROOT}"

BLENDER_ARG_FLAGS=()
for arg in "${BLENDER_BASE_ARGS[@]}" "${BLENDER_ARGS[@]}"; do
    BLENDER_ARG_FLAGS+=(--blender-arg="${arg}")
done

conda run -n 4dgaussians python "${FOURD_REPO}/scripts/render_with_cache.py" \
    --blend-file "${BLENDER_BLEND}" \
    --render-script "${BLENDER_SCRIPT}" \
    --render-dir "${RENDER_DIR}" \
    --transforms-json "${TRANSFORMS_JSON}" \
    --dataset-name "${DATASET_NAME}" \
    --n-train-views "${N_TRAIN_VIEWS}" \
    --animation-views "${ANIMATION_VIEWS}" \
    --expected-frames "${TOTAL_RENDER_FRAMES}" \
    --cache-root "${CACHE_ROOT}" \
    --progress-interval 20 \
    "${BLENDER_ARG_FLAGS[@]}"
echo ""

echo ">>> Converting Blender outputs to 4DGaussians dataset..."
conda run -n 4dgaussians python "${FOURD_REPO}/blender_to_4dgs.py" \
    "${FOURD_CONVERT_BASE_ARGS[@]}" \
    "${FOURD_CONVERT_ARGS[@]}"
echo ""

DATASET_CONFIG_PATH="${FOURD_DATASET}/config.py"
ARG_CONFIG_PATH="${FOURD_REPO}/arguments/multipleview/${DATASET_NAME}.py"
if [[ -f "${DATASET_CONFIG_PATH}" ]]; then
    echo ">>> Using dataset-specific config at ${DATASET_CONFIG_PATH}"
    FOURD_TRAIN_ARGS+=(--configs "${DATASET_CONFIG_PATH}")
elif [[ -f "${ARG_CONFIG_PATH}" ]]; then
    echo ">>> Dataset config missing; using ${ARG_CONFIG_PATH}"
    FOURD_TRAIN_ARGS+=(--configs "arguments/multipleview/${DATASET_NAME}.py")
else
    echo ">>> Dataset config missing; falling back to arguments/multipleview/default.py"
    FOURD_TRAIN_ARGS+=(--configs "arguments/multipleview/default.py")
fi
echo ""

echo ">>> Clearing Python caches in ${FOURD_REPO}..."
find "${FOURD_REPO}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "${FOURD_REPO}" -name "*.pyc" -delete 2>/dev/null || true
echo ""

FOURD_MODEL_DIR="${FOURD_REPO}/${FOURD_MODEL_PATH_REL}"
mkdir -p "${FOURD_MODEL_DIR}"
if [[ -f "${FOURD_MODEL_DIR}/cfg_args" ]]; then
    echo ">>> Removing stale cfg_args at ${FOURD_MODEL_DIR}/cfg_args..."
    rm -f "${FOURD_MODEL_DIR}/cfg_args"
fi

pushd "${FOURD_REPO}" >/dev/null
echo ">>> Training 4DGaussians model (includes evaluation & metrics)..."
conda run -n 4dgaussians python train.py \
    "${FOURD_TRAIN_BASE_ARGS[@]}" \
    "${FOURD_TRAIN_ARGS[@]}"
popd >/dev/null

echo ""
echo "=========================================="
echo "ARRAY TASK ${TASK_ID} COMPLETE (${CAMERA_DESC}, ${N_TRAIN_VIEWS} view[s])"
echo "Results saved to: ${FOURD_REPO}/${FOURD_MODEL_PATH_REL}"
echo "=========================================="
