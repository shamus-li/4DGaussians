#!/bin/bash

# Convert Blender multi-cam data to 4DGaussians format
# Usage: ./convert_blender_data.sh [dataset_name]

DATASET_NAME=${1:-"lego_blender"}
BLENDER_DIR="gs7/blender/results"
OUTPUT_DIR="data/multipleview/${DATASET_NAME}"

echo "Converting Blender data to 4DGaussians format..."
echo "Dataset name: ${DATASET_NAME}"
echo "Blender dir: ${BLENDER_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

# Create output directory structure
mkdir -p "${OUTPUT_DIR}"

# Path to VGGT demo script (adjust if needed)
VGGT_SCRIPT="${VGGT_SCRIPT:-$HOME/repos/vggt/demo_colmap.py}"
VGGT_CONDA_ENV="${VGGT_CONDA_ENV:-transformers}"

# Run the conversion script (VGGT generates the point cloud)
python blender_to_4dgs.py \
    --blender_dir "${BLENDER_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --vggt_script "${VGGT_SCRIPT}" \
    --vggt_conda_env "${VGGT_CONDA_ENV}"

echo ""
echo "Conversion complete!"
echo ""
echo "Next steps:"
echo "1. Train 4DGaussians:"
echo "   cd 4DGaussians"
echo "   python train.py -s ${OUTPUT_DIR} --port 6017 --expname \"multipleview/${DATASET_NAME}\" --configs arguments/multipleview/${DATASET_NAME}.py"
