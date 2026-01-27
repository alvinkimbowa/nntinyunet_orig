#!/bin/bash
set -euo pipefail

# Initialize conda for this script
eval "$(conda shell.bash hook)"
conda activate nnunet

export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"

WIDTHS="${WIDTHS:-1 2 4 8 16 32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCHES="${BATCHES:-3}"
DEVICE="${DEVICE:-cuda}"
IN_CHANNELS="${IN_CHANNELS:-3}"
DATASET_NAME="${DATASET_NAME:-Dataset300_isic2018}"
SPLIT="${SPLIT:-Tr}"
SPLIT_TYPE="${SPLIT_TYPE:-train}"
FOLD="${FOLD:-0}"
OUT_DIR="${OUT_DIR:-results/naswot}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/naswot_unet_widths_$(date +%Y%m%d_%H%M%S).txt}"

mkdir -p "${OUT_DIR}"

for w in ${WIDTHS}; do
  python score_net.py \
    --width "${w}" \
    --in_channels "${IN_CHANNELS}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --batches "${BATCHES}" \
    --device "${DEVICE}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --split_type "${SPLIT_TYPE}" \
    --fold "${FOLD}" \
    | tee -a "${OUT_FILE}"
done


python plot_naswot_vs_params.py
