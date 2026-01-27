#!/bin/bash
set -euo pipefail

# Initialize conda for this script
eval "$(conda shell.bash hook)"
conda activate nnunet

WIDTHS="${WIDTHS:-1 2 4 8 16 32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCHES="${BATCHES:-3}"
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-results/naswot}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/naswot_unet_widths_$(date +%Y%m%d_%H%M%S).txt}"

mkdir -p "${OUT_DIR}"

for w in ${WIDTHS}; do
  python score_net.py \
    --width "${w}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --batches "${BATCHES}" \
    --device "${DEVICE}" \
    | tee -a "${OUT_FILE}"
done
