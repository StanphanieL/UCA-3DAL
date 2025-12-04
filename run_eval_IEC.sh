#!/bin/bash

# Per-category evaluation for IEC3DAD using UCA-3DAL eval.py
# NOTE: adjust CKPT_DIR / CPE_CKPT / IEC_ROOT according to your setup.

categories=(
  "ButterflyBolt" "ButterflyNut" "DoubleEndStud" "FenderRing" "HexagonalBolt" \
  "HexagonalNut" "HoleRetainingRing" "JointingStud" "KNut" "PullingNail" \
  "RoundNut" "SlipknotBolt" "SquareWeldedNut" "TScrew" "Washer"
)

CKPT_DIR=""       # folder containing Stage-2 CPON checkpoints for IEC3DAD
CKPT_NAME="best.pth"  # checkpoint filename (e.g., best.pth)
CPE_CKPT=""       # Stage-1 CPE checkpoint path for IEC3DAD (with prototypes)
IEC_ROOT=""       # root of IEC3DAD dataset (folder containing category subfolders)

LOG_DIR="./logs/UCA-3DAL/IEC3DAD"
mkdir -p "${LOG_DIR}"

for category in "${categories[@]}"; do
  echo "[IEC3DAD] Evaluating category: ${category}"

python eval.py \
    --gpu_id 0 \
    --dataset IEC3DAD \
    --iec_root "${IEC_ROOT}" \
    --categories all \
    --category "" \
    --eval_category_only "${category}" \
    --class_embed_dim 32 \
    --conditional_mode film \
    --logpath "${CKPT_DIR}" \
    --checkpoint_name "${CKPT_NAME}" \
    --contrastive_ckpt "${CPE_CKPT}" \
    --proj_dim 128 \
    --cluster_norm \
    --cluster_norm_type mad \
    --smooth_knn 16 \
    --tta_views 3 \
    --tta_rotate_deg 5.0 \
    --tta_scale 0.05 \
    --tta_jitter 0.002 \
    --score_method quantile \
    --score_quantile 0.99 \
    --print_pos_rate \
    --cache_io

  if [ $? -ne 0 ]; then
    echo "[ERROR] IEC3DAD category ${category} failed."
    exit 1
  fi

  echo "[DONE] IEC3DAD category ${category}"
  echo "---------------------------------------------"
done

echo "[IEC3DAD] All categories evaluated."