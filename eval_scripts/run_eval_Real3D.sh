#!/bin/bash

# Per-category evaluation for Real3D-AD using UCA-3DAL eval.py
# NOTE: adjust CKPT_DIR / CPE_CKPT according to your training logs.

categories=(
  "airplane" "candybar" "car" "chicken" "diamond" "duck" "fish" "gemstone" \
  "seahorse" "shell" "starfish" "toffees"
)

CKPT_DIR=""       # folder containing Stage-2 CPON checkpoints for Real3D-AD
CKPT_NAME="best.pth"  # checkpoint filename (e.g., best.pth)
CPE_CKPT=""       # Stage-1 CPE checkpoint path for Real3D-AD (with prototypes)

for category in "${categories[@]}"; do
  echo "[Real3D] Evaluating category: ${category}"

  python eval.py \
    --gpu_id 0 \
    --dataset Real3D \
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
    --smooth_knn 28 \
    --tta_views 4 \
    --tta_rotate_deg 5.0 \
    --tta_scale 0.05 \
    --tta_jitter 0.002 \
    --score_method quantile \
    --score_quantile 0.999 \
    --print_pos_rate \
    --cache_io

  if [ $? -ne 0 ]; then
    echo "[ERROR] Real3D category ${category} failed."
    exit 1
  fi

  echo "[DONE] Real3D category ${category}"
  echo "---------------------------------------------"
done

echo "[Real3D] All categories evaluated."
