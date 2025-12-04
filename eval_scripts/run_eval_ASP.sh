#!/bin/bash

# Per-category evaluation for AnomalyShapeNet using UCA-3DAL eval.py
# NOTE: adjust CKPT_DIR / CPE_CKPT according to your training logs.

# Example subset of categories; extend this list as needed.
categories=("ashtray0" "bag0" "bottle0" "bottle1" "bottle3" "bowl0" "bowl1" "bowl2" "bowl3" "bowl4" 
            "bowl5" "bucket0" "bucket1" "cap0" "cap3" "cap4" "cap5" "cup0" "cup1" "eraser0" 
            "headset0" "headset1" "helmet0" "helmet1" "helmet2" "helmet3" "jar0" "microphone0" "shelf0" "tap0" 
            "tap1" "vase0" "vase1" "vase2" "vase3" "vase4" "vase5" "vase7" "vase8" "vase9")

CKPT_DIR=""       # folder containing Stage-2 CPON checkpoints for AnomalyShapeNet
CKPT_NAME=""  # checkpoint filename (e.g., best.pth)
CPE_CKPT=""       # Stage-1 CPE checkpoint path for AnomalyShapeNet (with prototypes)

# Loop over categories
for category in "${categories[@]}"; do
  echo "[AnomalyShapeNet] Evaluating category: ${category}"

python eval.py \
    --gpu_id 0 \
    --dataset AnomalyShapeNet \
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
    echo "[ERROR] AnomalyShapeNet category ${category} failed."
    exit 1
  fi

  echo "[DONE] AnomalyShapeNet category ${category}"
  echo "---------------------------------------------"
done

echo "[AnomalyShapeNet] All categories evaluated."
