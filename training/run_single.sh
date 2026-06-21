#!/bin/bash
set -e

# Single-GPU fine-tuning launcher for MDETR on GridVQA.
#
# Run this from the root of your patched MDETR checkout, after copying
# train_mdetr_finetune_gridvqa.py into that directory (see ../mdetr/README.md).
#
# Edit the paths below to point at your generated GridVQA data and the
# pretrained EfficientNet-B5 MDETR checkpoint.

# Optional: activate a conda environment. Comment out if not using conda.
CONDA_ENV="${CONDA_ENV:-mdetr_env}"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# ---- Edit these ----
GRIDVQA_ROOT="${GRIDVQA_ROOT:-/path/to/GridVQA}"      # output of dataset_generation
MDETR_CKPT="${MDETR_CKPT:-/path/to/pretrained_EB5_checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
# --------------------

GPU_ID=${GPU_ID:-0}
LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/train_single_gpu_gpu${GPU_ID}_${TIMESTAMP}.log"

echo "Launching single-GPU training on CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "Logs -> $LOGFILE"

CUDA_VISIBLE_DEVICES=$GPU_ID python train_mdetr_finetune_gridvqa.py \
    --train_jsonl "$GRIDVQA_ROOT/grounding_train.jsonl" \
    --val_jsonl "$GRIDVQA_ROOT/grounding_val.jsonl" \
    --images_root "$GRIDVQA_ROOT" \
    --mdetr_ckpt "$MDETR_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --epochs_stage 4 \
    --lr 2.5e-5 \
    --lr_backbone 1e-5 \
    --num_workers 6 \
    --wandb \
    --wandb_project "Stage 1 ft MDETR on GridVQA" \
    > "$LOGFILE" 2>&1
