#!/usr/bin/env bash
# run_ddp.sh - launcher for distributed (multi-GPU) fine-tuning of MDETR on GridVQA.
#
# Run this from the root of your patched MDETR checkout, after copying
# train_mdetr_finetune_gridvqa.py into that directory (see ../mdetr/README.md).
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2 ./run_ddp.sh

# ---- Edit these ----
NGPUS=${NGPUS:-3}
MASTER_PORT=${MASTER_PORT:-12355}

GRIDVQA_ROOT="${GRIDVQA_ROOT:-/path/to/GridVQA}"     # output of dataset_generation
MDETR_CKPT="${MDETR_CKPT:-/path/to/pretrained_EB5_checkpoint.pth}"  # or 'hub'
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
# --------------------

TRAIN_JSONL="$GRIDVQA_ROOT/grounding_train.jsonl"
VAL_JSONL="$GRIDVQA_ROOT/grounding_val.jsonl"
IMAGES_ROOT="$GRIDVQA_ROOT"
BATCH_SIZE=4
EPOCHS_STAGE=4
LR=2.5e-5
LR_BACKBONE=1e-5
NUM_WORKERS=6
WANDB="--wandb"   # set to empty string to disable
WANDB_PROJECT="Stage 1 ft MDETR on GridVQA"

export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS=1

echo "Launching DDP with $NGPUS GPUs"
echo "TRAIN_JSONL = $TRAIN_JSONL"
echo "VAL_JSONL   = $VAL_JSONL"
echo "IMAGES_ROOT = $IMAGES_ROOT"
echo "MDETR_CKPT  = $MDETR_CKPT"
echo "OUTPUT_DIR  = $OUTPUT_DIR"
echo "BATCH_SIZE  = $BATCH_SIZE (per-GPU)"
echo "EPOCHS_STAGE= $EPOCHS_STAGE"

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$MASTER_PORT \
    train_mdetr_finetune_gridvqa.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --images_root $IMAGES_ROOT \
    --mdetr_ckpt $MDETR_CKPT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs_stage $EPOCHS_STAGE \
    --lr $LR \
    --lr_backbone $LR_BACKBONE \
    --num_workers $NUM_WORKERS \
    $WANDB \
    --wandb_project "$WANDB_PROJECT"
