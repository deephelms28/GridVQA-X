# Model Training

Fine-tunes (or trains from scratch) an **MDETR** model with an EfficientNet-B5
backbone on the GridVQA grounding task.

## Prerequisites

These scripts `import` from the MDETR package (`from models import mdetr`), so
they must be run from inside a **patched MDETR checkout**:

1. Clone upstream MDETR and apply our patch — see [`../mdetr/README.md`](../mdetr/README.md).
2. Copy the scripts in this folder into the root of that checkout:
   ```bash
   cp train_mdetr_finetune_gridvqa.py train_mdetr_scratch_gridvqa.py \
      run_ddp.sh run_single.sh  /path/to/mdetr/
   ```
3. Generate the dataset first — see [`../dataset_generation/README.md`](../dataset_generation/README.md).
4. Obtain the pretrained MDETR EfficientNet-B5 checkpoint (`pretrained_EB5_checkpoint.pth`)
   from the [upstream MDETR model zoo](https://github.com/ashkamath/mdetr#model-zoo),
   or pass `--mdetr_ckpt hub` to download via torch hub.

> Already-trained GridVQA checkpoints (pure / spurious) are available on the
> Hugging Face Hub at
> [`Aikyam-Lab/gridvqa-models`](https://huggingface.co/Aikyam-Lab/gridvqa-models)
> if you only want to evaluate rather than retrain.

## Scripts

| Script | Purpose |
|--------|---------|
| `train_mdetr_finetune_gridvqa.py` | **Main entry point.** Fine-tunes from the pretrained EB5 checkpoint on GridVQA. |
| `train_mdetr_scratch_gridvqa.py`  | Trains the same architecture from scratch (no pretrained weights). |
| `run_single.sh` | Convenience launcher for single-GPU fine-tuning. |
| `run_ddp.sh`    | Convenience launcher for multi-GPU (DistributedDataParallel) fine-tuning. |

## Usage

Edit the paths at the top of the launcher (or export `GRIDVQA_ROOT`,
`MDETR_CKPT`, `OUTPUT_DIR`) and run from the MDETR checkout root:

```bash
# Single GPU
GRIDVQA_ROOT=/data/GridVQA MDETR_CKPT=/ckpts/pretrained_EB5_checkpoint.pth \
  ./run_single.sh

# Multi-GPU (e.g. 3 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 \
  GRIDVQA_ROOT=/data/GridVQA MDETR_CKPT=/ckpts/pretrained_EB5_checkpoint.pth \
  ./run_ddp.sh
```

Or call the script directly:

```bash
python train_mdetr_finetune_gridvqa.py \
    --train_jsonl /data/GridVQA/grounding_train.jsonl \
    --val_jsonl   /data/GridVQA/grounding_val.jsonl \
    --images_root /data/GridVQA \
    --mdetr_ckpt  /ckpts/pretrained_EB5_checkpoint.pth \
    --output_dir  ./checkpoints \
    --batch_size 4 --epochs_stage 4 --lr 2.5e-5 --lr_backbone 1e-5
```

### Key arguments

| Flag | Default | Meaning |
|------|---------|---------|
| `--train_jsonl` / `--val_jsonl` | (required) | GridVQA grounding annotation files. |
| `--images_root` | (required) | Directory containing the rendered images. |
| `--mdetr_ckpt` | `hub` | Path to the pretrained EB5 checkpoint, or `hub` to download. |
| `--output_dir` | `./checkpoints` | Where checkpoints are written. |
| `--batch_size` | `4` | Per-GPU batch size. |
| `--epochs_stage` | `4` | Epochs to train. |
| `--lr` / `--lr_backbone` | `2.5e-5` / `1e-5` | Learning rates for heads vs. backbone. |
| `--freeze_text_encoder` | off | Freeze the RoBERTa text encoder. |
| `--freeze_backbone_layers` | `0` | Number of backbone layers to freeze. |
| `--wandb` / `--wandb_project` | off | Enable Weights & Biases logging. |

Run `python train_mdetr_finetune_gridvqa.py --help` for the full list.

To train on the **spurious** variant, point `--train_jsonl` / `--val_jsonl` /
`--images_root` at the dataset produced by `dataset_generation/spurious/`.

After training, evaluate the resulting checkpoint with the scripts in
[`../evaluation/`](../evaluation/).
