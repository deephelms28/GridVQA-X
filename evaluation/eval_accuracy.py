#!/usr/bin/env python3
"""
Bucket-wise accuracy evaluation on GridVQA test sets.

Usage:
    python eval_accuracy.py --config pure
    python eval_accuracy.py --config spurious
    python eval_accuracy.py --config custom \
        --grounding /path/to/grounding_test.jsonl \
        --qa /path/to/qa_test.jsonl \
        --images /path/to/images \
        --checkpoint /path/to/checkpoint.pth \
        --output results.json
"""

import torch
import json
import os
import tqdm
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import argparse

from models import mdetr as mdetr_module

# ========================
# Preset configs
# ========================
# Edit these to point at your generated datasets and trained checkpoints,
# or use --config custom and pass paths on the command line.
PURE_BASE = "/path/to/GridVQA"        # output of dataset_generation/pure
SPUR_BASE = "/path/to/GridVQA_SP"     # output of dataset_generation/spurious
CKPT_DIR = "/path/to/checkpoints"

PRESETS = {
    "pure": {
        "grounding": f"{PURE_BASE}/grounding_test.jsonl",
        "qa":        f"{PURE_BASE}/qa_test.jsonl",
        "images":    f"{PURE_BASE}/images",
        "checkpoint": f"{CKPT_DIR}/checkpoint_pure.pth",
        "output":    "pure_accuracy_results.json",
    },
    "spurious": {
        "grounding": f"{SPUR_BASE}/grounding_test.jsonl",
        "qa":        f"{SPUR_BASE}/qa_test.jsonl",
        "images":    f"{SPUR_BASE}/images",
        "checkpoint": f"{CKPT_DIR}/checkpoint_spur.pth",
        "output":    "spurious_accuracy_results.json",
    },
}

# -----------------------
# Vocab
# -----------------------
def get_vocab():
    v = {str(i): i for i in range(101)}
    v['yes'] = 101
    v['no'] = 102
    return v

VOCAB = get_vocab()
INV_VOCAB = {v: k for k, v in VOCAB.items()}

# -----------------------
# Bucket from path
# -----------------------
def get_bucket_from_path(path):
    parts = path.split('/')
    depth = qtype = form = dens = "Unk"
    for p in parts:
        if "depth" in p:
            depth = p.replace("depth", "D")
        elif p in ("A", "M", "CMP", "CO", "SO"):
            qtype = p
        elif p.startswith("form"):
            form = p.replace("form", "F")
        elif "d03" in p:
            dens = "d03"
        elif "d07" in p:
            dens = "d07"
    return f"{depth}_{qtype}_{form}_{dens}"

# -----------------------
# Dataset
# -----------------------
class GridVQAQADataset(torch.utils.data.Dataset):
    def __init__(self, grounding_jsonl, qa_jsonl, images_root):
        self.images_root = Path(images_root)
        self.entries = []

        print(f"[DATA] Mapping images from {grounding_jsonl}...")
        image_map = {}
        with open(grounding_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    img_id = e.get("image", {}).get("id", e.get("image_id"))
                    fname = e.get("image", {}).get("file_name", "")
                    if img_id and fname:
                        image_map[img_id] = fname

        print(f"[DATA] Loading QA from {qa_jsonl}...")
        with open(qa_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    qa = json.loads(line)
                    if qa.get("image_id") in image_map:
                        qa["file_path"] = image_map[qa["image_id"]]
                        self.entries.append(qa)
        print(f"[DATA] Loaded {len(self.entries)} samples.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        rel_path = (e["file_path"].replace("images/", "", 1)
                    if e["file_path"].startswith("images/") else e["file_path"])
        try:
            image = Image.open(self.images_root / rel_path).convert("RGB")
        except Exception:
            return torch.zeros((3, 320, 320)), None, -100, {}

        image = image.resize((320, 320))
        img_t = transforms.functional.pil_to_tensor(image).float() / 255.0
        img_t = transforms.functional.normalize(
            img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        ans = str(e["answer"]).lower()
        return img_t, e["question"], VOCAB.get(ans, -100), e

def collate_fn(batch):
    batch = [b for b in batch if b[1] is not None]
    if not batch:
        return None, None, None, None
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],
        torch.tensor([b[2] for b in batch], dtype=torch.long),
        [b[3] for b in batch],
    )

# -----------------------
# Model Args
# -----------------------
def build_args_for_mdetr(device_str):
    a = SimpleNamespace()
    a.device = device_str
    a.dataset_name = "gridvqa"
    a.backbone = "resnet101"
    a.backbone_name = a.backbone
    a.pretrained_backbone = True
    a.dilation = False
    a.lr_backbone = 0.0
    a.masks = False; a.position_embedding = "sine"
    a.input_image_size = 320; a.hidden_dim = 256; a.enc_layers = 6; a.dec_layers = 6
    a.nheads = 8; a.dim_feedforward = 2048; a.dropout = 0.1; a.activation = "relu"
    a.pre_norm = False; a.normalize_before = False
    a.return_intermediate_dec = True; a.pass_pos_and_query = True
    a.num_queries = 100
    a.do_qa = True; a.qa_dataset = "gqa"; a.split_qa_heads = True
    a.predict_final = False; a.qa_loss_coef = 1.0; a.no_detection = False
    a.set_loss = "hungarian"; a.set_cost_class = 1; a.set_cost_bbox = 5; a.set_cost_giou = 2
    a.eos_coef = 0.1; a.ce_loss_coef = 1; a.bbox_loss_coef = 5; a.giou_loss_coef = 2
    a.aux_loss = True; a.contrastive_loss = True; a.contrastive_align_loss = True
    a.contrastive_loss_hdim = 64; a.contrastive_loss_coef = 0.1; a.contrastive_align_loss_coef = 1
    a.temperature_NCE = 0.07; a.text_encoder_type = "roberta-base"; a.freeze_text_encoder = False
    a.text_encoder_pretrained = True; a.mask_model = "none"; a.mask_loss_coef = 1; a.dice_loss_coef = 1
    return a

# -----------------------
# Decode prediction
# -----------------------
def decode_prediction(out, idx=0):
    pred_type = out['pred_answer_type'][idx].argmax().item()
    if pred_type == 3:
        pred_idx = out['pred_answer_global'][idx].argmax().item()
        answer = INV_VOCAB.get(pred_idx, f"<unk:{pred_idx}>")
    elif pred_type == 0:
        p_idx = out['pred_answer_obj'][idx].argmax().item()
        answer = "yes" if p_idx == 0 else "no"
    else:
        pred_idx = out['pred_answer_global'][idx].argmax().item()
        answer = INV_VOCAB.get(pred_idx, f"<unk:{pred_idx}>")
    return answer, pred_type

# -----------------------
# Save
# -----------------------
def save_results(bucket_stats, total_correct, total_count, output_path):
    results = {
        "overall_accuracy": total_correct / total_count if total_count else 0.0,
        "total_correct": total_correct,
        "total_samples": total_count,
        "buckets": {},
    }
    for bucket in sorted(bucket_stats.keys()):
        s = bucket_stats[bucket]
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        results["buckets"][bucket] = {
            "accuracy": acc,
            "correct": s["correct"],
            "total": s["total"],
        }
    temp_path = str(output_path) + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(results, f, indent=4)
    os.replace(temp_path, str(output_path))

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Bucket-wise accuracy evaluation")
    parser.add_argument("--config", choices=["pure", "spurious", "custom"], default="pure",
                        help="Preset config or 'custom' for manual paths")
    parser.add_argument("--grounding", type=str, help="Grounding JSONL path (custom)")
    parser.add_argument("--qa", type=str, help="QA JSONL path (custom)")
    parser.add_argument("--images", type=str, help="Images root (custom)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path (custom)")
    parser.add_argument("--output", type=str, help="Output JSON path (custom)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_freq", type=int, default=100)
    args = parser.parse_args()

    if args.config == "custom":
        cfg = {
            "grounding": args.grounding,
            "qa": args.qa,
            "images": args.images,
            "checkpoint": args.checkpoint,
            "output": args.output or "eval_results.json",
        }
    else:
        cfg = PRESETS[args.config]
        # Allow CLI overrides
        if args.grounding: cfg["grounding"] = args.grounding
        if args.qa: cfg["qa"] = args.qa
        if args.images: cfg["images"] = args.images
        if args.checkpoint: cfg["checkpoint"] = args.checkpoint
        if args.output: cfg["output"] = args.output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Config       : {args.config}")
    print(f"Grounding    : {cfg['grounding']}")
    print(f"QA           : {cfg['qa']}")
    print(f"Images       : {cfg['images']}")
    print(f"Checkpoint   : {cfg['checkpoint']}")
    print(f"Output       : {cfg['output']}")
    print(f"Batch Size   : {args.batch_size}")
    print("-" * 70)

    # Build & load model
    build_args = build_args_for_mdetr(str(device))
    model, _, _, _, _ = mdetr_module.build(build_args)

    print(f"Loading checkpoint: {cfg['checkpoint']}")
    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # Dataset
    ds = GridVQAQADataset(cfg['grounding'], cfg['qa'], cfg['images'])
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

    # Stats
    bucket_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_count = 0

    print("=" * 70)
    print("Running evaluation …")
    print("=" * 70)

    try:
        pbar = tqdm.tqdm(loader, desc="Eval")
        for batch_i, batch in enumerate(pbar):
            if batch[0] is None:
                continue

            imgs, questions, labels, entries = batch
            imgs = imgs.to(device)

            with torch.no_grad():
                mem = model(imgs, questions, encode_and_save=True)
                out = model(imgs, questions, encode_and_save=False, memory_cache=mem)

            for j in range(len(questions)):
                gt_label = labels[j].item()
                if gt_label == -100:
                    continue

                gt_answer = str(entries[j].get("answer", "")).lower()
                file_path = entries[j].get("file_path", "")
                bucket = get_bucket_from_path(file_path)

                pred_answer, pred_type = decode_prediction(out, idx=j)

                bucket_stats[bucket]["total"] += 1
                total_count += 1

                if str(pred_answer).lower() == gt_answer:
                    bucket_stats[bucket]["correct"] += 1
                    total_correct += 1

            if (batch_i + 1) % args.save_freq == 0:
                save_results(bucket_stats, total_correct, total_count, cfg['output'])
                pbar.set_postfix(acc=f"{total_correct/total_count:.4f}" if total_count else "N/A")

    except KeyboardInterrupt:
        print("\n[!] Interrupted. Saving current results …")
    except Exception as e:
        print(f"\n[!] Error: {e}. Saving current results …")
        raise
    finally:
        save_results(bucket_stats, total_correct, total_count, cfg['output'])
        print(f"\nResults saved to {cfg['output']}")

    # Print summary
    print(f"\nTotal samples : {total_count}")
    print(f"Total correct : {total_correct}")
    print(f"Overall acc   : {total_correct / total_count if total_count else 0:.4f}")
    print()
    print(f"{'Bucket':<30s} {'Total':>7s} {'Correct':>8s} {'Acc':>8s}")
    print("-" * 58)
    for bucket in sorted(bucket_stats.keys()):
        s = bucket_stats[bucket]
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"{bucket:<30s} {s['total']:>7d} {s['correct']:>8d} {acc:>8.4f}")


if __name__ == "__main__":
    main()
