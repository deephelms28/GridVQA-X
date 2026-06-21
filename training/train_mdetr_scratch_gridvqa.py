#!/usr/bin/env python3
"""
Final training script for training MDETR from scratch on GridVQA.
- Uses ResNet-101 backbone for stability.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from types import SimpleNamespace
from math import ceil

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image

# Import local repo modules
from models import mdetr as mdetr_module
from models.postprocessors import build_postprocessors

# -----------------------
# Dataset
# -----------------------
class GridVQAGroundingDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, images_root, filter_fn=None, resize_short=320):
        self.jsonl_path = Path(jsonl_path)
        assert self.jsonl_path.exists(), f"{jsonl_path} missing"
        self.images_root = Path(images_root)
        self.filter_fn = filter_fn
        self.resize_short = resize_short
        self.entries = []
        with open(self.jsonl_path, "r") as f:
            for ln in f:
                if not ln.strip():
                    continue
                e = json.loads(ln)
                if (self.filter_fn is None) or self.filter_fn(e):
                    self.entries.append(e)
        print(f"[DATA] Loaded {len(self.entries)} examples from {self.jsonl_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        relative_img_path = e["image"]["file_name"].replace("images/", "")
        img_p = self.images_root / relative_img_path
        im = Image.open(img_p).convert("RGB")
        im = im.resize((self.resize_short, self.resize_short))
        img_t = transforms.functional.pil_to_tensor(im).float() / 255.0
        img_t = transforms.functional.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img_t, e

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    entries = [b[1] for b in batch]
    return imgs, entries

# -----------------------
# Model Builder Arguments
# -----------------------
def build_args_for_mdetr(device_str: str):
    a = SimpleNamespace()
    a.device = device_str
    a.dataset_name = "gridvqa"

    # [FIX] Switched to resnet101 for stability as per GitHub issue
    a.backbone = "resnet101"
    a.backbone_name = a.backbone
    a.dilation = False # Required for resnet
    
    a.pretrained_backbone = True
    a.lr_backbone = 2e-5
    a.masks = False
    a.position_embedding = "sine"
    a.input_image_size = 320

    # Transformer architecture
    a.hidden_dim = 256
    a.enc_layers = 6
    a.dec_layers = 6
    a.nheads = 8
    a.dim_feedforward = 2048
    a.dropout = 0.1
    a.activation = "relu"
    a.pre_norm = False
    a.normalize_before = False
    a.return_intermediate_dec = True
    a.pass_pos_and_query = True
    a.num_queries = 100
    
    # QA / heads options (disabled)
    a.do_qa = False
    a.split_qa_heads = False
    a.predict_final = False
    a.no_detection = False

    # Loss / matcher
    a.set_loss = "hungarian"
    a.set_cost_class = 1.0
    a.set_cost_bbox = 5.0
    a.set_cost_giou = 2.0
    a.eos_coef = 0.1
    a.ce_loss_coef = 1.0
    a.bbox_loss_coef = 5.0
    a.giou_loss_coef = 2.0
    a.aux_loss = True # Re-enable aux loss, should be stable with ResNet

    # Contrastive
    a.contrastive_loss = True
    a.contrastive_align_loss = True
    a.contrastive_loss_hdim = 64
    a.contrastive_loss_coef = 0.1
    a.contrastive_align_loss_coef = 1.0
    a.temperature_NCE = 0.07

    # Text encoder
    a.text_encoder_type = "roberta-base"
    a.freeze_text_encoder = False
    a.text_encoder_pretrained = True

    # mask / segmentation defaults
    a.mask_model = "none"
    a.mask_loss_coef = 1.0
    a.dice_loss_coef = 1.0

    return a

# -----------------------
# Validation Routine
# -----------------------
def run_validation(model, postprocessor, val_loader, device, iou_thresh=0.5):
    model.eval()
    total = 0
    matched = 0
    with torch.no_grad():
        for images, entries in val_loader:
            images = images.to(device)
            captions = [entry.get("sentence", {}).get("text", "") for entry in entries]
            
            memory_cache = model(images, captions, encode_and_save=True)
            outputs = model(images, captions, encode_and_save=False, memory_cache=memory_cache)
            
            orig_target_sizes = torch.stack([torch.tensor([320, 320], device=device) for _ in range(len(images))])
            results = postprocessor(outputs, orig_target_sizes)
            
            for i, entry in enumerate(entries):
                anns = entry.get("annotations", [])
                if not anns: continue

                pred_boxes = results[i]['boxes']
                gt_boxes = torch.tensor([a["bbox"] for a in anns], device=device)

                # Convert to xyxy for IoU
                pred_xyxy = pred_boxes
                gt_xyxy = torch.stack([gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]], dim=1)

                best_iou = 0.0
                if len(pred_xyxy) > 0 and len(gt_xyxy) > 0:
                    for p in pred_xyxy:
                        for g in gt_xyxy:
                            inter_w = torch.max(torch.tensor(0.0), torch.min(p[2], g[2]) - torch.max(p[0], g[0]))
                            inter_h = torch.max(torch.tensor(0.0), torch.min(p[3], g[3]) - torch.max(p[1], g[1]))
                            inter = inter_w * inter_h
                            union = (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-6
                            best_iou = max(best_iou, inter / union)
                
                if best_iou >= iou_thresh:
                    matched += 1
                total += 1

    return {"grounding_acc": matched / total if total > 0 else 0.0, "total": total, "matches": matched}

# -----------------------
# Main Training
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_dir", default="./checkpoints_scratch")
    parser.add_argument("--resume_from", help="Path to checkpoint to resume training from")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs_stage", default=10, type=int, help="Epochs per curriculum stage")
    parser.add_argument("--lr", default=1e-4, type=float, help="LR for transformer")
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument("--lr_text_encoder", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MDETR-GridVQA-Scratch-ResNet")
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()

    # DDP Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    # Model Build
    build_args = build_args_for_mdetr(device_str=str(device))
    build_args.num_classes = 256
    model, criterion, _, _, weight_dict = mdetr_module.build(build_args)
    model.to(device)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and "text_encoder" not in n and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": args.lr_text_encoder},
    ]
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

    # --- NEW: Checkpoint Loading Logic ---
    start_epoch = 1
    if args.resume_from:
        if local_rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    postprocessors = build_postprocessors(build_args, dataset_name="coco")

    # Curriculum Stages
    stages = [
        ("depth1_g5_d03", lambda e: ("depth1" in e['image']['file_name'] and "g5_d03" in e['image']['file_name'])),
        ("depth1_g5_d07", lambda e: ("depth1" in e['image']['file_name'] and "g5_d07" in e['image']['file_name'])),
        ("depth2_g5_d03", lambda e: ("depth2" in e['image']['file_name'] and "g5_d03" in e['image']['file_name'])),
        ("depth2_g5_d07", lambda e: ("depth2" in e['image']['file_name'] and "g5_d07" in e['image']['file_name'])),
        ("depth3_g5_d03", lambda e: ("depth3" in e['image']['file_name'] and "g5_d03" in e['image']['file_name'])),
        ("depth3_g5_d07", lambda e: ("depth3" in e['image']['file_name'] and "g5_d07" in e['image']['file_name'])),
    ]
    
    if args.wandb and local_rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, config=args, resume="allow")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training Loop
    for stage_name, filter_fn in stages:
        if local_rank == 0:
            print(f"\n{'='*20} STARTING STAGE: {stage_name} {'='*20}")
        
        ds_tr = GridVQAGroundingDataset(args.train_jsonl, args.images_root, filter_fn=filter_fn)
        ds_val = GridVQAGroundingDataset(args.val_jsonl, args.images_root, filter_fn=filter_fn)
        
        if len(ds_tr) == 0:
            if local_rank == 0: print(f"[STAGE] Skipping empty stage {stage_name}")
            continue

        sampler_tr = DistributedSampler(ds_tr) if distributed else None
        sampler_val = DistributedSampler(ds_val, shuffle=False) if distributed else None
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=(sampler_tr is None), sampler=sampler_tr, collate_fn=collate_fn, num_workers=args.num_workers)
        loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, sampler=sampler_val, collate_fn=collate_fn, num_workers=2)

        for epoch in range(start_epoch, args.epochs_stage + 1):
            if distributed: sampler_tr.set_epoch(epoch)
            model.train()
            criterion.train()
            
            for i, (images, entries) in enumerate(loader_tr):
                images = images.to(device)
                captions = [entry.get("sentence", {}).get("text", "") for entry in entries]
                
                memory_cache = model(images, captions, encode_and_save=True)
                outputs = model(images, captions, encode_and_save=False, memory_cache=memory_cache)

                targets = []
                positive_maps_for_batch = []
                tokenized = outputs["tokenized"]
                FIXED_TOKEN_LEN = 256

                for j, entry in enumerate(entries):
                    anns = entry.get("annotations", [])
                    id_to_char_span = {} # ... (character span logic)
                    id_to_span = {} # ... (token span logic)
                    # (This block is long, keeping it collapsed for brevity, no changes here)
                    sentence_text = entry.get("sentence", {}).get("text", "")
                    words = entry.get("sentence", {}).get("words", [])
                    word_char_starts = []
                    current_pos = 0
                    if sentence_text and words:
                        for word in words:
                            try:
                                start = sentence_text.index(word, current_pos)
                                word_char_starts.append(start)
                                current_pos = start + len(word)
                            except ValueError:
                                word_char_starts.append(-1)
                    
                    id_to_char_span = {}
                    for span_info in entry.get("sentence", {}).get("spans", []):
                        word_level_span = span_info.get("span")
                        if word_level_span and word_char_starts:
                            start_word_idx, end_word_idx = word_level_span
                            if 0 <= start_word_idx < len(word_char_starts) and 0 <= end_word_idx < len(words):
                                char_start = word_char_starts[start_word_idx]
                                end_word = words[end_word_idx]
                                char_end = word_char_starts[end_word_idx] + len(end_word)
                                if char_start != -1:
                                    for box_id in span_info.get("box_ids", []):
                                        id_to_char_span[box_id] = (char_start, char_end)

                    id_to_span = {}
                    for span_info in entry.get("sentence", {}).get("spans", []):
                        for box_id in span_info.get("box_ids", []):
                            id_to_span[box_id] = span_info
                    
                    gt_boxes, gt_labels, tokens_positive = [], [], []
                    num_tokens = tokenized.attention_mask[j].sum()

                    for ann in anns:
                        x, y, w, h = ann['bbox']
                        img_h, img_w = images.shape[-2:]
                        gt_boxes.append([(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h])
                        gt_labels.append(int(ann["category_id"]) - 1)
                        
                        char_span = id_to_char_span.get(ann["id"])
                        tokens_positive.append([char_span] if char_span else [])
                        
                        token_span = id_to_span.get(ann["id"], {}).get("token_span")
                        box_pos_map = torch.zeros(num_tokens, device=device)
                        if token_span and token_span[1] <= num_tokens:
                            indices = torch.arange(num_tokens, device=device)
                            mask = (indices >= token_span[0]) & (indices < token_span[1])
                            box_pos_map = mask.float()
                        
                        if box_pos_map.sum() > 0: box_pos_map = box_pos_map / box_pos_map.sum()
                        
                        padded_map = torch.nn.functional.pad(box_pos_map, (0, FIXED_TOKEN_LEN - num_tokens.item()))
                        positive_maps_for_batch.append(padded_map)
                    
                    targets.append({
                        "boxes": torch.tensor(gt_boxes, device=device) if gt_boxes else torch.zeros((0,4), device=device),
                        "labels": torch.tensor(gt_labels, device=device) if gt_labels else torch.zeros((0,), device=device),
                        "tokens_positive": tokens_positive
                    })
                
                positive_map_cat = torch.stack(positive_maps_for_batch, dim=0) if positive_maps_for_batch else torch.empty(0, FIXED_TOKEN_LEN, device=device)

                optimizer.zero_grad()
                loss_dict = criterion(outputs, targets, positive_map_cat)
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                if not torch.isfinite(loss):
                    print(f"WARNING: non-finite loss, skipping update. Loss: {loss.item()}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                if (i % args.log_every == 0) and (local_rank == 0):
                    print(f"[Train] Stage: {stage_name} Epoch: {epoch} Step: {i}/{len(loader_tr)} Loss: {loss.item():.4f}")
                    if args.wandb: wandb.log({f"{stage_name}/train_loss_step": loss.item()})
            
            if local_rank == 0:
                # val_metrics = run_validation(model.module if distributed else model, postprocessors['bbox'], loader_val, device)
                # print(f"[VAL] Stage: {stage_name} Epoch: {epoch} Metrics: {val_metrics}")
                # if args.wandb: wandb.log({f"{stage_name}/val_grounding_acc": val_metrics["grounding_acc"], "epoch": epoch})

                ckpt_path = Path(args.output_dir) / f"{stage_name}_ep{epoch}.pth"
                sd = model.module.state_dict() if distributed else model.state_dict()
                torch.save({"model_state_dict": sd, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "stage": stage_name}, str(ckpt_path))
                print(f"[INFO] Saved checkpoint to {ckpt_path}")

            if distributed: torch.distributed.barrier()
        
        # Reset start_epoch for the next stage in the curriculum
        start_epoch = 1

    if args.wandb and local_rank == 0: wandb.finish()
    if distributed: torch.distributed.destroy_process_group()
    print("Training finished.")

if __name__ == "__main__":
    main()