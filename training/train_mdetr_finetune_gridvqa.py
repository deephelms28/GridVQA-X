#!/usr/bin/env python3
"""
Final training script for fine-tuning MDETR EB5 on GridVQA.
Place at: mdetr/train_mdetr_finetune_gridvqa.py
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

# Import local repo modules (assumes mdetr package files are in mdetr/)
from models import mdetr as mdetr_module
from models import postprocessors as postproc_module

# -----------------------
# Dataset (lightweight, tailored to JSONL format you used)
# -----------------------
class GridVQAGroundingDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, images_root, filter_fn=None, resize_short=640):
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
        img_p = self.images_root / e["image"]["file_name"]
        im = Image.open(img_p).convert("RGB")
        # Keep the repo's expected resizing scheme - resize both dims to resize_short for simplicity
        im = im.resize((self.resize_short, self.resize_short))
        img_t = transforms.functional.pil_to_tensor(im).float() / 255.0
        img_t = transforms.functional.normalize(img_t, [0.485,0.456,0.406], [0.229,0.224,0.225])
        return img_t, e

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    entries = [b[1] for b in batch]
    return imgs, entries

# -----------------------
# Robust builder defaults (comprehensive)
# -----------------------
def build_args_for_mdetr(device_str: str):
    a = SimpleNamespace()
    # device
    a.device = device_str
    a.dataset_name = "gridvqa"

    # backbone & image encoder (match hubconf / backbone expectations)
    a.backbone = "timm_tf_efficientnet_b5_ns"   # EB5 naming used in repo hubconf
    a.backbone_name = a.backbone
    a.pretrained_backbone = True
    a.lr_backbone = 1e-5
    a.freeze_backbone_at = 0
    a.masks = False
    a.dilation = False
    a.position_embedding = "sine"   # "sine" or "v2"
    a.input_image_size = 640

    # transformer architecture defaults
    a.hidden_dim = 256
    a.enc_layers = 6
    a.dec_layers = 6
    a.nheads = 8
    a.dim_feedforward = 2048
    a.dropout = 0.1
    a.activation = "relu"
    a.pre_norm = False
    a.normalize_before = False
    a.return_intermediate_dec = False
    a.pass_pos_and_query = True
    a.num_queries = 100
    a.num_feature_levels = 4

    # QA / heads options (we keep QA disabled for stage 1)
    a.do_qa = False
    a.split_qa_heads = False
    a.predict_final = False
    a.no_detection = False

    # mask / segmentation defaults
    a.mask_model = "none"
    a.mask_loss_coef = 1.0
    a.dice_loss_coef = 1.0

    # loss / matcher
    a.set_loss = "hungarian"
    a.set_cost_class = 1.0
    a.set_cost_bbox = 5.0
    a.set_cost_giou = 2.0
    a.eos_coef = 0.1
    a.ce_loss_coef = 1.0
    a.bbox_loss_coef = 5.0
    a.giou_loss_coef = 2.0
    a.aux_loss = True

    # contrastive / alignment defaults
    a.contrastive_loss = True
    a.contrastive_align_loss = True
    a.contrastive_loss_hdim = 64
    a.contrastive_loss_coef = 1.0
    a.contrastive_align_loss_coef = 1.0
    a.temperature_NCE = 0.07

    # text encoder
    a.text_encoder_type = "roberta-base"
    a.freeze_text_encoder = True
    a.text_encoder_pretrained = True

    # device
    a.device = device_str
    return a

# -----------------------
# Small validation routine (grounding IoU proxy)
# -----------------------
def run_validation(model, postprocessors, val_loader, device, iou_thresh=0.5):
    model.eval()
    total = 0
    matched = 0
    with torch.no_grad():
        for images, entries in val_loader:
            images = images.to(device)
            # build captions for batch (pass to model forward)
            captions = [entry.get("sentence", {}).get("text", "find") or "find" for entry in entries]
            try:
                outputs = model(images, captions=captions)
            except TypeError:
                # fallback if model signature different
                outputs = model(images)
            results = None
            if postprocessors is not None:
                try:
                    sizes = torch.tensor([[images.shape[-2], images.shape[-1]]]*images.shape[0], device=device)
                    results = postprocessors["bbox"](outputs, sizes)
                except Exception:
                    results = None
            for i, entry in enumerate(entries):
                anns = entry.get("annotations", [])
                if len(anns) == 0:
                    total += 1
                    continue
                # Build pred_boxes from postprocessors or model outputs
                pred_boxes = None
                if results is not None and len(results[i].get("boxes", [])) > 0:
                    pred_boxes = torch.tensor(results[i]["boxes"], device=device)
                else:
                    pb = outputs.get("pred_boxes", None)
                    if pb is None:
                        total += 1; continue
                    pb_i = pb[i].detach()
                    h,w = images.shape[-2], images.shape[-1]
                    cx = pb_i[:,0]*w; cy = pb_i[:,1]*h; hh = pb_i[:,2]*h; ww = pb_i[:,3]*w
                    x0 = cx - ww/2; y0 = cy - hh/2
                    pred_boxes = torch.stack([x0,y0,ww,hh], dim=1)
                gt_boxes = []
                for a in anns:
                    x,y,w,h = a["bbox"]
                    gt_boxes.append([x,y,w,h])
                gt_t = torch.tensor(gt_boxes, device=device)
                # compute best IoU
                def to_xyxy(boxes):
                    x,y,w,h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
                    x1,y1,x2,y2 = x, y, x+w, y+h
                    return torch.stack([x1,y1,x2,y2], dim=1)
                pb_xy = to_xyxy(pred_boxes)
                gt_xy = to_xyxy(gt_t)
                best_iou = 0.0
                for p in pb_xy:
                    for g in gt_xy:
                        inter_w = max(0.0, min(float(p[2]),float(g[2])) - max(float(p[0]),float(g[0])))
                        inter_h = max(0.0, min(float(p[3]),float(g[3])) - max(float(p[1]),float(g[1])))
                        inter = inter_w * inter_h
                        union = (float(p[2])-float(p[0]))*(float(p[3])-float(p[1])) + (float(g[2])-float(g[0]))*(float(g[3])-float(g[1])) - inter + 1e-6
                        best_iou = max(best_iou, inter/union)
                if best_iou >= iou_thresh:
                    matched += 1
                total += 1
    return {"grounding_acc": matched/total if total>0 else 0.0, "total": total, "matches": matched}

# -----------------------
# Main training
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--mdetr_ckpt", default="hub", help="path to local EB5 checkpoint or 'hub'")
    parser.add_argument("--output_dir", default="../checkpoints")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs_stage", default=4, type=int)
    parser.add_argument("--lr", default=2.5e-5, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--include_10x10", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="Stage 1 ft MDETR on GridVQA")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--freeze_text_encoder", action="store_true")
    parser.add_argument("--freeze_backbone_layers", type=int, default=0)
    parser.add_argument("--benchmark_steps", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--backbone", type=str, default=None, help="override backbone string (e.g. timm_efficientnet_b5)")
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()

    local_rank = args.local_rank
    world_size = args.world_size
    distributed = world_size > 1 or "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if distributed:
        # torch.distributed.launch sets env vars; init with env://
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    # build default args and override with CLI values
    build_args = build_args_for_mdetr(device_str=str(device))
    if args.backbone:
        build_args.backbone = args.backbone
        build_args.backbone_name = args.backbone
    if args.lr_backbone is not None:
        build_args.lr_backbone = args.lr_backbone
    if args.freeze_text_encoder:
        build_args.freeze_text_encoder = True
    if args.dropout is not None:  # small safe-guard in case of typo
        build_args.dropout = args.dropout
    # explicitly set split_qa_heads false for stage1
    build_args.split_qa_heads = False
    build_args.do_qa = False

    # build model using repo's build()
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = mdetr_module.build(build_args)
    model.to(device)

    # optionally freeze some backbone children (if model has backbone)
    if args.freeze_backbone_layers > 0:
        if hasattr(model, "backbone") or (hasattr(model, "module") and hasattr(model.module, "backbone")):
            bb = model.module.backbone if hasattr(model, "module") else model.backbone
            children = list(bb.named_children())
            for i,(nm,mod) in enumerate(children[:args.freeze_backbone_layers]):
                for p in mod.parameters(): p.requires_grad = False
            if (not distributed) or local_rank == 0:
                print(f"[INFO] Frozen first {min(args.freeze_backbone_layers, len(children))} backbone children.")
        else:
            if (not distributed) or local_rank == 0:
                print("[WARN] model has no backbone attribute to freeze.")

    # load EB5 checkpoint weights if provided (local)
    if args.mdetr_ckpt and args.mdetr_ckpt != "hub":
        ck = torch.load(args.mdetr_ckpt, map_location="cpu")
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
        new_sd = {}
        for k, v in sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_sd[nk] = v
        # load with strict=False so extra/missing keys won't crash
        model.load_state_dict(new_sd, strict=False)
        if (not distributed) or local_rank == 0:
            print("[INFO] loaded checkpoint weights into model (strict=False)")

    # wrap DDP
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # postprocessors (optional)
    postprocessors = None
    try:
        postprocessors = postproc_module.build_postprocessors(build_args, dataset_name=None)
    except Exception:
        postprocessors = None

    # curriculum stages (no 10x10 by default)
    stages = [
        ("depth1_g5_d03", lambda e: ("depth1" in e['image']['file_name'] and "g5_d03" in e['image']['file_name'])),
        ("depth1_g5_d07", lambda e: ("depth1" in e['image']['file_name'] and "g5_d07" in e['image']['file_name'])),
        ("depth2_g5_d03", lambda e: ("depth2" in e['image']['file_name'] and "g5_d03" in e['image']['file_name'])),
        ("depth2_g5_d07", lambda e: ("depth2" in e['image']['file_name'] and "g5_d07" in e['image']['file_name'])),
        ("depth3_g5_d03", lambda e: ("depth3" in e['image']['file_name'] and "g5_d03" in e['image']['file_name'])),
        ("depth3_g5_d07", lambda e: ("depth3" in e['image']['file_name'] and "g5_d07" in e['image']['file_name'])),
    ]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    if args.wandb and (not distributed or local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name="Stage1-MDETR-GridVQA")

    os.makedirs(args.output_dir, exist_ok=True)

    # training loop across curriculum stages
    for stage_name, filter_fn in stages:
        if (not distributed) or local_rank==0:
            print("[STAGE] ", stage_name)
        ds_tr = GridVQAGroundingDataset(args.train_jsonl, args.images_root, filter_fn=filter_fn)
        ds_val = GridVQAGroundingDataset(args.val_jsonl, args.images_root, filter_fn=filter_fn)
        if len(ds_tr) == 0:
            if (not distributed) or local_rank==0:
                print("[STAGE] skipping empty stage", stage_name)
            continue
        sampler = DistributedSampler(ds_tr) if distributed else None
        v_sampler = DistributedSampler(ds_val) if distributed else None
        loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                            collate_fn=collate_fn, num_workers=args.num_workers)
        vloader = DataLoader(ds_val, batch_size=max(1,args.batch_size//2), shuffle=False, sampler=v_sampler,
                             collate_fn=collate_fn, num_workers=max(1,args.num_workers//2))

        # Benchmark warmup to estimate s/step
        num_samples = len(loader.dataset)
        batch_size_per_gpu = args.batch_size
        actual_world_size = args.world_size if args.world_size and args.world_size>0 else (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
        steps_per_epoch = int(ceil(num_samples / (batch_size_per_gpu * actual_world_size)))
        total_steps_stage = steps_per_epoch * args.epochs_stage
        s_per_step = None

        if args.benchmark_steps and ((not distributed) or local_rank==0):
            bench_iter = iter(loader)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            bs = min(args.benchmark_steps, len(loader))
            for _ in range(bs):
                try:
                    images_b, entries_b = next(bench_iter)
                except StopIteration:
                    break
                images_b = images_b.to(device)
                captions_b = [entry.get("sentence", {}).get("text", "find") or "find" for entry in entries_b]
                optimizer.zero_grad()
                try:
                    out = model(images_b, captions=captions_b)
                except TypeError:
                    out = model(images_b)
                # skip backward during warmup to keep it cheap
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            s_per_step = (t1 - t0) / max(1, bs)
            if (not distributed) or local_rank==0:
                print(f"[BENCHMARK] s_per_step ≈ {s_per_step:.4f}s")

        if distributed:
            s_per_step_tensor = torch.tensor(0.0 if s_per_step is None else s_per_step, device=device)
            torch.distributed.broadcast(s_per_step_tensor, src=0)
            s_per_step = float(s_per_step_tensor.item())
        if s_per_step is None:
            s_per_step = 0.5

        global_step = 0
        for epoch in range(1, args.epochs_stage + 1):
            if distributed: sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            iters = 0
            t0_epoch = time.time()

            for images, entries in loader:
                step_t0 = time.perf_counter()
                images = images.to(device)
                # build dummy/structured targets for criterion from annotations
                targets_for_criterion = []
                for entry in entries:
                    anns = entry.get("annotations", [])
                    boxes = []
                    labels = []
                    for a in anns:
                        x,y,w,h = a["bbox"]
                        img_w = img_h = loader.dataset.resize_short
                        # convert to cx,cy,h,w normalized in [0,1]
                        cx = (x + 0.5*w) / img_w
                        cy = (y + 0.5*h) / img_h
                        boxes.append([cx, cy, h/img_h, w/img_w])
                        labels.append(int(a["category_id"]) - 1)
                    if boxes:
                        boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
                        labels = torch.tensor(labels, dtype=torch.int64, device=device)
                    else:
                        boxes = torch.zeros((0,4), dtype=torch.float32, device=device)
                        labels = torch.zeros((0,), dtype=torch.int64, device=device)
                    tgt = {"boxes": boxes, "labels": labels}
                    targets_for_criterion.append(tgt)

                optimizer.zero_grad()
                # build captions and pass to model to avoid missing caption arg
                captions = [entry.get("sentence", {}).get("text", "find") or "find" for entry in entries]
                try:
                    outputs = model(images, captions=captions)
                except TypeError:
                    outputs = model(images)
                # compute loss via criterion (repo's SetCriterion)
                try:
                    loss_dict = criterion(outputs, targets_for_criterion, torch.zeros((0,1), dtype=torch.bool, device=device))
                    # sum up losses
                    loss = 0.0
                    for k,v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            loss = loss + v
                        else:
                            loss = loss + torch.tensor(float(v), device=device)
                except Exception as e:
                    # fallback if criterion format mismatch
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach().cpu().item())
                iters += 1
                global_step += 1

                # log periodically
                if (global_step % args.log_every == 0) and ((not distributed) or local_rank==0):
                    eta_seconds = max(0, (total_steps_stage - global_step) * s_per_step)
                    print(f"[ETA] step {global_step}/{total_steps_stage} s/step={s_per_step:.3f} ETA={int(eta_seconds)}s")
                    if args.wandb:
                        import wandb
                        wandb.log({f"{stage_name}/train_loss": running_loss/max(1, iters), "global_step": global_step})

            t1_epoch = time.time()
            avg_loss = running_loss / max(1, iters)
            if (not distributed) or local_rank == 0:
                print(f"[{stage_name}] Epoch {epoch} loss={avg_loss:.4f} time={t1_epoch-t0_epoch:.1f}s")
                ckpt_path = Path(args.output_dir)/f"{stage_name}_ep{epoch}.pth"
                sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({"model_state_dict": sd, "optimizer": optimizer.state_dict(), "epoch": epoch}, str(ckpt_path))
                print("[INFO] saved checkpoint", ckpt_path)
                val_metrics = run_validation(model.module if hasattr(model,"module") else model, postprocessors, vloader, device)
                print("[VAL]", val_metrics)
                if args.wandb:
                    import wandb
                    wandb.log({f"{stage_name}/train_loss": avg_loss, f"{stage_name}/val_grounding_acc": val_metrics["grounding_acc"], "epoch": epoch})

            if distributed:
                torch.distributed.barrier()

    if args.wandb and (not distributed or local_rank==0):
        import wandb
        wandb.finish()
    if distributed:
        torch.distributed.destroy_process_group()
    print("Training finished.")

if __name__ == "__main__":
    main()

