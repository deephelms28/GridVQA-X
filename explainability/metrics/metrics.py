#!/usr/bin/env python3
"""
Algorithm-agnostic plausibility metrics (RMA / IoU) for GridVQA attributions.

This module does NOT know or care which explainability algorithm produced an
attribution map. It takes attribution maps + ground-truth masks and computes:

    RMA (Relevant Mass Accuracy) - fraction of attribution mass inside the GT.
    IoU (Intersection over Union) - overlap of the thresholded attribution and GT.

Both metrics work on any-dimensional arrays, so the same functions score image
heatmaps (H x W) and text token-importance vectors (T,).

----------------------------------------------------------------------------
Generic input format (`--attributions-dir`)
----------------------------------------------------------------------------
One `.npz` file per sample, each containing:

    attribution   : float array. For an image, shape (H, W); for text, shape (T,).
    image_id      : int, optional. The id in `grounding_val.jsonl` used to build
                    the ground-truth image mask from the bounding boxes.
                    (Alternatively provide `sample_idx` + `--id-mapping`.)
    sample_idx    : int, optional. Your algorithm's running index; resolved to an
                    image_id through `--id-mapping` when `image_id` is absent.
    text_gt_mask  : binary array (T,), required for `--modality text`. A 1 marks
                    each token that belongs to the ground-truth rationale. (Text
                    ground truth is question-dependent, so the caller supplies it.)
    bucket        : str, optional. If present, results are also grouped by it.

Produce these `.npz` files from whatever attribution algorithm you run against
the `GridQAMDETR` analysis model - that is the only coupling point, and it lives
in your code, not here.
"""

import json
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ----------------------------------------------------------------------------
# Metrics  (identical for image and text - just different array shapes)
# ----------------------------------------------------------------------------

def compute_rma(attribution: np.ndarray, gt_mask: np.ndarray) -> float:
    """Relevant Mass Accuracy: fraction of |attribution| mass inside gt_mask."""
    attr = np.abs(attribution)
    total = attr.sum()
    if total == 0:
        return 0.0
    return float(((attr / total) * gt_mask).sum())


def compute_iou(attribution: np.ndarray, gt_mask: np.ndarray,
                threshold: str = "otsu") -> float:
    """IoU between a binarised attribution and the binary gt_mask.

    Binarisation strategies: "otsu" (default), "mean", "median", "top_k"
    (top-k entries where k = number of GT-positive entries).
    """
    attr = np.abs(attribution).astype(np.float64)
    if attr.max() == 0:
        return 0.0

    if threshold == "otsu":
        try:
            from skimage.filters import threshold_otsu
            thr = threshold_otsu(attr[attr > 0])
        except (ValueError, ImportError):
            thr = attr.mean()
    elif threshold == "mean":
        thr = attr.mean()
    elif threshold == "median":
        pos = attr[attr > 0]
        thr = np.median(pos) if pos.size else 0.0
    elif threshold == "top_k":
        k = int(gt_mask.sum())
        if k == 0:
            return 0.0
        flat = attr.flatten()
        thr = np.partition(flat, -k)[-k] if k < flat.size else 0.0
    else:
        raise ValueError(f"Unknown threshold strategy: {threshold}")

    pred = (attr >= thr).astype(np.float32)
    gt = gt_mask.astype(np.float32)
    inter = (pred * gt).sum()
    union = ((pred + gt) > 0).astype(np.float32).sum()
    return float(inter / union) if union else 0.0


# ----------------------------------------------------------------------------
# Ground-truth helpers
# ----------------------------------------------------------------------------

def load_grounding_lookup(grounding_jsonl: str) -> dict:
    """image_id -> grounding entry (dict with 'image' and 'annotations')."""
    lookup = {}
    with open(grounding_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            lookup.setdefault(entry["image"]["id"], entry)
    return lookup


def load_id_mapping(id_mapping_jsonl: str) -> dict:
    """sample_idx (new_id) -> image_id (original_id)."""
    mapping = {}
    with open(id_mapping_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            mapping[entry["new_id"]] = entry["original_id"]
    return mapping


def bboxes_to_mask(annotations: list, H: int, W: int) -> np.ndarray:
    """Binary (H, W) mask from COCO-format bboxes [x, y, w, h]."""
    mask = np.zeros((H, W), dtype=np.float32)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x + w), W), min(int(y + h), H)
        mask[y1:y2, x1:x2] = 1.0
    return mask


# ----------------------------------------------------------------------------
# Generic evaluation driver
# ----------------------------------------------------------------------------

def evaluate(attributions_dir: str, grounding_jsonl: str, modality: str,
             iou_threshold: str = "otsu", id_mapping: str = None) -> dict:
    """Score every attribution .npz in `attributions_dir` and aggregate.

    modality : "image" (GT mask built from grounding bboxes) or
               "text"  (GT mask read from each npz's `text_gt_mask`).
    """
    grounding = load_grounding_lookup(grounding_jsonl)
    idmap = load_id_mapping(id_mapping) if id_mapping else None

    records, skipped = [], 0
    for npz_path in sorted(glob.glob(f"{attributions_dir}/*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        attribution = np.asarray(data["attribution"], dtype=np.float64)

        # Resolve the grounding entry for this sample.
        if "image_id" in data:
            image_id = int(data["image_id"])
        elif idmap is not None and "sample_idx" in data:
            image_id = idmap.get(int(data["sample_idx"]))
        else:
            skipped += 1
            continue
        entry = grounding.get(image_id)
        if entry is None or not entry.get("annotations"):
            skipped += 1
            continue

        # Build the ground-truth mask for the requested modality.
        if modality == "image":
            H, W = attribution.shape
            gt_mask = bboxes_to_mask(entry["annotations"], H=H, W=W)
        else:  # text
            if "text_gt_mask" not in data:
                skipped += 1
                continue
            gt_mask = np.asarray(data["text_gt_mask"], dtype=np.float32)

        rec = {
            "image_id": image_id,
            "rma": compute_rma(attribution, gt_mask),
            "iou": compute_iou(attribution, gt_mask, threshold=iou_threshold),
        }
        if "bucket" in data:
            rec["bucket"] = str(data["bucket"])
        records.append(rec)

    return _aggregate(records, skipped)


def _aggregate(records: list, skipped: int) -> dict:
    def agg(recs):
        return {
            "n": len(recs),
            "rma_mean": float(np.mean([r["rma"] for r in recs])),
            "rma_std": float(np.std([r["rma"] for r in recs])),
            "iou_mean": float(np.mean([r["iou"] for r in recs])),
            "iou_std": float(np.std([r["iou"] for r in recs])),
        }

    if not records:
        return {"overall": {}, "per_bucket": {}, "per_sample": [], "skipped": skipped}

    per_bucket = {}
    if any("bucket" in r for r in records):
        buckets = defaultdict(list)
        for r in records:
            buckets[r.get("bucket", "unknown")].append(r)
        per_bucket = {b: agg(rs) for b, rs in sorted(buckets.items())}

    return {
        "overall": agg(records),
        "per_bucket": per_bucket,
        "per_sample": records,
        "skipped": skipped,
    }


def print_results(results: dict) -> None:
    overall = results.get("overall")
    if not overall:
        print("No results to display.")
        return
    rows = [("OVERALL", overall)] + list(results.get("per_bucket", {}).items())
    print(f"\n{'Bucket':<24} {'N':>6} {'RMA':>16} {'IoU':>16}")
    print("-" * 64)
    for label, a in rows:
        rma = f"{a['rma_mean']:.4f}±{a['rma_std']:.4f}"
        iou = f"{a['iou_mean']:.4f}±{a['iou_std']:.4f}"
        print(f"{label:<24} {a['n']:>6} {rma:>16} {iou:>16}")
    print(f"\n(skipped {results.get('skipped', 0)} samples with no GT / no id)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Algorithm-agnostic RMA / IoU for GridVQA attribution maps.")
    p.add_argument("--attributions-dir", required=True,
                   help="Directory of per-sample attribution .npz files "
                        "(see module docstring for the format).")
    p.add_argument("--grounding-jsonl", required=True,
                   help="Path to grounding_val.jsonl (for image ground truth).")
    p.add_argument("--modality", choices=["image", "text"], default="image",
                   help="image: GT from grounding bboxes; text: GT from each "
                        "npz's text_gt_mask.")
    p.add_argument("--iou-threshold", default="otsu",
                   choices=["otsu", "mean", "median", "top_k"])
    p.add_argument("--id-mapping", default=None,
                   help="Optional sample_idx -> image_id JSONL, if your npz files "
                        "carry sample_idx instead of image_id.")
    p.add_argument("--save-json", default=None,
                   help="If set, write the full results to this JSON path.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate(
        attributions_dir=args.attributions_dir,
        grounding_jsonl=args.grounding_jsonl,
        modality=args.modality,
        iou_threshold=args.iou_threshold,
        id_mapping=args.id_mapping,
    )
    print_results(results)
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[info] Results saved to {args.save_json}")
