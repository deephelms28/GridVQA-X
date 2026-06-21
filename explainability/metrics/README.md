# Plausibility metrics (RMA / IoU)

Pipeline **step 4**: score attribution maps against the ground-truth regions /
tokens. This code is **algorithm-agnostic** — it does not know or care whether an
attribution came from DIME, gradients, SHAP, or anything else. It only sees
attribution arrays and ground-truth masks.

`metrics.py` provides:

- **RMA (Relevant Mass Accuracy)** — `compute_rma(attribution, gt_mask)`: the
  fraction of total attribution mass that lands inside the ground truth.
- **IoU (Intersection over Union)** — `compute_iou(attribution, gt_mask, threshold)`:
  overlap between the thresholded attribution and the ground-truth mask
  (`--iou-threshold` ∈ `otsu`/`mean`/`median`/`top_k`).

Both functions are shape-agnostic, so they score image heatmaps `(H, W)` and
text token-importance vectors `(T,)` with the same code.

## Input format

Your explainability run (against the `GridQAMDETR` analysis model) writes one
`.npz` per sample into a directory. Each `.npz` contains:

| key | required | meaning |
|-----|----------|---------|
| `attribution` | yes | float array — `(H, W)` for image, `(T,)` for text |
| `image_id` | for `--modality image` | id in `grounding_val.jsonl` (builds the GT box mask) |
| `sample_idx` | alt. to `image_id` | your running index, resolved via `--id-mapping` |
| `text_gt_mask` | for `--modality text` | binary `(T,)` GT rationale over tokens |
| `bucket` | optional | string; if present, results are also grouped by it |

This is the **only** coupling point, and it lives in *your* attribution code:
dump the arrays in this format and the metrics work for any algorithm.

## Usage

```bash
# Image RMA/IoU (ground truth from grounding bounding boxes)
python metrics.py \
    --attributions-dir /path/to/my_attributions \
    --grounding-jsonl  /path/to/GridVQA/grounding_val.jsonl \
    --modality image \
    --iou-threshold otsu \
    --save-json results.json

# Text RMA/IoU (ground truth supplied per-sample as text_gt_mask)
python metrics.py \
    --attributions-dir /path/to/my_text_attributions \
    --grounding-jsonl  /path/to/GridVQA/grounding_val.jsonl \
    --modality text
```

If your `.npz` files carry `sample_idx` instead of `image_id`, pass a
`--id-mapping sample_idx→image_id` JSONL (lines of `{"new_id": ..., "original_id": ...}`).

You can also import the metric functions directly:

```python
from metrics import compute_rma, compute_iou, bboxes_to_mask
rma = compute_rma(attribution_map, gt_mask)
iou = compute_iou(attribution_map, gt_mask, threshold="otsu")
```
