# Explainability / Analysis

This component wraps a trained MDETR GridVQA model behind a uniform
**analysis-model API** so that multimodal explainability algorithms (DIME,
gradient-based saliency, MultiSHAP, EMAP, LIME, sparse linear encoding, …) can
treat the model as a black/grey box and call it directly — without knowing
anything about MDETR internals. The design follows the
[MultiViz / MultiBench](https://github.com/pliang279/MultiViz) "analysis model"
abstraction.

The explainability **algorithms themselves are not included here** — bring your
own (or any MultiViz-compatible implementation) and point it at the
`GridQAMDETR` analysis model. What this component provides is the model API, the
dataset loader, and the RMA/IoU plausibility metrics for scoring attributions.

## Layout

```
explainability/
├── out_models/
│   ├── analysismodel.py   # abstract base class: the analysis-model interface
│   └── gridqa_mdetr.py    # GridQAMDETR — MDETR wrapped to implement that interface
├── datasets/
│   └── gridqa.py          # GridQADataset — loads the GridVQA analysis (.npz) data
└── metrics/               # RMA / IoU plausibility metrics over attribution outputs
```

## The analysis-model API

`out_models/analysismodel.py` defines the abstract interface; `GridQAMDETR` in
`out_models/gridqa_mdetr.py` implements it. The key methods an explainability
algorithm relies on:

| Method | Purpose |
|--------|---------|
| `getmodalitynames()` / `getmodalitytypes()` | Names/types of the modalities (`image`, `text`). |
| `getunimodaldata(instance, modality)` | Extract one modality's raw data from a data instance. |
| `replaceunimodaldata(instance, modality, newinput)` | Swap one modality's data (used to build perturbations / cross-matrices). |
| `forward(instance)` / `forwardbatch(instances, batch_size)` | Run the model; returns result objects with QA logits (`pred_answer_global`, `pred_answer_obj`, `pred_answer_type`). |
| `getlogit` / `getpredlabel` / `getcorrectlabel` | Read logits / predicted label / ground-truth label. |
| `getprelinear` / `getprelinearsize` | Pre-final-layer features (for sparse linear encoding). |
| `getgrad` / `getgradtext` / `getdoublegrad` | Gradients w.r.t. image / text inputs (for gradient saliency). |
| `get_attention_maps` / `get_detection_boxes` | MDETR-specific introspection helpers. |

Because every algorithm goes through this interface, swapping in a different
model only requires writing a new `analysismodel` subclass.

## Setup

1. **Patched MDETR on the import path.** `GridQAMDETR` does
   `from mdetr.models import mdetr`, so the patched MDETR checkout (see
   [`../mdetr/README.md`](../mdetr/README.md)) must be importable as the `mdetr`
   package. Point `MDETR_ROOT` at it (or add it to `PYTHONPATH`):
   ```bash
   export MDETR_ROOT=/path/to/mdetr
   ```
2. **Checkpoint.** Download a trained checkpoint from the Hugging Face model repo
   [`Aikyam-Lab/gridvqa-models`](https://huggingface.co/Aikyam-Lab/gridvqa-models)
   and point `GRIDVQA_CHECKPOINT` at it:
   ```bash
   export GRIDVQA_CHECKPOINT=/path/to/checkpoint_sp.pth
   ```
3. **Analysis data.** Download the `.npz` analysis split from the Hugging Face
   dataset repo
   [`Aikyam-Lab/gridvqa-dataset`](https://huggingface.co/datasets/Aikyam-Lab/gridvqa-dataset)
   and point `GRIDVQA_DATA_ROOT` at it (expects `<root>/val/depth*/form*/category/grid_config/*.npz`):
   ```bash
   export GRIDVQA_DATA_ROOT=/path/to/d07_pure_data
   ```

## Minimal usage

Run from the `explainability/` root so `out_models.*` / `datasets.*` resolve:

```python
from out_models.gridqa_mdetr import GridQAMDETR
from datasets.gridqa import GridQADataset

dataset = GridQADataset("val")
model = GridQAMDETR(device="cuda", checkpoint_path="/path/to/checkpoint_sp.pth")

instance = dataset.getdata_full(0)
result = model.forward(instance)
print(model.getpredlabel(result), model.getcorrectlabel(instance))

# An explainability algorithm builds perturbations through the same API:
img = model.getunimodaldata(instance, "image")
perturbed = model.replaceunimodaldata(instance, "image", img)  # swap in your own
```

## Scoring attributions

Once an explainability algorithm has produced attribution maps, score them
against ground truth with **RMA** and **IoU** — see
[`metrics/README.md`](metrics/README.md).
