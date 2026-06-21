# Dataset Generation

Procedurally generates the **GridVQA** dataset: a grid of rendered shape objects,
together with grounding boxes and question/answer annotations. Two variants are
provided:

- [`pure/`](pure/) — the base benchmark, no injected shortcut.
- [`spurious/`](spurious/) — the same generation pipeline with a controlled
  **spurious correlation** (a directional / spatial shortcut) baked into how
  targets are placed, so that a shortcut-exploiting model can score well without
  the intended grounding. Used to measure shortcut reliance.

Both variants share the same module structure.

> A pre-generated copy of the dataset is hosted on the Hugging Face Hub at
> [`Aikyam-Lab/gridvqa-dataset`](https://huggingface.co/datasets/Aikyam-Lab/gridvqa-dataset).
> Use the generators below only if you want to (re)generate or customize the data.

## Files

| File | Role |
|------|------|
| `__init__.py` | **Entry point.** Holds the generation config (`GENERATION_CONFIG`, `SPLITS`), drives scene sampling, and writes the JSONL annotations + images. |
| `generator.py` | `DatasetGenerator` — samples scenes (anchor/target placement, density, depth, question type). |
| `renderer.py`  | `SceneRenderer` — renders a scene to a PNG image. |
| `masks.py`     | `MaskGenerator` — ground-truth region/segmentation masks. |
| `templates.py` | `QuestionTemplates` — natural-language question/answer templates per depth & form. |
| `evaluation.py`| Helpers for the spurious-correlation / shortcut reporting. |

## Requirements

From the repo root: `pip install -r ../requirements.txt`. Generation itself only
needs `numpy`, `Pillow`, `tqdm`, and `transformers` (a RoBERTa tokenizer is used
to align answer spans — `roberta-base`, downloaded on first run).

## Usage

Run from inside the chosen variant directory (the modules import each other by
bare name, so the working directory must be that folder):

```bash
# Pure variant
cd pure
python __init__.py --out /path/to/output/GridVQA

# Spurious variant
cd ../spurious
python __init__.py --out /path/to/output/GridVQA_SP
```

### Output

Each run writes, under `--out`:

- `grounding_{split}.jsonl` — image + referring-expression + box annotations.
- `qa_{split}.jsonl` — image + question + answer annotations.
- `images/` — rendered scene PNGs.
- `spurious_correlation_report.csv`, `direction_shortcut_report*.csv` —
  diagnostics quantifying how strong the injected/observed shortcuts are.

### Configuring what gets generated

The number of samples per **depth × question-type × density × form** bucket is
set in `GENERATION_CONFIG` near the top of `__init__.py`. The train/val/test
split is controlled by `SPLITS` (the released configs ship with `{'val': 1.0}`;
edit it to emit train/test splits). Grid size and object density are the `d03`
(density 0.3) / `d07` (density 0.7) keys referenced throughout the config.

These JSONL files are exactly the inputs the [training](../training/) and
[evaluation](../evaluation/) scripts expect via their `--train_jsonl` /
`--val_jsonl` / `--images_root` arguments.
