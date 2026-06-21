# Model Evaluation

Pipeline **step 2**: run forward passes of a trained MDETR checkpoint over a
GridVQA split to get the model's answers and compute accuracy.

A single script, `eval_accuracy.py`, does this: it loads a checkpoint, decodes
each sample's predicted answer, and reports **overall** and **bucket-wise**
accuracy (by depth / question-type / form / density).

## Prerequisites

The script `imports` from the MDETR package (`from models import mdetr`), so it
must be run from inside a **patched MDETR checkout** (see
[`../mdetr/README.md`](../mdetr/README.md)). Copy it into the checkout root:

```bash
cp eval_accuracy.py /path/to/mdetr/
```

> Trained GridVQA checkpoints are on the Hugging Face Hub at
> [`Aikyam-Lab/gridvqa-models`](https://huggingface.co/Aikyam-Lab/gridvqa-models);
> the dataset is at
> [`Aikyam-Lab/gridvqa-dataset`](https://huggingface.co/datasets/Aikyam-Lab/gridvqa-dataset).

## Usage

`eval_accuracy.py` has `pure` / `spurious` presets (edit the path constants near
the top of the file) and a `custom` mode that takes paths on the command line:

```bash
# Preset (after setting PURE_BASE / SPUR_BASE / CKPT_DIR at the top of the file)
python eval_accuracy.py --config pure

# Fully explicit
python eval_accuracy.py --config custom \
    --grounding  /data/GridVQA/grounding_test.jsonl \
    --qa         /data/GridVQA/qa_test.jsonl \
    --images     /data/GridVQA/images \
    --checkpoint ./checkpoints/checkpoint_pure.pth \
    --output     pure_accuracy_results.json \
    --batch_size 256
```

It writes a JSON report (overall + per-bucket accuracy), saving incrementally
every `--save_freq` batches so a long run can be resumed. Run with `--help` for
the full argument list.

To evaluate on the **spurious** variant, use `--config spurious` (or point the
custom paths at the spurious dataset / checkpoint).
