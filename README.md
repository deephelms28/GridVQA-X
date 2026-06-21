# GridVQA

Code release for the **GridVQA** synthetic grounded visual-question-answering
benchmark and the experiments fine-tuning [MDETR](https://github.com/ashkamath/mdetr)
on it. The benchmark is built in two variants:

- **Pure** — questions whose answers depend only on the intended visual reasoning.
- **Spurious** — the same task with a controlled spurious correlation injected
  (e.g. a directional/spatial shortcut), used to probe whether models exploit
  shortcuts instead of the intended grounding.

This repository contains three self-contained components:

| Component | Folder | What it does |
|-----------|--------|--------------|
| Dataset generation | [`dataset_generation/`](dataset_generation/) | Procedurally renders grid scenes and generates grounding + QA annotations, for both the **pure** and **spurious** variants. |
| Model training | [`training/`](training/) | Fine-tunes (or trains from scratch) an MDETR EfficientNet-B5 model on GridVQA. |
| Model evaluation | [`evaluation/`](evaluation/) | Runs forward passes of a trained MDETR checkpoint over a split and computes overall + bucket-wise accuracy. |
| Explainability | [`explainability/`](explainability/) | Wraps a trained model behind a uniform analysis-model API so external multimodal explainability algorithms can call it directly, plus RMA/IoU plausibility metrics. (The algorithms themselves are not bundled.) |

## Pretrained models & data

Trained checkpoints and the dataset are hosted on the Hugging Face Hub:

- **Models:** [`Aikyam-Lab/gridvqa-models`](https://huggingface.co/Aikyam-Lab/gridvqa-models)
- **Dataset:** [`Aikyam-Lab/gridvqa-dataset`](https://huggingface.co/datasets/Aikyam-Lab/gridvqa-dataset)

The training and evaluation scripts build on a **fork of MDETR**. Because MDETR
is an external repository with its own license, we do **not** redistribute it
here. Instead, [`mdetr/`](mdetr/) contains a patch and instructions to reproduce
our fork from a pinned upstream commit. See [`mdetr/README.md`](mdetr/README.md).

## Repository layout

```
GridVQA_Release/
├── dataset_generation/      # Component 1: pure + spurious dataset generators
│   ├── pure/
│   └── spurious/
├── training/                # Component 2: MDETR fine-tune / scratch training
├── evaluation/              # Component 3: model forward passes -> accuracy
├── explainability/          # Component 4: analysis-model API + RMA/IoU metrics
├── mdetr/                   # MDETR fork patch + upstream attribution
├── requirements.txt         # Python dependencies for the MDETR pipeline
├── LICENSE                  # MIT (this work) + Apache-2.0 attribution for MDETR
└── .python-version
```

## Quick start

1. **Generate data** — see [`dataset_generation/README.md`](dataset_generation/README.md).
   Produces `grounding_{train,val,test}.jsonl`, `qa_{train,val,test}.jsonl`
   and the rendered images.
2. **Set up MDETR** — clone the upstream MDETR repo and apply our patch as
   described in [`mdetr/README.md`](mdetr/README.md), then drop the training and
   evaluation scripts into it.
3. **Train** — see [`training/README.md`](training/README.md).
4. **Evaluate** — see [`evaluation/README.md`](evaluation/README.md).

## Environment

- **Python:** 3.10+.
- **Core stack:** PyTorch 2.x, torchvision, `transformers`, and the MDETR
  dependencies listed in [`requirements.txt`](requirements.txt).

> **Note on `transformers` versions.** Upstream MDETR pins `transformers==4.5.1`,
> but this code was run against a much newer `transformers` release. The MDETR
> patch in [`mdetr/`](mdetr/) adapts the text-encoder tokenizer call to the
> modern `transformers` API (see the patch notes).

See each component's README for the exact commands.

## License

This repository is released under the [MIT License](LICENSE)