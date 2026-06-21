# MDETR integration

The training and evaluation code in this repository builds on **MDETR**
([github.com/ashkamath/mdetr](https://github.com/ashkamath/mdetr)), an external
project with its own Apache-2.0 license. We do **not** redistribute MDETR here.
Instead, this folder lets you reproduce our fork from a pinned upstream commit:

- [`gridvqa_mdetr.patch`](gridvqa_mdetr.patch) — our modifications to MDETR, as a
  `git apply`-able patch (3 files changed).
- [`CHANGES.md`](CHANGES.md) — a human-readable description of every change.

## Upstream reference

| | |
|---|---|
| Repository | https://github.com/ashkamath/mdetr |
| Pinned commit | `ea09acc44ca067072c4b143b726447ee7ff66f5f` |
| License | Apache-2.0 |
| Paper | Kamath et al., *MDETR — Modulated Detection for End-to-End Multi-Modal Understanding*, ICCV 2021 |

## Setup

```bash
# 1. Clone upstream MDETR at the pinned commit
git clone https://github.com/ashkamath/mdetr.git
cd mdetr
git checkout ea09acc44ca067072c4b143b726447ee7ff66f5f

# 2. Apply our changes
git apply /path/to/GridVQA_Release/mdetr/gridvqa_mdetr.patch
#   (or: patch -p1 < /path/to/GridVQA_Release/mdetr/gridvqa_mdetr.patch)

# 3. Drop in the GridVQA training + evaluation scripts
cp /path/to/GridVQA_Release/training/*.py   .
cp /path/to/GridVQA_Release/training/*.sh   .
cp /path/to/GridVQA_Release/evaluation/eval_accuracy.py  .
```

After this, follow [`../training/README.md`](../training/README.md) and
[`../evaluation/README.md`](../evaluation/README.md). All those scripts run from
the root of this patched MDETR checkout (they do `from models import mdetr`).

The explainability wrapper ([`../explainability/`](../explainability/)) also
imports the patched MDETR as the `mdetr` package — see its README for how to put
this checkout on the import path.

## Verifying the patch applies

```bash
git apply --check /path/to/GridVQA_Release/mdetr/gridvqa_mdetr.patch
```

If upstream has moved and the patch no longer applies cleanly, use
[`CHANGES.md`](CHANGES.md) to re-apply the three edits by hand.
