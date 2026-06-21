# Changes to upstream MDETR

This file documents every change we made to the upstream MDETR source so they
can be reviewed independently of the patch. The machine-applicable form of these
changes is [`gridvqa_mdetr.patch`](gridvqa_mdetr.patch).

**Upstream:** https://github.com/ashkamath/mdetr
**Pinned commit:** `ea09acc44ca067072c4b143b726447ee7ff66f5f`
**License:** Apache-2.0 (see the `LICENSE` file in the upstream repo)

Three source files are modified. No upstream files are deleted.

## 1. `models/mdetr.py` — QA dataset selection in `build()`

Upstream derives the QA dataset (`clevr` vs `gqa`) from `args.combine_datasets`
and asserts that one of those datasets is present. Our GridVQA training scripts
do not use the `combine_datasets` machinery, so this is replaced with a direct
read of `args.qa_dataset` (which must be `"gqa"` or `"clevr"`), raising a clear
error if it is unset. This lets the GridVQA training entry points configure the
QA head without pulling in the CLEVR/GQA dataset wiring.

## 2. `models/transformer.py` — text encoder + decoder + feature export

- **Modern `transformers` tokenizer API.** The text encoder call is changed from
  the deprecated `tokenizer.batch_encode_plus(...)` to the standard
  `tokenizer(...)` call, and we explicitly add `token_type_ids` (zeros) when the
  tokenizer does not return them. This is required to run against current
  `transformers` releases (upstream pins `transformers==4.5.1`).
- **Re-enable decoder text cross-attention.** Upstream comments out the decoder's
  `cross_attn_text` block (and its `norm2`/`dropout2`). We restore it so the
  decoder attends to the text modality — needed for the GridVQA QA/grounding task.
- **Avoid an in-place aliasing issue** by adding `.clone()` to the repeated
  `query_embed`.
- **Export pre-fusion image features.** We save `img_src_pre_fusion` (the image
  tokens before encoder cross-attention) into the returned dict. This is used by
  downstream analysis/interpretability tooling and is otherwise inert for
  training/eval.

## 3. `util/box_ops.py` — robust box conversion

`box_cxcywh_to_xyxy` now clamps width and height to `>= 0` before converting to
xyxy, preventing negative box dimensions that otherwise trip a downstream
assertion on some GridVQA samples.
