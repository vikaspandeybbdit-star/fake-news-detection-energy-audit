# Combined pruning + distillation framework (RO2's "propose a framework" deliverable)

## What this is

Every P3 milestone before this one (quantize/prune/distill, plus the exploratory
EBT/SNN/Linformer/Performer/Charformer adaptations) evaluated **one technique at a
time**. This directory composes the two techniques that actually reduced measured GPU
energy on their own — structured pruning (−11 to −14%) and knowledge distillation
(−46 to −49%) — into a single pipeline: distill to a DistilBERT student, then prune
that student's own attention heads, and see whether the two savings compound.

This is the original plan's Stage 2 / RO2 milestone that asks to "extend whichever of
quant/prune/distill/EBT worked... run `combine.py` + `search.py`... pick one
Pareto-favorable config as the proposed framework" — see
had not been started as of 2026-08-16; this directory is that milestone.

**Honest scope**: the novelty here is a *combination*, not a new mechanism. Pruning and
distillation are both well-established, individually-published techniques (see
`P3_compression_draft.md` refs [6], [7]). What hasn't been published (as far as this
project's own literature review, P2, found) is running them together on a fake-news
classifier and reporting the joint accuracy-energy Pareto frontier under one fixed
protocol. If this turns out not to compound — pruning a distilled model gives no
extra energy win, or costs more accuracy than either technique alone — that is exactly
as reportable a finding as every other negative/mixed result in this project (see
`P3_compression_draft.md` Section 5's discussion of EBT/SNN/Linformer/Performer).

## Files

- `combine.py` — runs ONE config: `{plain baseline | distilled student}` x
  `{prune ratio 0/25/50/75%}`. Distillation reuses `distill.py`'s own
  `DistillationTrainer` verbatim. Pruning uses `prune.py`'s existing BERT-specific
  logic for un-distilled configs, and a new DistilBERT-specific implementation
  (`prune_distilbert_heads`) for distilled configs — DistilBERT's attention module has
  a genuinely different layout (`q_lin/k_lin/v_lin/out_lin` on the attention module
  itself, vs. BERT's `self.query/key/value` + a separate `output.dense`) confirmed by
  reading the real `DistilBertSelfAttention` source before writing this, not assumed.
- `search.py` — loops the full 8-cell grid (4 prune ratios x 2 distill settings),
  writes `results/{dataset}/search_results.csv` (all 8) and
  `results/{dataset}/pareto_frontier.csv` (the non-dominated subset — the actual
  "proposed framework" output, a set of tradeoff points, not one number).
- `METHODOLOGY_NOTES.md` — paper-prep notes: novelty framing, open questions to
  resolve before drafting, a starter section outline.

## Status: written, NOT yet run on Kaggle

No GPU on this local machine for an 8-cell grid (each cell up to 3 epochs of
distillation + 1 epoch of prune-recovery). Written and reviewed locally
(`python -m py_compile` clean on both files; DistilBERT's module structure verified
against real source, matching this project's established practice of reading real code
before writing an adaptation — same discipline used for every M9-M13 architecture).

**Before spending Kaggle quota**: smoke-test one cell manually first —
```
python combine.py --dataset liar --prune_ratio 0.5 --distill --smoke_test
```
— (8 examples, 1 epoch per stage, genuinely cheap — `--smoke_test` was added after
noticing this file, unlike every other adaptation script in the project, initially had
no truncation flag at all) to catch any real bug (device placement, shape mismatch,
DistilBERT attribute typo) the way every prior milestone in this project caught at
least one such bug before its first full run. Only run `search.py`'s full 8-cell grid
(also supports `--smoke_test`, runs all 8 cells at tiny scale) after that single cell
works end-to-end.

## How to run (when ready to push to Kaggle)

Same pattern as `p3_compression/notebooks/push_linformer/` etc. — a Kaggle notebook
that mounts `p1-baseline` + `p3-compression` (+ the per-dataset checkpoint dataset,
same as Linformer/Performer needed — see `p1_baseline_pipeline_kaggle` memory for why
a dedicated checkpoint dataset is needed instead of relying on `p1-baseline` itself),
then runs:
```
!cd {P3_DIR} && python combined_framework/search.py --dataset liar
```
No push folder has been created yet for this — build one (`push_combined/`,
`push_combined_full/`) following the exact pattern in `notebooks/push_linformer/`
when ready to actually run this.

## Dataset scope

LIAR only, for the same reason every exploratory M9-M13 milestone was LIAR-only: it's
the hardest of the three datasets under this project's baseline, and the one where
prior techniques' effects were most differentiated (P3_compression_draft.md Section
4.4). Extending the winning frontier config to ISOT/FakeNewsNet, once found, is a
reasonable follow-up — cheap by comparison, since it's one config, not an 8-cell grid.
