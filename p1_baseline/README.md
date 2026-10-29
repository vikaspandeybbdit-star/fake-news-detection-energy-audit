# P1 — Baseline BERT-base + IrEne-style energy profiling

Stage 1 / RO1 of the 4-paper series. Produces the reference accuracy + component-wise
energy budget that P3 (compression + EBT) and P4 (comparative benchmarking) build on.
See `..\P2_paper_creation_process_en.md` and `..\vikas_pandey_research_plan.md` for the
research context, and the plan at
full pipeline (P1/P3/P4) and milestone sequencing.

## Fixed protocol (config.py) — never override per-run

| Setting | Value |
|---|---|
| Backbone | `bert-base-uncased` |
| Max sequence length | 256 |
| Batch size | 16 |
| Precision | fp16 |
| Seed | 42 |

This is fixed because P2 (survey) Section 3.3 found that batching/serving-config choices
alone can swing per-request energy by up to two orders of magnitude — any energy delta
reported across datasets or later compression conditions must come from the model, not
from an accidentally different batch/seq/precision setting.

## Datasets

- **LIAR** — HF `liar`, official split, collapsed to **binary** (true/false).
- **ISOT** — Kaggle `clmentbisaillon/fake-and-real-news-dataset`, our own stratified
  80/10/10 split (no official split exists).
- **FakeNewsNet** — Kaggle `mdepak/fakenewsnet`, **title-only** (politifact + gossipcop
  combined), stratified 80/10/10. Narrower than the original multimodal/propagation
  benchmark — name this as a scope limitation in the P1 paper.

## Execution — Kaggle only

All training/profiling runs on Kaggle (T4/P100, ~16GB VRAM, 9h session cap, no
persistent disk). Sync this folder as a Kaggle Dataset, run `notebooks/p1_kaggle.ipynb`.
Run `notebooks/p1_smoketest.ipynb` **first** (Milestone 0) — it confirms whether NVML
power-draw queries work on Kaggle's GPU or are blocked by the hypervisor. If blocked,
`profile_energy.py`'s `PowerSampler` automatically falls back to a fixed-TDP energy
estimate (labeled `tdp_estimated` in the output, vs. `nvml_measured`) — per sign-off,
no dependency on the local machine for this fallback.

## Milestones

M0 NVML smoke test → M1 LIAR train → M2 LIAR profile (sanity-check the table) →
M3 ISOT → M4 FakeNewsNet → M5 assemble cross-dataset summary, P1 paper-writing starts.

## Local machine note

The local GTX 1650 (4GB, NVML power-draw confirmed working) is available as an optional
secondary calibration point but is not part of the primary pipeline (Kaggle only, per
project decision).
