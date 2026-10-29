# Accuracy-energy benchmark for falsehood detection: code and results

Code, result files and figure scripts for the manuscript "Beyond Standard Attention: An Accuracy-Energy Benchmark for Falsehood Detection". The study measures the accuracy and inference energy of BERT-base, structured head pruning, distillation (DistilBERT, TinyBERT), INT8 quantization, a 64-token input control, an Energy-Based Transformer (EBT), SpikeBERT and a dense non-spiking control on LIAR, ISOT and FakeNewsNet.

## Contents

| Folder | What it holds |
|---|---|
| `p1_baseline/` | Fixed serving protocol (`config.py`: BERT-base, 256 tokens, batch size 16, FP16), baseline fine-tuning (`train.py`), NVML power sampler and component profiler (`profile_energy.py`), dataset download and split scripts (`data/`). |
| `p3_compression/` | `prune.py`, `distill.py`, `quantize.py`, `combined_framework/` (distillation then head pruning), `ebt/adapt_classification.py` (Energy-Based Transformer), `snn/adapt_spiking.py` (SpikeBERT), `linear_attention/` (earlier adaptations, imported by the seed driver). |
| `p3_compression/revision/` | The scripts behind the paper's results: `run_seed.py`, `run_ebt.py`, `run_snn.py`, `energy_unified.py`, `ood_eval.py`, `isot_leakage_check.py`, `final_tables.py`, `final_analysis.py`. |
| `p3_compression/results_rev_final/` | Metrics and per-example test predictions of the three-seed compression grid. |
| `p3_compression/results_rev_altarch_pull/` | Metrics and predictions for TinyBERT, INT8, the 64-token control, EBT, SpikeBERT, the dense control, the two energy replicates (`energy-unified-a`, `energy-unified-b`), and the generalization checks. `FINAL_TABLES.md` and `FINAL_TABLES.json` hold the rolled-up test-split tables. |
| `figures/` | `generate_figures_final.py` regenerates every figure of the paper from the result files; the figure PDFs are included. |

Model checkpoints and the raw datasets are not included.

## Environment

Python 3.10 or newer. `pip install -r requirements.txt`. Training and energy measurement need a CUDA GPU with NVML power readings (the reported numbers come from a Tesla T4 on Kaggle, power limit 70 W). Dynamic INT8 quantization runs on the CPU. SpikeBERT needs `spikingjelly`.

## Data

LIAR is collapsed to binary labels, ISOT uses a stratified 80/10/10 split with seed 42 (35,918 / 4,490 / 4,490), and FakeNewsNet is title-only (PolitiFact and GossipCop combined; 18,556 / 2,320 / 2,320). Download and split:

```
python p1_baseline/data/download_liar.py
python p1_baseline/data/download_isot.py       # needs Kaggle access to the public ISOT dataset
python p1_baseline/data/download_fakenewsnet.py
python p1_baseline/data/prepare_splits.py       # sanity check of the common schema and row counts
```

The scripts write `p1_baseline/results/data/{dataset}_{split}.parquet`, which the later scripts read.

## Reproducing the results

All commands run from the repository root, once per dataset (`liar`, `isot`, `fakenewsnet`) and seed (`42`, `43`, `44`).

```
# BERT-base, pruning and distillation grid (8 configurations)
python p3_compression/revision/run_seed.py --dataset liar --seed 42 --jobs bert,grid

# TinyBERT, INT8 (CPU), BERT-base at 64 tokens
python p3_compression/revision/run_seed.py --dataset liar --seed 42 --jobs bert,quant,tinybert,bert64

# Energy-Based Transformer (three refinement steps; use --mcmc_num_steps 1 or 2 for the ablation)
python p3_compression/revision/run_ebt.py --dataset liar --seed 42

# SpikeBERT, and the dense control under three recipes
python p3_compression/revision/run_snn.py --dataset liar --seed 42 --variant spiking
python p3_compression/revision/run_snn.py --dataset liar --seed 42 --variant dense
python p3_compression/revision/run_snn.py --dataset liar --seed 42 --variant dense --lr 1e-4 --warmup_steps 500 --tag _warm
python p3_compression/revision/run_snn.py --dataset liar --seed 42 --variant dense --lr 1e-4 --warmup_steps 500 --prenorm --tag _preln

# Energy, latency and memory of every model in one session (run twice on separate machines for the two replicates)
python p3_compression/revision/energy_unified.py

# ISOT checks
python p3_compression/revision/isot_leakage_check.py
python p3_compression/revision/ood_eval.py --mode cross --train liar    # also isot, fakenewsnet
python p3_compression/revision/ood_eval.py --mode temporal
python p3_compression/revision/ood_eval.py --mode ablate
```

Place each run's output under `p3_compression/results_rev_final/` or `p3_compression/results_rev_altarch_pull/` with the folder names used in the shipped results, then:

```
python p3_compression/revision/final_tables.py     # test-split, three-seed tables and relative energy
python p3_compression/revision/final_analysis.py   # frontier membership and paired bootstrap intervals
python figures/generate_figures_final.py           # figures
```

`final_analysis.py` and `final_tables.py` run on the shipped result files without any GPU and reproduce the numbers in the manuscript.

## Key settings

Fine-tuning: AdamW, learning rate 5e-5 with linear decay, 3 epochs, batch size 16, 256 tokens, FP16 mixed precision, checkpoint selected by validation F1. Distillation: temperature 2.0, weight 0.5, DistilBERT student and a TinyBERT-4L student initialized from `huawei-noah/TinyBERT_General_4L_312D`. Pruning: 25, 50 and 75% of attention heads by magnitude, one recovery epoch. EBT: two-layer energy transformer, learned step size initialized at 0.1, Gaussian logit noise 0.02 during training, learning rate 2e-5, 3 epochs, trained in FP32. SpikeBERT: 6 blocks, width 768, 16 timesteps, 64 tokens, 15 epochs, batch size 16, learning rate 6e-4.

Energy protocol: NVML power at about 10 Hz integrated over 20 and 200 forward passes on fixed-length synthetic inputs, a 30 s warm-up under load, a fixed shuffled configuration order, 10 repeats (5 for the slowest models), two replicate sessions. Energy is reported relative to BERT-base measured in the same session.

## Notes

- EBT inference starts from random logits, so a single evaluation is one draw; variability across draws was not measured.
- Absolute joules differ between machines of the same GPU type by 20 to 36%; use the relative energies.
- ISOT is saturated by source artifacts (see `isot_leakage_check.py` and `ood_eval.py`).
- No license file is included yet; add the license you intend before making the repository public.
