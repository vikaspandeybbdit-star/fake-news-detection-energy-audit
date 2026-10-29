| model | dataset | n seeds | accuracy (mean +- SD) | MCC (mean +- SD) |
|---|---|---|---|---|
| bert_base | liar | 3 | 61.98 +- 2.63 | 0.206 +- 0.060 |
| bert_base | isot | 3 | 99.95 +- 0.06 | 0.999 +- 0.001 |
| bert_base | fakenewsnet | 3 | 84.81 +- 0.86 | 0.571 +- 0.019 |
| bert_prune25 | liar | 3 | 63.61 +- 2.32 | 0.246 +- 0.051 |
| bert_prune25 | isot | 3 | 99.98 +- 0.02 | 1.000 +- 0.000 |
| bert_prune25 | fakenewsnet | 3 | 84.68 +- 0.66 | 0.577 +- 0.011 |
| bert_prune50 | liar | 3 | 64.09 +- 1.54 | 0.256 +- 0.034 |
| bert_prune50 | isot | 3 | 99.96 +- 0.06 | 0.999 +- 0.001 |
| bert_prune50 | fakenewsnet | 3 | 84.90 +- 0.62 | 0.584 +- 0.009 |
| bert_prune75 | liar | 3 | 63.61 +- 1.01 | 0.248 +- 0.019 |
| bert_prune75 | isot | 3 | 99.90 +- 0.03 | 0.998 +- 0.001 |
| bert_prune75 | fakenewsnet | 3 | 84.84 +- 0.38 | 0.579 +- 0.004 |
| distil_prune0 | liar | 3 | 61.80 +- 1.81 | 0.202 +- 0.045 |
| distil_prune0 | isot | 3 | 99.93 +- 0.04 | 0.999 +- 0.001 |
| distil_prune0 | fakenewsnet | 3 | 85.24 +- 0.50 | 0.583 +- 0.018 |
| distil_prune25 | liar | 3 | 64.35 +- 0.71 | 0.262 +- 0.017 |
| distil_prune25 | isot | 3 | 99.95 +- 0.05 | 0.999 +- 0.001 |
| distil_prune25 | fakenewsnet | 3 | 84.89 +- 0.43 | 0.584 +- 0.008 |
| distil_prune50 | liar | 3 | 63.83 +- 1.16 | 0.252 +- 0.025 |
| distil_prune50 | isot | 3 | 99.94 +- 0.01 | 0.999 +- 0.000 |
| distil_prune50 | fakenewsnet | 3 | 84.78 +- 0.16 | 0.581 +- 0.003 |
| distil_prune75 | liar | 3 | 62.64 +- 0.40 | 0.227 +- 0.007 |
| distil_prune75 | isot | 3 | 99.93 +- 0.04 | 0.999 +- 0.001 |
| distil_prune75 | fakenewsnet | 3 | 84.70 +- 0.30 | 0.575 +- 0.009 |
| suite_bert_base | liar | 3 | 60.59 +- 1.89 | 0.172 +- 0.046 |
| suite_bert_base | isot | 3 | 99.97 +- 0.05 | 0.999 +- 0.001 |
| suite_bert_base | fakenewsnet | 3 | 85.09 +- 0.24 | 0.583 +- 0.006 |
| suite_tinybert | liar | 3 | 61.56 +- 1.89 | 0.195 +- 0.047 |
| suite_tinybert | isot | 3 | 99.90 +- 0.03 | 0.998 +- 0.001 |
| suite_tinybert | fakenewsnet | 3 | 83.84 +- 0.65 | 0.554 +- 0.013 |
| suite_quant_int8 | liar | 3 | 58.64 +- 1.86 | 0.116 +- 0.064 |
| suite_quant_int8 | isot | 3 | 99.98 +- 0.04 | 1.000 +- 0.001 |
| suite_quant_int8 | fakenewsnet | 3 | 85.45 +- 0.42 | 0.582 +- 0.012 |
| suite_bert_len64 | liar | 3 | 63.35 +- 2.04 | 0.237 +- 0.048 |
| suite_bert_len64 | isot | 3 | 99.95 +- 0.03 | 0.999 +- 0.001 |
| suite_bert_len64 | fakenewsnet | 3 | 84.93 +- 0.35 | 0.570 +- 0.012 |
| ebt_T3 | liar | 3 | 62.80 +- 2.04 | 0.224 +- 0.048 |
| ebt_T3 | isot | 3 | 99.73 +- 0.21 | 0.995 +- 0.004 |
| ebt_T3 | fakenewsnet | 3 | 85.01 +- 0.61 | 0.571 +- 0.019 |
| spikebert | liar | 3 | 60.59 +- 0.12 | 0.172 +- 0.003 |
| spikebert | isot | 3 | 99.87 +- 0.10 | 0.997 +- 0.002 |
| spikebert | fakenewsnet | 3 | 83.58 +- 0.09 | 0.529 +- 0.013 |
| dense_preln | liar | 3 | 60.83 +- 0.43 | 0.180 +- 0.014 |
| dense_preln | isot | 3 | 99.83 +- 0.01 | 0.997 +- 0.000 |
| dense_preln | fakenewsnet | 3 | 82.96 +- 0.22 | 0.503 +- 0.014 |
| dense_warm | liar | 3 | 57.75 +- 1.21 | 0.081 +- 0.070 |
| dense_warm | isot | 3 | 99.81 +- 0.01 | 0.996 +- 0.000 |
| dense_warm | fakenewsnet | 3 | 82.34 +- 0.42 | 0.485 +- 0.008 |

| config | energy relative to BERT-base (mean of 2 sessions) |
|---|---|
| BERT-base 64 tokens | 0.220 |
| TinyBERT-4L | 0.296 |
| DistilBERT prune 75% | 0.375 |
| DistilBERT prune 50% | 0.414 |
| DistilBERT prune 25% | 0.417 |
| DistilBERT prune 0% | 0.492 |
| Dense control FP32 | 0.580 |
| BERT-base prune 75% | 0.753 |
| BERT-base prune 50% | 0.830 |
| BERT-base prune 25% | 0.907 |
| BERT-base prune 0% | 1.000 |
| EBT T=1 FP16 | 1.430 |
| EBT T=2 FP16 | 1.863 |
| EBT T=3 FP16 | 2.281 |
| SpikeBERT T=16 FP32 | 9.170 |
| EBT T=3 FP32 | 9.797 |