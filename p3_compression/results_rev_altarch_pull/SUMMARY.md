## EBT, multi-seed test-split (T=3, main track)

| Dataset | seed42 | seed43 | seed44 | mean | SD |
|---|---|---|---|---|---|
| liar | 0.6377 | 0.6046 | 0.6417 | 0.6280 | 0.0204 |
| isot | 0.9987 | 0.9984 | 0.9949 | 0.9973 | 0.0021 |
| fakenewsnet | 0.8496 | 0.8444 | 0.8565 | 0.8501 | 0.0061 |

### EBT energy / latency / throughput (mean over 5 repeats, per run) + energy-descent check

| Dataset | Seed | Energy (J) mean±SD | Latency (s/iter) | Throughput (ex/s) | Energy descends monotonically |
|---|---|---|---|---|---|
| liar | 42 | 160.71 ± 4.77 | 0.1253 | 127.7 | True |
| liar | 43 | 165.67 ± 4.15 | 0.1283 | 124.7 | True |
| liar | 44 | 159.06 ± 3.35 | 0.1220 | 131.2 | True |
| isot | 42 | 153.05 ± 1.70 | 0.1193 | 134.1 | True |
| isot | 43 | 154.00 ± 7.22 | 0.1175 | 136.1 | True |
| isot | 44 | 144.31 ± 2.48 | 0.1131 | 141.5 | True |
| fakenewsnet | 42 | 145.68 ± 3.81 | 0.1144 | 139.9 | True |
| fakenewsnet | 43 | 152.72 ± 3.76 | 0.1182 | 135.3 | True |
| fakenewsnet | 44 | 177.52 ± 6.09 | 0.1382 | 115.8 | True |

## EBT refinement-step ablation (LIAR, seed 42)

| T | Accuracy | F1 | MCC | Energy (J) mean±SD |
|---|---|---|---|---|
| 1 | 0.5801 | 0.6833 | 0.1108 | 86.72 ± 2.81 |
| 2 | 0.6354 | 0.7243 | 0.2415 | 121.08 ± 5.59 |
| 3 (main) | 0.6377 | 0.7151 | 0.2468 | 160.71 ± 4.77 |

## SpikeBERT (spiking), multi-seed test-split

| Dataset | seed42 | seed43 | mean | SD |
|---|---|---|---|---|
| liar | 0.6062 | 0.6046 | 0.6054 | 0.0011 |
| isot | 0.9993 | 0.9976 | 0.9984 | 0.0013 |
| fakenewsnet | 0.8349 | 0.8366 | 0.8358 | 0.0012 |

## SpikeBERT dense-from-scratch control (seed 42, test-split)

| Dataset | Accuracy | F1 | MCC | Collapsed to majority class? |
|---|---|---|---|---|
| liar | 0.5635 | 0.7208 | 0.0000 | YES |
| isot | 0.5229 | 0.0000 | 0.0000 | YES |
| fakenewsnet | 0.7517 | 0.8583 | 0.0000 | YES |

### Spiking vs. dense-control, same seed (42), test-split — isolates the confound

| Dataset | Spiking acc | Spiking MCC | Dense-control acc | Dense-control MCC |
|---|---|---|---|---|
| liar | 0.6062 | 0.1726 | 0.5635 | 0.0000 |
| isot | 0.9993 | 0.9987 | 0.5229 | 0.0000 |
| fakenewsnet | 0.8349 | 0.5198 | 0.7517 | 0.0000 |

## TinyBERT second baseline vs. its own teacher (seed 42, test-split, same harness as DistilBERT)

| Dataset | BERT-base (teacher) acc | TinyBERT acc | TinyBERT F1 | TinyBERT MCC | TinyBERT params |
|---|---|---|---|---|---|
| liar | 0.6069 | 0.6164 | 0.7213 | 0.1982 | 14350874 |
| isot | 0.9989 | 0.9989 | 0.9988 | 0.9978 | 14350874 |
| fakenewsnet | 0.8453 | 0.8358 | 0.8910 | 0.5583 | 14350874 |
