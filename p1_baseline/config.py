"""Fixed protocol for P1/P3/P4. Import these constants everywhere; never override per-run.

Why fixed: P2 (survey, Section 3.3) found request-scheduling/batching choices alone can
swing per-request energy by up to two orders of magnitude. Holding these constant across
every dataset and every condition is what makes energy deltas attributable to the model
being measured instead of serving-config noise.
"""

MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
PRECISION = "fp16"
SEED = 42

DATASETS = ("liar", "isot", "fakenewsnet")
