"""Spiking Transformer (SpikeBERT-style) adapted to text classification on LIAR.

M10, exploratory: the source PhD-scoping document's Stage 3 (neuromorphic/spiking
transformers) names fake news detection as an open target, and the most advanced
text-capable spiking transformers published to date -- SpikeBERT (Lu et al. 2023,
arXiv:2308.15122, github.com/Lvchangze/SpikeBERT) and SpikingBERT -- were only ever
evaluated on generic text classification (GLUE-style, sentiment/topic), never on fake
news detection specifically. Same shape of gap as the EBT-for-classification gap that
motivated M9, on a different architecture family.

This file ports SpikeBERT's own published architecture directly (`spiking_self_attention`,
`mlp`, `block`, `transform`, `new_spikformer`, adapted from
github.com/Lvchangze/SpikeBERT/blob/main/model/new_spikformer.py, read verbatim from
source rather than from a paraphrase) -- unlike M9, a real, working reference
implementation exists here, so the contribution is applying it to a new task (LIAR),
not inventing a classification adaptation from a generative-only reference.

CRITICAL SCOPE CAVEAT, stated up front per explicit user direction (2026-08-15): a
Spiking Neural Network's actual energy-efficiency claim comes from running on real
event-driven neuromorphic silicon (Intel Loihi 2, SpiNNaker2) that only draws power
where a spike fires. We do not have access to that hardware (it requires a separate
institutional-access application, not obtainable via Kaggle). Simulating this model on
a Kaggle GPU processes every one of the T timesteps as a dense tensor op regardless of
actual spike sparsity, so the measured GPU energy number below is a real measurement of
THIS SIMULATION, not evidence about neuromorphic efficiency -- it is reported anyway,
per explicit user direction, purely as a same-protocol benchmarking data point, and
MUST carry this disclaimer wherever it is cited (ebt_metrics.csv-equivalent output
below always writes it into the CSV itself, not just this docstring).

Deliberate scope/deviation decisions, time-boxed like M9 (LIAR only, single go/no-go):

- T (timesteps) = 16, not SpikeBERT's own default of 32: compute-tractability for a
  time-boxed Kaggle run (T multiplies per-layer compute directly).
- max_length = 64, not this project's fixed protocol seq_len=256: attention cost is
  O(L^2) per timestep per layer, so seq_len=256 combined with T=16 timesteps and
  depths=6 would not fit a time-boxed run. LIAR statements are short single sentences,
  so 64 tokens covers the great majority of examples without truncation loss. This
  means SNN results are NOT run under the same fixed protocol as P1/P3/M9 and are not
  directly comparable on that axis -- an explicit scope difference, not an oversight,
  in the same spirit as quantize.py's CPU-only scope note.
- epochs = 15, not SpikeBERT's own 100: this model trains its embedding table and all
  weights from scratch (no pretrained BERT weights anywhere in it, unlike M9 which
  reuses the fine-tuned BERT context encoder) -- on ~10k LIAR examples with only 15
  epochs, accuracy is expected to land well below the P1 baseline (62.3%) and likely
  below M9's EBT result too. This is an expected consequence of training an
  architecture from random initialization on a small labeled set, not a bug, and is
  reported plainly rather than run for 100 epochs to try to close that gap (out of
  scope for a go/no-go check).
- vocab: bert-base-uncased tokenizer (matches this project's convention), not
  SpikeBERT's own bert-base-cased -- only used for tokenization/vocab size, no BERT
  weights are loaded (the spiking model's nn.Embedding is trained from scratch either
  way).

Usage: python adapt_spiking.py --dataset liar
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from spikingjelly.activation_based import functional, neuron, surrogate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "p1_baseline"))
from config import MAX_SEQ_LEN as PROJECT_SEQ_LEN, SEED  # noqa: E402,F401
from profile_energy import PowerSampler  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NUM_LABELS = 2

def make_lif(tau, thr, step_mode="m"):
    """spikingjelly's LIFNode dropped the `backend` kwarg in some newer releases
    (confirmed: TypeError on Kaggle's auto-installed latest version) -- pass it only if
    the installed version still accepts it, so this works across spikingjelly releases
    without pinning a specific version."""
    import inspect
    kwargs = dict(tau=tau, step_mode=step_mode, detach_reset=DETACH_RESET,
                  surrogate_function=surrogate.ATan(), v_threshold=thr)
    if "backend" in inspect.signature(neuron.LIFNode.__init__).parameters:
        kwargs["backend"] = BACKEND
    return neuron.LIFNode(**kwargs)


TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 64          # see module docstring: deviates from project's fixed seq_len=256
# SpikeBERT's own default is batch=32; reduced to 8 here after that OOM'd a Kaggle T4
# (16GB) on the very first training step -- LIF multi-step mode ('m') keeps all T=16
# timesteps' activations resident simultaneously across all 6 layers, so memory scales
# with T*depths*batch*seq_len, not just batch*seq_len like a normal transformer forward.
# Confirmed empirically: batch=32 missed fitting by only ~96MiB, so this isn't a wild
# overcorrection, just enough headroom for optimizer state + autograd bookkeeping too.
BATCH_SIZE = 8
BACKEND = "torch"
DETACH_RESET = True


# ---------------------------------------------------------------------------
# Model: ported from github.com/Lvchangze/SpikeBERT/blob/main/model/new_spikformer.py
# (Lu et al. 2023, arXiv:2308.15122), adapted only in default hyperparameters and the
# training/eval/profiling harness around it -- the SSA/MLP/block mechanism is unchanged.
# ---------------------------------------------------------------------------

class SpikingSelfAttention(nn.Module):
    def __init__(self, tau, common_thr, dim, heads=8, qk_scale=0.125):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        def lif(thr=common_thr):
            return make_lif(tau, thr)

        self.q_m, self.q_ln, self.q_lif = nn.Linear(dim, dim), nn.LayerNorm(dim), lif()
        self.k_m, self.k_ln, self.k_lif = nn.Linear(dim, dim), nn.LayerNorm(dim), lif()
        self.v_m, self.v_ln, self.v_lif = nn.Linear(dim, dim), nn.LayerNorm(dim), lif()
        self.attn_lif = lif(common_thr / 2)
        self.last_m, self.last_ln, self.last_lif = nn.Linear(dim, dim), nn.LayerNorm(dim), lif()

    def forward(self, x):  # x: B T L D
        x = x.transpose(0, 1)  # T B L D
        T, B, L, D = x.shape
        x_flat = x.flatten(0, 1)  # TB L D

        q = self.q_lif(self.q_ln(self.q_m(x_flat)).reshape(T, B, L, D).contiguous())
        q = q.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()
        k = self.k_lif(self.k_ln(self.k_m(x_flat)).reshape(T, B, L, D).contiguous())
        k = k.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()
        v = self.v_lif(self.v_ln(self.v_m(x_flat)).reshape(T, B, L, D).contiguous())
        v = v.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = q @ k.transpose(-2, -1)          # no softmax -- spikes are already binary
        x = (attn @ v) * self.qk_scale          # T B heads L head_dim
        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_ln(x)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x.transpose(0, 1)  # B T L D


class SpikingMLP(nn.Module):
    def __init__(self, tau, common_thr, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.lif1 = make_lif(tau, common_thr)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.ln2 = nn.LayerNorm(in_features)
        self.lif2 = make_lif(tau, common_thr)

    def forward(self, x):  # B T L D
        x = x.transpose(0, 1)
        T, B, L, D = x.shape
        x = x.flatten(0, 1)
        x = self.lif1(self.ln1(self.fc1(x)).reshape(T, B, L, -1).contiguous())
        x = x.flatten(0, 1)
        x = self.lif2(self.ln2(self.fc2(x)).reshape(T, B, L, D).contiguous())
        return x.transpose(0, 1)


class SpikingBlock(nn.Module):
    def __init__(self, tau, common_thr, dim, heads=8, qk_scale=0.125):
        super().__init__()
        self.attn = SpikingSelfAttention(tau, common_thr, dim, heads, qk_scale)
        self.mlp = SpikingMLP(tau, common_thr, dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingClassifier(nn.Module):
    def __init__(self, vocab_size, dim=768, depths=6, T=16, tau=10.0, common_thr=1.0,
                 num_labels=NUM_LABELS, heads=8, qk_scale=0.125):
        super().__init__()
        self.T = T
        self.atan = surrogate.ATan()
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([SpikingBlock(tau, common_thr, dim, heads, qk_scale) for _ in range(depths)])
        self.last_ln = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_labels)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, input_ids):  # B L
        x = self.emb(input_ids)  # B L D
        x = x.unsqueeze(1).repeat(1, self.T, 1, 1)  # B T L D -- same embedding, T copies
        x = self.atan(x)  # surrogate-binarized: this IS the text-to-spike encoding step
        for blk in self.blocks:
            x = blk(x)
        x = self.last_ln(x)
        x = x.mean(2)  # B T D -- mean-pool over sequence length
        return self.classifier(x)  # B T num_labels


# ---------------------------------------------------------------------------
# Data / training harness (project conventions: same metric set and results-dir layout
# as adapt_classification.py, not SpikeBERT's own train_spikformer.py harness).
# ---------------------------------------------------------------------------

def load_split(dataset: str, split: str) -> Dataset:
    df = pd.read_parquet(os.path.join(P1_RESULTS_DIR, "data", f"{dataset}_{split}.parquet"))
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def make_loader(ds, tokenizer, batch_size, shuffle):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "label"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def timestep_logits(model, input_ids, T, num_labels):
    """Runs the model and mean-pools its T-timestep output into final class logits,
    matching SpikeBERT's own train_spikformer.py reduction (B T num_labels -> B num_labels)."""
    out = model(input_ids)                       # B T num_labels
    return out.mean(dim=1)                        # B num_labels


def evaluate(model, loader, device, T, num_labels):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = timestep_logits(model, input_ids, T, num_labels)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())
            functional.reset_net(model)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": matthews_corrcoef(labels, preds),
    }


def profile_energy_snn(model, loader, device, T, num_labels, n_iters=20):
    """GPU-simulation energy measurement. See module docstring's CRITICAL SCOPE CAVEAT --
    this is NOT a neuromorphic-efficiency measurement, it is a same-GPU-protocol
    benchmarking number reported per explicit user direction, disclaimer carried into
    the output CSV itself so it can't be quoted out of context."""
    model.eval()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)

    with PowerSampler() as sampler:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                timestep_logits(model, input_ids, T, num_labels)
                functional.reset_net(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.perf_counter() - t0

    return {
        "n_iters": n_iters,
        "batch_size": input_ids.shape[0],
        "total_time_s": total_time,
        "energy_j_total": sampler.energy_joules(),
        "energy_method": sampler.method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liar", choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--depths", type=int, default=6)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--common_thr", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                         help="Override BATCH_SIZE (default 8, required for the LIAR/FakeNewsNet 16GB-T4 memory "
                              "ceiling at batch=32). ISOT's 15-epoch run at batch=8 takes ~12h and repeatedly "
                              "hit Kaggle's session cap right before finishing; batch=32 OOM'd by only 96MB, so "
                              "batch=16 has real headroom and roughly halves wall-clock without changing epochs.")
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny run (few examples, 1 epoch) to check the pipeline doesn't crash.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = SpikingClassifier(vocab_size=len(tokenizer), dim=args.dim, depths=args.depths,
                               T=args.T, tau=args.tau, common_thr=args.common_thr).to(device)

    train_ds = load_split(args.dataset, "train")
    val_ds = load_split(args.dataset, "val")
    if args.smoke_test:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        val_ds = val_ds.select(range(min(8, len(val_ds))))
        args.epochs = 1

    batch_size = 4 if args.smoke_test else args.batch_size
    print(f"RUNTIME CONFIG: batch_size={batch_size}, epochs={args.epochs}, "
          f"train_examples={len(train_ds)}, steps_per_epoch={-(-len(train_ds) // batch_size)}")
    train_loader = make_loader(train_ds, tokenizer, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, tokenizer, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = timestep_logits(model, input_ids, args.T, NUM_LABELS)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model)

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, args.T, NUM_LABELS)
        print(f"epoch {epoch} done, avg_train_loss={running_loss / (step + 1):.4f}, val={val_metrics}")

    if args.smoke_test:
        print("Smoke test finished without crashing:", val_metrics)
        return

    results_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(results_dir, "snn_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    profiled = profile_energy_snn(model, val_loader, device, args.T, NUM_LABELS)
    print(profiled)

    with open(os.path.join(results_dir, "snn_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in ("accuracy", "precision", "recall", "f1", "mcc"):
            writer.writerow([k, val_metrics[k]])
        writer.writerow(["energy_j_total_gpu_simulated", profiled["energy_j_total"]])
        writer.writerow(["energy_method", profiled["energy_method"]])
        writer.writerow(["T_timesteps", args.T])
        writer.writerow(["max_length", MAX_LENGTH])
        writer.writerow(["epochs", args.epochs])
        writer.writerow([
            "disclaimer",
            "GPU-simulated SNN energy is NOT a neuromorphic-efficiency measurement -- "
            "real neuromorphic silicon (Loihi2/SpiNNaker2) only draws power on an actual "
            "spike event; this GPU processes all T timesteps as dense ops regardless of "
            "spike sparsity, so this number is reported only as a same-GPU-protocol "
            "benchmarking data point, not a claim about spiking-hardware efficiency.",
        ])
    print(f"Wrote {results_dir}/snn_metrics.csv")


if __name__ == "__main__":
    main()
