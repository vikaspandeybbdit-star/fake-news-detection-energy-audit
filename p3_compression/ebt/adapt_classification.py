"""EBT (Energy-Based Transformer, Gladstone et al. 2026, ICLR Oral;
github.com/alexiglad/EBT) adapted to text classification on LIAR.

M9 of the P3 (compression) scope. Confirmed by reading model/nlp/ebt.py
directly (not just the P2 survey's gap analysis): the reference repo's NLP
module (`EBT_NLP`) implements next-token autoregressive language modeling
only -- there is no classification path anywhere in the repo. This file is
an independent re-implementation of EBT's core mechanism -- corrupt an
initial prediction, then refine it via gradient descent on a learned scalar
energy landscape (paper Section 3; `ebt.py` lines 69-188 for the reference
per-token version) -- retargeted from "predict next token" to "predict
class label".

Time-boxed single go/no-go attempt: if this doesn't train to a usable,
evaluable classifier within one Kaggle run, "evaluated, found impractical
to adapt within scope" is the citable finding, stated plainly in
results/liar/ebt_metrics.csv, not forced.

Deliberate scope decisions (deviations from both EBT's reference NLP module
and this project's own fixed protocol in p1_baseline/config.py), stated
explicitly so a later consistency pass doesn't flag them as oversights:

- fp32, not the fixed-protocol fp16: torch.autograd.grad(create_graph=True)
  (needed so the outer training step can backprop through the MCMC gradient
  steps themselves) is unsupported/unstable for several ops under fp16
  autocast.
- BERT context is encoded ONCE per example, not re-run every MCMC step. In
  EBT's reference implementation the full transformer *is* the energy
  function and reprocesses the whole (real+predicted) sequence jointly
  every step; porting that literally would mean re-running all 12 BERT
  layers `mcmc_num_steps` times per training step, which does not fit a
  Kaggle time-box. Instead a shallow bidirectional transformer performs
  fresh joint attention over [this-step-fixed context, current label
  guess] at every MCMC step -- that is where the actual energy-based
  refinement lives; only the (unchanging-within-a-step) context
  featurization is hoisted out of the loop. Gradients still flow back into
  the BERT encoder through every step (same mechanism the reference repo
  uses for its own once-computed token embedding table), so the encoder is
  fine-tuned end-to-end, keeping this comparable to the P1/P3 baselines
  which all fine-tune bert-base-uncased.
- Only the core mechanism is ported: corrupted init -> per-step Langevin
  noise (train only) -> softmax-normalize -> embed -> joint energy ->
  autograd.grad -> gradient step -> repeat, loss averaged over all MCMC
  steps. Optional reference-repo features not needed to test whether the
  mechanism works at all (contrastive loss, replay buffer, adaLN/time-embed
  variants, multi-sample inference, gradient clamping, energy-scaled step
  size) are intentionally omitted.

Usage: python adapt_classification.py --dataset liar
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "p1_baseline"))
from config import BATCH_SIZE, MAX_SEQ_LEN, MODEL_NAME, SEED  # noqa: E402
from profile_energy import PowerSampler  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NUM_LABELS = 2


class EnergyClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, energy_layers=2,
                 mcmc_num_steps=3, mcmc_step_size=0.1, langevin_noise=0.02, init_noise_scale=1.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.num_labels = num_labels
        self.mcmc_num_steps = mcmc_num_steps
        self.langevin_noise = langevin_noise
        self.init_noise_scale = init_noise_scale

        # analog of EBT's vocab_to_embed, retargeted from vocab-size to num_labels
        self.label_to_embed = nn.Linear(num_labels, hidden_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.energy_transformer = nn.TransformerEncoder(layer, num_layers=energy_layers)
        self.energy_head = nn.Linear(hidden_dim, 1)
        # analog of EBT's self.alpha (learnable MCMC step size)
        self.alpha = nn.Parameter(torch.tensor(float(mcmc_step_size)))

    def _energy(self, context_hidden, context_key_padding_mask, y_probs, need_double_backward: bool):
        label_embed = self.label_to_embed(y_probs).unsqueeze(1)  # (B, 1, D)
        combined = torch.cat([context_hidden, label_embed], dim=1)  # (B, S+1, D)
        pad = torch.zeros(context_key_padding_mask.shape[0], 1,
                           dtype=torch.bool, device=context_key_padding_mask.device)
        key_padding_mask = torch.cat([context_key_padding_mask, pad], dim=1)
        if need_double_backward:
            # Force the math SDPA backend: the flash/efficient-attention kernels PyTorch
            # picks by default don't implement a backward-of-backward, which the outer
            # loss.backward() needs when this energy call sits inside an
            # autograd.grad(..., create_graph=True) MCMC step (training only -- at eval
            # create_graph=False, a single backward, which flash/efficient support fine
            # and is faster than math, so only pay this cost when actually training).
            # Confirmed empirically on Kaggle T4 -- flash/mem-efficient raised "derivative
            # for aten::_scaled_dot_product_efficient_attention_backward is not implemented".
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                out = self.energy_transformer(combined, src_key_padding_mask=key_padding_mask)
        else:
            out = self.energy_transformer(combined, src_key_padding_mask=key_padding_mask)
        return self.energy_head(out[:, -1, :])  # (B, 1) -- readout at the label-token position

    def forward(self, input_ids, attention_mask, training: bool):
        # BERT's own gradient is only needed to fine-tune it during training; running
        # its forward under grad-tracking at eval/inference time wastes memory and
        # compute for no benefit (confirmed: this was silently inflating the measured
        # inference energy far beyond what the MCMC mechanism itself requires).
        with torch.set_grad_enabled(training):
            context_hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        key_padding_mask = attention_mask == 0  # True = ignore, matches nn.Transformer convention

        batch_size = input_ids.shape[0]
        y = torch.randn(batch_size, self.num_labels, device=input_ids.device,
                         dtype=self.alpha.dtype) * self.init_noise_scale
        alpha = torch.clamp(self.alpha, min=1e-4)

        distributions, energies = [], []
        with torch.set_grad_enabled(True):
            for _ in range(self.mcmc_num_steps):
                y = y.detach().requires_grad_()
                if training and self.langevin_noise != 0:
                    y = y + torch.randn_like(y) * self.langevin_noise
                y_probs = F.softmax(y, dim=-1)
                energy = self._energy(context_hidden, key_padding_mask, y_probs, need_double_backward=training)
                energies.append(energy)
                grad = torch.autograd.grad(energy.sum(), y, create_graph=training)[0]
                y = y - alpha * grad
                distributions.append(F.log_softmax(y, dim=-1))

        return distributions, energies


def ebt_loss(distributions, labels):
    total = distributions[0].new_zeros(())
    for dist in distributions:
        total = total + F.nll_loss(dist, labels)
    return total / len(distributions)


def load_split(dataset: str, split: str) -> Dataset:
    df = pd.read_parquet(os.path.join(P1_RESULTS_DIR, "data", f"{dataset}_{split}.parquet"))
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def make_loader(ds, tokenizer, batch_size, shuffle):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _final_preds(distributions):
    return distributions[-1].argmax(dim=-1)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        distributions, _ = model(input_ids, attention_mask, training=False)
        all_preds.append(_final_preds(distributions).cpu())
        all_labels.append(labels.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": matthews_corrcoef(labels, preds),
    }


def profile_energy_ebt(model, loader, device, n_iters=20):
    """PowerSampler-only energy measurement (no per-component ModuleTimer breakdown --
    that hierarchy is BERT-specific and doesn't apply to this custom architecture).

    Profiles BOTH fp32 and fp16: the P1/P3 baseline's own energy figures were measured
    with the model cast to fp16 (profile_energy.py's main()), while this model trains in
    fp32 (see forward()'s docstring -- double-backward instability under fp16 autocast).
    An earlier fp32-only profiling pass produced a number ~2x the fp16 baseline that
    turned out to barely move after fixing an unrelated eval-time inefficiency (grad
    tracking through BERT) -- so the fp32-vs-fp16 gap itself needed to be measured
    directly rather than assumed, since eval never needs create_graph=True and single-
    order backward is fp16-stable."""
    import time
    model.eval()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    def run(label):
        with PowerSampler() as sampler:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                model(input_ids, attention_mask, training=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
        result = {
            "n_iters": n_iters,
            "batch_size": input_ids.shape[0],
            "total_time_s": total_time,
            "energy_j_total": sampler.energy_joules(),
            "energy_method": sampler.method,
        }
        print(f"[{label}]", result)
        return result

    fp32_result = run("fp32")
    fp16_result = None
    if torch.cuda.is_available():
        model.half()
        fp16_result = run("fp16")
        model.float()  # restore fp32 in case the caller (e.g. checkpoint save) reuses model

    return {"fp32": fp32_result, "fp16": fp16_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liar", choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mcmc_num_steps", type=int, default=3)
    parser.add_argument("--mcmc_step_size", type=float, default=0.1)
    parser.add_argument("--langevin_noise", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--energy_layers", type=int, default=2)
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny run (few examples, 1 epoch) to check the pipeline doesn't crash.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EnergyClassifier(
        mcmc_num_steps=args.mcmc_num_steps,
        mcmc_step_size=args.mcmc_step_size,
        langevin_noise=args.langevin_noise,
        energy_layers=args.energy_layers,
    ).to(device)

    train_ds = load_split(args.dataset, "train")
    val_ds = load_split(args.dataset, "val")
    if args.smoke_test:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        val_ds = val_ds.select(range(min(8, len(val_ds))))
        args.epochs = 1

    batch_size = 4 if args.smoke_test else BATCH_SIZE
    train_loader = make_loader(train_ds, tokenizer, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, tokenizer, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            distributions, _ = model(input_ids, attention_mask, training=True)
            loss = ebt_loss(distributions, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch} done, avg_train_loss={running_loss / (step + 1):.4f}, val={val_metrics}")

    if args.smoke_test:
        print("Smoke test finished without crashing:", val_metrics)
        return

    results_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(results_dir, "ebt_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    profiled = profile_energy_ebt(model, val_loader, device)
    print(profiled)

    with open(os.path.join(results_dir, "ebt_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in ("accuracy", "precision", "recall", "f1", "mcc"):
            writer.writerow([k, val_metrics[k]])
        writer.writerow(["energy_j_total_fp32", profiled["fp32"]["energy_j_total"]])
        writer.writerow(["energy_method_fp32", profiled["fp32"]["energy_method"]])
        if profiled["fp16"]:
            writer.writerow(["energy_j_total_fp16", profiled["fp16"]["energy_j_total"]])
            writer.writerow(["energy_method_fp16", profiled["fp16"]["energy_method"]])
        writer.writerow(["mcmc_num_steps", args.mcmc_num_steps])
        writer.writerow(["precision_note",
                          "training in fp32 (create_graph=True unstable under fp16 autocast); "
                          "eval/profiling measured in both fp32 and fp16, fp16 is the number "
                          "comparable to the P1/P3 baseline's own fp16-profiled energy figures"])
    print(f"Wrote {results_dir}/ebt_metrics.csv")


if __name__ == "__main__":
    main()
