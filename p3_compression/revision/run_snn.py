"""Stage-4 revision driver for SpikeBERT (JournalPaper1 resubmission).

Addresses Reviewer 1 #5/#6 and Reviewer 2 #2/#3/#6b: adapt_spiking.py trains/evaluates
with a single seed, evaluates only on val, and keeps whatever the LAST epoch produces
(no best-epoch checkpointing). Also adds a dense "no-spiking" control (same depths/dim/
embedding/training regime, standard dense self-attention instead of LIF spiking neurons,
no T-timestep repetition) so the paper can separate "spiking computation" from "training
without pretrained language representations" as two different, currently-confounded
causes of SpikeBERT's weak accuracy -- exactly what Reviewer 2 #6b asks for.

Adds, relative to adapt_spiking.py:
  - multi-seed (best val-F1 epoch kept IN MEMORY per seed)
  - held-out TEST-split evaluation
  - --variant spiking|dense (dense = confound-isolation control)
  - repeated-trial energy profiling (spiking variant only -- dense control's energy is
    not a claim this paper makes, it exists purely to isolate the accuracy confound)
  - per-example predictions saved
  - latency (already measured as a profiling byproduct) written to the CSV

Usage:
  python run_snn.py --dataset liar --seed 42 --variant spiking
  python run_snn.py --dataset liar --seed 42 --variant dense
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "snn"))
sys.path.insert(0, os.path.join(P3, "..", "p1_baseline"))

from adapt_spiking import SpikingClassifier, load_split, make_loader, MAX_LENGTH  # noqa: E402
from profile_energy import PowerSampler  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_ENERGY_REPEATS = 5


def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DenseBlock(nn.Module):
    """Confound-isolation control: same width/depth as SpikingBlock, standard dense
    self-attention + MLP instead of LIF spiking neurons, no T-timestep repetition."""

    def __init__(self, dim, heads=8, prenorm=False):
        super().__init__()
        self.prenorm = prenorm
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        if self.prenorm:
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.mlp(x))
        return x


class DenseClassifier(nn.Module):
    """Same from-scratch embedding + depths + dim + no-pretraining regime as
    SpikingClassifier, minus spiking: isolates whether SpikeBERT's accuracy is limited
    by spiking computation itself or by training without pretrained weights."""

    def __init__(self, vocab_size, dim=768, depths=6, num_labels=2, heads=8, prenorm=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([DenseBlock(dim, heads, prenorm) for _ in range(depths)])
        self.last_ln = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_labels)
        self.apply(SpikingClassifier._init_weights)

    def forward(self, input_ids, key_padding_mask=None):
        x = self.emb(input_ids)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.last_ln(x)
        x = x.mean(1)
        return self.classifier(x)


def make_loader_with_mask(ds, tokenizer, batch_size, shuffle):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def score(p1, y):
    pred = (p1 >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=pr, recall=rc, f1=f1,
                mcc=matthews_corrcoef(y, pred), tn=tn, fp=fp, fn=fn, tp=tp)


@torch.no_grad()
def predict_spiking(model, loader, T, num_labels, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids).mean(dim=1)
        probs.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        labels.append(batch["label"].numpy())
        functional.reset_net(model)
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def predict_dense(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        key_padding_mask = batch["attention_mask"].to(device) == 0
        logits = model(input_ids, key_padding_mask=key_padding_mask)
        probs.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def profile_energy_repeated_spiking(model, loader, T, num_labels, device, n_iters=20, n_repeats=N_ENERGY_REPEATS):
    model.eval()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    energies, times = [], []
    for _ in range(n_repeats):
        with PowerSampler() as sampler:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iters):
                    model(input_ids).mean(dim=1)
                    functional.reset_net(model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
        energies.append(sampler.energy_joules())
        times.append(total_time)
    return {
        "n_repeats": n_repeats, "n_iters": n_iters, "batch_size": input_ids.shape[0],
        "energy_j_mean": float(np.mean(energies)), "energy_j_std": float(np.std(energies)),
        "latency_s_per_iter_mean": float(np.mean(times) / n_iters),
        "latency_s_per_iter_std": float(np.std([t / n_iters for t in times])),
        "throughput_examples_per_s": input_ids.shape[0] / (float(np.mean(times)) / n_iters),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["liar", "isot", "fakenewsnet"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", default="spiking", choices=["spiking", "dense"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--depths", type=int, default=6)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--common_thr", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--batch_size", type=int, default=16)  # faster-validated config, see adapt_spiking.py
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--prenorm", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default=os.path.join(P3, "results_rev_altarch"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    set_all_seeds(a.seed)
    out = os.path.join(a.out_dir, "snn", a.variant + a.tag, a.dataset, f"s{a.seed}")
    os.makedirs(out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_ds, val_ds, test_ds = (load_split(a.dataset, s) for s in ("train", "val", "test"))
    if a.smoke:
        train_ds, val_ds, test_ds = (d.select(range(min(16, len(d)))) for d in (train_ds, val_ds, test_ds))
        a.epochs = 1
    batch_size = 4 if a.smoke else a.batch_size

    if a.variant == "spiking":
        model = SpikingClassifier(vocab_size=len(tokenizer), dim=a.dim, depths=a.depths, T=a.T,
                                   tau=a.tau, common_thr=a.common_thr).to(DEV)
        train_loader = make_loader(train_ds, tokenizer, batch_size, shuffle=True)
        val_loader = make_loader(val_ds, tokenizer, batch_size, shuffle=False)
        test_loader = make_loader(test_ds, tokenizer, batch_size, shuffle=False)
    else:
        model = DenseClassifier(vocab_size=len(tokenizer), dim=a.dim, depths=a.depths, prenorm=a.prenorm).to(DEV)
        train_loader = make_loader_with_mask(train_ds, tokenizer, batch_size, shuffle=True)
        val_loader = make_loader_with_mask(val_ds, tokenizer, batch_size, shuffle=False)
        test_loader = make_loader_with_mask(test_ds, tokenizer, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=0)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best = {"f1": -1.0, "state": None, "epoch": None}

    t_train0 = time.time()
    base_lr, gstep = a.lr, 0
    for epoch in range(a.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEV)
            labels = batch["label"].to(DEV)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if a.variant == "spiking":
                    logits = model(input_ids).mean(dim=1)
                else:
                    key_padding_mask = batch["attention_mask"].to(DEV) == 0
                    logits = model(input_ids, key_padding_mask=key_padding_mask)
                loss = F.cross_entropy(logits, labels)
            gstep += 1
            if a.warmup_steps:
                for g in optimizer.param_groups:
                    g["lr"] = base_lr * min(1.0, gstep / a.warmup_steps)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if a.variant == "spiking":
                functional.reset_net(model)
            running += loss.item()
            if step % 50 == 0:
                print(f"[{a.variant}] epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
        scheduler.step()
        base_lr = scheduler.get_last_lr()[0]

        if a.variant == "spiking":
            p1, y = predict_spiking(model, val_loader, a.T, 2, DEV)
        else:
            p1, y = predict_dense(model, val_loader, DEV)
        val_metrics = score(p1, y)
        print(f"[{a.variant}] epoch {epoch} val f1={val_metrics['f1']:.4f} mcc={val_metrics['mcc']:.4f}", flush=True)
        if val_metrics["f1"] > best["f1"]:
            best.update(f1=val_metrics["f1"], epoch=epoch,
                         state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best["state"])
    train_time_s = time.time() - t_train0
    print(f"best epoch {best['epoch']} val F1 {best['f1']:.4f}; trained in {train_time_s:.0f}s", flush=True)

    rows = []
    for split, loader in (("val", val_loader), ("test", test_loader)):
        if a.variant == "spiking":
            p1, y = predict_spiking(model, loader, a.T, 2, DEV)
        else:
            p1, y = predict_dense(model, loader, DEV)
        m = score(p1, y)
        m.update(split=split, dataset=a.dataset, seed=a.seed, variant=a.variant, tag=a.tag,
                  best_epoch=best["epoch"], train_time_s=train_time_s, n=len(y))
        rows.append(m)
        pd.DataFrame({"row": np.arange(len(y)), "label": y, "pred": (p1 >= 0.5).astype(int), "p_true": p1}).to_csv(
            os.path.join(out, f"preds_{split}.csv"), index=False)

    if not a.smoke and a.variant == "spiking":
        profiled = profile_energy_repeated_spiking(model, val_loader, a.T, 2, DEV)
        for r in rows:
            r.update({f"profile_{k}": v for k, v in profiled.items()})

    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False)
    print(f"wrote {out}/metrics.csv")


if __name__ == "__main__":
    main()
