"""Stage-4 revision driver for EBT (JournalPaper1 resubmission).

Addresses Reviewer 1 #4/#6 and Reviewer 2 #2/#3/#6a: EBT in adapt_classification.py
trains/evaluates with a single seed, evaluates only on val, and keeps whatever the
LAST epoch produces (no best-epoch checkpointing) -- the same category of gap already
fixed for the Trainer-based jobs in run_seed.py, applied here to EBT's own hand-rolled
training loop (EBT can't use HF Trainer: its forward pass needs
torch.autograd.grad(create_graph=True) through the MCMC refinement steps).

Adds, relative to adapt_classification.py:
  - multi-seed (best val-F1 epoch kept IN MEMORY per seed, not just the final epoch)
  - held-out TEST-split evaluation (not just val)
  - repeated-trial energy profiling (N_ENERGY_REPEATS passes of the existing fixed
    20-iteration window), reporting mean/SD instead of a single estimate
  - refinement-step-count ablation support (--mcmc_num_steps already existed as a CLI
    flag in adapt_classification.py; this script just makes it easy to sweep)
  - per-example predictions saved (for McNemar / error analysis, matching run_seed.py)
  - latency (already measured as a byproduct of energy profiling) written to the CSV

Usage:
  python run_ebt.py --dataset liar --seed 42
  python run_ebt.py --dataset liar --seed 42 --mcmc_num_steps 1   # refinement-step ablation
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "ebt"))
sys.path.insert(0, os.path.join(P3, "..", "p1_baseline"))

from config import BATCH_SIZE, MAX_SEQ_LEN  # noqa: E402
from adapt_classification import EnergyClassifier, ebt_loss, make_loader, load_split  # noqa: E402
from profile_energy import PowerSampler  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_ENERGY_REPEATS = 5  # Reviewer 2 #1: report variance across repeated profiling windows, not one estimate


def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        distributions, _ = model(input_ids, attention_mask, training=False)
        p = distributions[-1].exp()[:, 1]  # log_softmax -> prob of class 1
        probs.append(p.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def score(p1, y):
    pred = (p1 >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=pr, recall=rc, f1=f1,
                mcc=matthews_corrcoef(y, pred), tn=tn, fp=fp, fn=fn, tp=tp)


def energy_descent_check(model, loader, device, out_path):
    """Reviewer 1 #4 / Reviewer 2 #6a: verify the learned energy actually decreases across
    the 3 (or T) refinement steps on real held-out examples, rather than asserting it."""
    model.eval()
    batch = next(iter(loader))
    input_ids = batch["input_ids"][:8].to(device)
    attention_mask = batch["attention_mask"][:8].to(device)
    with torch.no_grad():
        context_hidden = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    key_padding_mask = attention_mask == 0
    y = torch.randn(input_ids.shape[0], model.num_labels, device=device) * model.init_noise_scale
    alpha = torch.clamp(model.alpha, min=1e-4)
    rows = []
    with torch.set_grad_enabled(True):
        for step in range(model.mcmc_num_steps):
            y = y.detach().requires_grad_()
            y_probs = torch.softmax(y, dim=-1)
            energy = model._energy(context_hidden, key_padding_mask, y_probs, need_double_backward=False)
            rows.append(energy.detach().mean().item())
            grad = torch.autograd.grad(energy.sum(), y, create_graph=False)[0]
            y = y - alpha * grad
    monotonic = all(rows[i + 1] <= rows[i] for i in range(len(rows) - 1))
    pd.DataFrame({"step": range(len(rows)), "mean_energy": rows}).to_csv(out_path, index=False)
    print(f"[energy-descent-check] per-step mean energy: {rows}; monotonic={monotonic}")
    return monotonic, rows


def profile_energy_repeated(model, loader, device, n_iters=20, n_repeats=N_ENERGY_REPEATS):
    model.eval()
    model.half()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    energies, times = [], []
    for _ in range(n_repeats):
        with PowerSampler() as sampler:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                model(input_ids, attention_mask, training=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
        energies.append(sampler.energy_joules())
        times.append(total_time)
    model.float()
    return {
        "n_repeats": n_repeats, "n_iters": n_iters, "batch_size": input_ids.shape[0],
        "energy_j_mean": float(np.mean(energies)), "energy_j_std": float(np.std(energies)),
        "energy_j_values": energies,
        "latency_s_per_iter_mean": float(np.mean(times) / n_iters),
        "latency_s_per_iter_std": float(np.std([t / n_iters for t in times])),
        "throughput_examples_per_s": input_ids.shape[0] / (float(np.mean(times)) / n_iters),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["liar", "isot", "fakenewsnet"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mcmc_num_steps", type=int, default=3)
    ap.add_argument("--mcmc_step_size", type=float, default=0.1)
    ap.add_argument("--langevin_noise", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--energy_layers", type=int, default=2)
    ap.add_argument("--out_dir", default=os.path.join(P3, "results_rev_altarch"))
    ap.add_argument("--n_draws", type=int, default=0, help="extra stochastic test evaluations (random initial logits) plus one zero-initialized evaluation")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    set_all_seeds(a.seed)
    tag = f"T{a.mcmc_num_steps}" if a.mcmc_num_steps != 3 else "s"
    out = os.path.join(a.out_dir, "ebt", a.dataset, f"{tag}{a.seed}")
    os.makedirs(out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = EnergyClassifier(mcmc_num_steps=a.mcmc_num_steps, mcmc_step_size=a.mcmc_step_size,
                              langevin_noise=a.langevin_noise, energy_layers=a.energy_layers).to(DEV)

    train_ds, val_ds, test_ds = (load_split(a.dataset, s) for s in ("train", "val", "test"))
    if a.smoke:
        train_ds, val_ds, test_ds = (d.select(range(min(16, len(d)))) for d in (train_ds, val_ds, test_ds))
        a.epochs = 1
    batch_size = 4 if a.smoke else BATCH_SIZE
    train_loader = make_loader(train_ds, tokenizer, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, tokenizer, batch_size, shuffle=False)
    test_loader = make_loader(test_ds, tokenizer, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    best = {"f1": -1.0, "state": None, "epoch": None}

    t_train0 = time.time()
    _ps = PowerSampler()
    _ps.__enter__()
    for epoch in range(a.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEV)
            attention_mask = batch["attention_mask"].to(DEV)
            labels = batch["label"].to(DEV)
            distributions, _ = model(input_ids, attention_mask, training=True)
            loss = ebt_loss(distributions, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)

        p1, y = predict(model, val_loader, DEV)
        val_metrics = score(p1, y)
        print(f"epoch {epoch} val f1={val_metrics['f1']:.4f} mcc={val_metrics['mcc']:.4f}", flush=True)
        if val_metrics["f1"] > best["f1"]:
            best.update(f1=val_metrics["f1"], epoch=epoch,
                         state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    _ps.__exit__(None, None, None)
    train_energy_j = _ps.energy_joules()
    model.load_state_dict(best["state"])
    train_time_s = time.time() - t_train0
    print(f"training NVML energy {train_energy_j:.0f} J over {train_time_s:.0f} s (including per-epoch validation)", flush=True)
    print(f"best epoch {best['epoch']} val F1 {best['f1']:.4f}; trained in {train_time_s:.0f}s", flush=True)

    rows = []
    for split, loader in (("val", val_loader), ("test", test_loader)):
        p1, y = predict(model, loader, DEV)
        m = score(p1, y)
        m.update(split=split, dataset=a.dataset, seed=a.seed, mcmc_num_steps=a.mcmc_num_steps,
                  best_epoch=best["epoch"], train_time_s=train_time_s, train_energy_j=train_energy_j, n=len(y))
        rows.append(m)
        pd.DataFrame({"row": np.arange(len(y)), "label": y, "pred": (p1 >= 0.5).astype(int), "p_true": p1}).to_csv(
            os.path.join(out, f"preds_{split}.csv"), index=False)

    if a.n_draws > 0:
        draws, ys = [], None
        for _ in range(a.n_draws):
            p1, ys = predict(model, test_loader, DEV)
            draws.append(p1)
        D = np.stack(draws)
        preds = (D >= 0.5).astype(int)
        base = model.init_noise_scale
        model.init_noise_scale = 0.0
        pz, _ = predict(model, test_loader, DEV)
        model.init_noise_scale = base
        vote = (preds.mean(axis=0) >= 0.5).astype(int)
        rows_d = [dict(draw=i, **{k: v for k, v in score(D[i], ys).items() if k in ("accuracy", "mcc")}) for i in range(a.n_draws)]
        rows_d.append(dict(draw="zero_init", **{k: v for k, v in score(pz, ys).items() if k in ("accuracy", "mcc")}))
        rows_d.append(dict(draw="majority_vote", accuracy=float((vote == ys).mean()), mcc=score(vote.astype(float), ys)["mcc"]))
        flip = float((preds.min(axis=0) != preds.max(axis=0)).mean())
        pd.DataFrame(rows_d).assign(example_flip_rate=flip).to_csv(os.path.join(out, "draws.csv"), index=False)
        print(f"[draws] {a.n_draws} draws: MCC mean {np.mean([r['mcc'] for r in rows_d[:a.n_draws]]):.4f} SD {np.std([r['mcc'] for r in rows_d[:a.n_draws]]):.4f}; "
              f"share of test examples whose prediction changes across draws {flip:.4f}", flush=True)

    if not a.smoke:
        monotonic, energy_steps = energy_descent_check(model, val_loader, DEV,
                                                         os.path.join(out, "energy_descent.csv"))
        for r in rows:
            r["energy_descends_monotonically"] = monotonic
        profiled = profile_energy_repeated(model, val_loader, DEV)
        for r in rows:
            r.update({f"profile_{k}": v for k, v in profiled.items() if k != "energy_j_values"})

    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False)
    print(f"wrote {out}/metrics.csv")


if __name__ == "__main__":
    main()
