"""Charformer / GBST (Tay et al. 2021, "Charformer: Fast Character Transformers via
Gradient-based Subword Tokenization", arXiv:2106.12672) adapted to text classification
on LIAR.

M13 of the exploratory (beyond-P1-P4) scope. GBST (Gradient-Based Subword Tokenization)
is ported near-verbatim from lucidrains/charformer-pytorch's GBST module
(charformer_pytorch.py, github.com/lucidrains/charformer-pytorch), read directly before
writing anything here: raw byte-level token embeddings are pooled over a learned soft
mixture of candidate block sizes (1..max_block_size) at every position, weighted by a
softmax score (optionally smoothed by a lightweight "consensus attention" over the score
vectors themselves, eq. 5 in the paper) -- then the sequence is downsampled by a fixed
factor via mean pooling. The output is a shorter sequence of continuous embeddings fed
into a standard Transformer encoder. This is architecturally the most different of the
three linear/efficient-attention explorations (M11 Linformer, M12 Performer): those
swap only the attention SCORE computation inside an otherwise-unchanged BERT; GBST
replaces the INPUT representation itself (raw UTF-8 bytes, not WordPiece tokens).

Deliberate scope decision that overrides the M11/M12 precedent, stated explicitly:

- FROM SCRATCH, NOT a retrofit onto the P1 fine-tuned checkpoint. Linformer/Performer
  could reuse bert-base-uncased's pretrained query/key/value weights unchanged because
  the input to attention (BERT's own hidden states) was untouched -- only how attention
  scores get computed changed. GBST changes what the *input* even is: a byte-level
  nn.Embedding(256, dim) has no correspondence to BERT's WordPiece nn.Embedding(30522,
  768), and no pretrained encoder layer has ever seen the resulting GBST-pooled
  representations. There is no coherent way to "inherit" weights across that boundary --
  attempting it would silently test nothing meaningful. Trained entirely from scratch,
  same discipline as EBT (M9) and SNN (M10).
- Deliberately smaller than BERT-base (dim=256, 4 layers, 4 heads vs 768/12/12): a
  from-scratch transformer fine-tuned for only a handful of epochs on LIAR's ~10k
  training examples cannot be expected to reach BERT-base-scale capacity usefully --
  Charformer's own paper pretrains at C4-corpus scale over many steps, far outside this
  project's single-Kaggle-run time-box. This tests "does a from-scratch byte-level GBST
  classifier train and predict at all on fake-news detection" (a real gap: SpikeBERT/
  SpikingBERT and Charformer itself were only ever evaluated on GLUE-style benchmarks,
  never FND), not "does Charformer match BERT-base accuracy" -- a narrower, honestly
  bounded claim, same framing discipline as SNN's own scope note.
- Byte-level sequence length (BYTE_MAX_LEN=1024) is NOT the fixed-protocol MAX_SEQ_LEN
  (256 WordPiece tokens) -- bytes and WordPiece tokens are different units, so the same
  numeric length would not represent the same amount of text. 1024 bytes with
  downsample_factor=4 lands the Transformer encoder's own input length at 256,
  matching the fixed protocol's sequence length at the point where it's actually
  comparable (the encoder's working sequence length), even though the raw input length
  differs.
- Higher learning rate (1e-4 vs the fixed protocol's 2e-5) and more epochs (5 vs 3):
  2e-5 is a fine-tuning rate for an already-pretrained model; from-scratch training of a
  freshly-initialized transformer needs a substantially higher rate to make progress in
  a handful of epochs. This is standard practice for training transformers from
  scratch, not a tuned/swept choice.
- Trains in plain fp32 (no autocast/GradScaler), unlike Linformer/Performer's use of
  HF Trainer's fp16=True: not a stability requirement here (no double-backward, no
  EBT-style exotica), just that this file's own hand-written training loop never wraps
  the forward pass in autocast. Only the post-training profiling step casts the model
  to fp16 (model.half()) to stay comparable with the fixed protocol's own fp16-profiled
  energy figures. This surfaced a real bug during development: masked_mean's division
  (mask.sum() is int64; fp16-tensor / int64-tensor promotes to fp32 under PyTorch's
  type-promotion rules) silently produced fp32 activations that crashed the first
  post-training fp16 forward pass with a Float/Half dtype mismatch inside score_fn's
  Linear -- invisible during fp32 training, since fp32/int64 already promotes to fp32.
  Fixed by casting mask.sum() to the input tensor's own dtype before dividing, in both
  masked_mean and the classifier's own final pooling step.

Time-boxed single go/no-go attempt, matching M9/M10/M11/M12's discipline: if this
doesn't train to a usable, evaluable classifier within one Kaggle run, "evaluated,
found impractical to adapt within scope" is the citable finding, stated plainly in
results/liar/charformer_metrics.csv, not forced.

Usage: python adapt_charformer.py --dataset liar
"""

import argparse
import csv
import math
from math import gcd
import functools
import os
import sys
import time

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "p1_baseline"))
from config import BATCH_SIZE, SEED  # noqa: E402
from profile_energy import PowerSampler  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
NUM_LABELS = 2
NUM_BYTE_TOKENS = 256  # raw UTF-8 bytes, tokenization-free

# --- GBST, ported from lucidrains/charformer-pytorch (verified against source before
# writing this file, see module docstring) -----------------------------------------


def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))


def masked_mean(tensor, mask, dim=-1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor = tensor.masked_fill(~mask, 0.0)
    # mask.sum() is int64; dividing a fp16 tensor by an int64 tensor promotes the result
    # to fp32 regardless of tensor's own dtype (PyTorch type-promotion rule for tensor/
    # tensor division, unlike a Python scalar which would respect tensor's dtype) --
    # silently breaks dtype-uniformity the moment the model is cast to fp16 (confirmed:
    # this crashed profile_energy_charformer's post-training model.half() step with
    # "mat1 and mat2 must have the same dtype, but got Float and Half", while training
    # itself, run in fp32 throughout, never surfaced it since fp32/int64 already -> fp32).
    total_el = mask.sum(dim=dim).to(tensor.dtype)
    mean = tensor.sum(dim=dim) / total_el.clamp(min=1.0)
    return mean.masked_fill(total_el == 0, 0.0)


def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple


def pad_to_multiple(tensor, multiple, *, seq_dim, dim=-1, value=0.0):
    seqlen = tensor.shape[seq_dim]
    length = next_divisible_length(seqlen, multiple)
    if length == seqlen:
        return tensor
    remainder = length - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


class Pad(nn.Module):
    def __init__(self, padding, value=0.0):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, value=self.value)


class DepthwiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, groups=dim_in)
        self.proj_out = nn.Conv1d(dim_out, dim_out, 1)

    def forward(self, x):
        return self.proj_out(self.conv(x))


class GBST(nn.Module):
    def __init__(self, num_tokens, dim, max_block_size=4, downsample_factor=4, score_consensus_attn=True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.blocks = tuple((size, 0) for size in range(1, max_block_size + 1))
        self.pos_conv = nn.Sequential(
            Pad((0, 0, 0, max_block_size - 1)),
            Rearrange("b n d -> b d n"),
            DepthwiseConv1d(dim, dim, kernel_size=max_block_size),
            Rearrange("b d n -> b n d"),
        )
        self.score_fn = nn.Sequential(nn.Linear(dim, 1), Rearrange("... () -> ..."))
        self.score_consensus_attn = score_consensus_attn
        self.block_pad_multiple = lcm(*[size for size, _ in self.blocks])
        self.downsample_factor = downsample_factor

    def forward(self, x, mask=None):
        b, n, block_mult, ds_factor = *x.shape, self.block_pad_multiple, self.downsample_factor
        m = next_divisible_length(n, ds_factor)

        x = self.token_emb(x)
        x = self.pos_conv(x)
        x = pad_to_multiple(x, block_mult, seq_dim=1, dim=-2)
        if mask is not None:
            mask = pad_to_multiple(mask, block_mult, seq_dim=1, dim=-1, value=False)

        block_masks, block_reprs = [], []
        for block_size, offset in self.blocks:
            blocks = rearrange(x, "b (n m) d -> b n m d", m=block_size)
            if mask is not None:
                mask_blocks = rearrange(mask, "b (n m) -> b n m", m=block_size)
                block_repr = masked_mean(blocks, mask_blocks, dim=-2)
            else:
                block_repr = blocks.mean(dim=-2)
            block_repr = repeat(block_repr, "b n d -> b (n m) d", m=block_size)
            block_reprs.append(block_repr)
            if mask is not None:
                block_masks.append(repeat(torch.any(mask_blocks, dim=-1), "b n -> b (n m)", m=block_size))

        block_reprs = torch.stack(block_reprs, dim=2)
        scores = self.score_fn(block_reprs)
        if mask is not None:
            block_masks = torch.stack(block_masks, dim=2)
            scores = scores.masked_fill(~block_masks, -torch.finfo(scores.dtype).max)
        scores = scores.softmax(dim=2)

        if self.score_consensus_attn:
            score_sim = torch.einsum("b i d, b j d -> b i j", scores, scores)
            if mask is not None:
                cross_mask = rearrange(mask, "b i -> b i ()") * rearrange(mask, "b j -> b () j")
                score_sim = score_sim.masked_fill(~cross_mask, -torch.finfo(score_sim.dtype).max)
            scores = torch.einsum("b i j, b j m -> b i m", score_sim.softmax(dim=-1), scores)

        x = (block_reprs * rearrange(scores, "b n m -> b n m ()")).sum(dim=2)
        x = x[:, :m]
        if mask is not None:
            mask = mask[:, :m]

        x = rearrange(x, "b (n m) d -> b n m d", m=ds_factor)
        if mask is not None:
            mask = rearrange(mask, "b (n m) -> b n m", m=ds_factor)
            x = masked_mean(x, mask, dim=2)
            mask = torch.any(mask, dim=-1)
        else:
            x = x.mean(dim=-2)
        return x, mask


# --- classifier on top of GBST (this project's own code, not from the reference repo,
# which only demonstrates GBST + a generic transformer, no classification head) -------


class CharformerClassifier(nn.Module):
    def __init__(self, dim=256, num_layers=4, nhead=4, gbst_max_block_size=4,
                 downsample_factor=4, num_labels=NUM_LABELS):
        super().__init__()
        self.gbst = GBST(NUM_BYTE_TOKENS, dim, max_block_size=gbst_max_block_size,
                          downsample_factor=downsample_factor)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4,
                                            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim, num_labels)

    def forward(self, byte_ids, byte_mask):
        x, mask = self.gbst(byte_ids, mask=byte_mask)
        x = self.encoder(x, src_key_padding_mask=~mask)
        # hardcoding .float() here would force fp32 activations into self.classifier
        # even when the model has been cast to fp16 (profiling) -- match x's own dtype
        # instead, same fix as masked_mean() above.
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask_f).sum(dim=1) / denom
        return self.classifier(pooled)


def encode_bytes(texts, max_len):
    ids = torch.zeros(len(texts), max_len, dtype=torch.long)
    mask = torch.zeros(len(texts), max_len, dtype=torch.bool)
    for i, text in enumerate(texts):
        b = text.encode("utf-8", errors="ignore")[:max_len]
        if b:
            ids[i, :len(b)] = torch.tensor(list(b), dtype=torch.long)
            mask[i, :len(b)] = True
    return ids, mask


def load_split_raw(dataset: str, split: str):
    df = pd.read_parquet(os.path.join(P1_RESULTS_DIR, "data", f"{dataset}_{split}.parquet"))
    return df["text"].tolist(), df["label"].tolist()


def make_loader(texts, labels, max_len, batch_size, shuffle):
    ids, mask = encode_bytes(texts, max_len)
    ds = TensorDataset(ids, mask, torch.tensor(labels, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids, mask, labels in loader:
            logits = model(ids.to(device), mask.to(device))
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels)
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


def profile_energy_charformer(model, loader, device, n_iters=20):
    model.eval()
    ids, mask, _ = next(iter(loader))
    ids, mask = ids.to(device), mask.to(device)
    with PowerSampler() as sampler:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(ids, mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
    return {
        "n_iters": n_iters,
        "batch_size": ids.shape[0],
        "total_time_s": total_time,
        "energy_j_total": sampler.energy_joules(),
        "energy_method": sampler.method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liar", choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--byte_max_len", type=int, default=1024)
    parser.add_argument("--gbst_max_block_size", type=int, default=4)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny run (few examples, 1 epoch) to check the pipeline doesn't crash.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_texts, train_labels = load_split_raw(args.dataset, "train")
    val_texts, val_labels = load_split_raw(args.dataset, "val")
    if args.smoke_test:
        train_texts, train_labels = train_texts[:8], train_labels[:8]
        val_texts, val_labels = val_texts[:8], val_labels[:8]
        args.epochs = 1

    batch_size = 4 if args.smoke_test else BATCH_SIZE
    train_loader = make_loader(train_texts, train_labels, args.byte_max_len, batch_size, shuffle=True)
    val_loader = make_loader(val_texts, val_labels, args.byte_max_len, batch_size, shuffle=False)

    model = CharformerClassifier(dim=args.dim, num_layers=args.num_layers, nhead=args.nhead,
                                  gbst_max_block_size=args.gbst_max_block_size,
                                  downsample_factor=args.downsample_factor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    curve_rows = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, (ids, mask, labels) in enumerate(train_loader):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        avg_loss = running_loss / (step + 1)
        print(f"epoch {epoch} done, avg_train_loss={avg_loss:.4f}, val={val_metrics}")
        curve_rows.append({"epoch": epoch, "avg_train_loss": avg_loss, **val_metrics})

    if args.smoke_test:
        print("Smoke test finished without crashing:", val_metrics)
        return

    results_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "charformer_training_curve.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "avg_train_loss", "accuracy", "precision",
                                                 "recall", "f1", "mcc"])
        writer.writeheader()
        writer.writerows(curve_rows)
    print(f"Wrote {results_dir}/charformer_training_curve.csv")

    if torch.cuda.is_available():
        model.half()
    profiled = profile_energy_charformer(model, val_loader, device)
    print(profiled)

    ckpt_path = os.path.join(results_dir, "charformer_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    with open(os.path.join(results_dir, "charformer_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in ("accuracy", "precision", "recall", "f1", "mcc"):
            writer.writerow([k, val_metrics[k]])
        writer.writerow(["energy_j_total", profiled["energy_j_total"]])
        writer.writerow(["energy_method", profiled["energy_method"]])
        writer.writerow(["dim", args.dim])
        writer.writerow(["num_layers", args.num_layers])
        writer.writerow(["byte_max_len", args.byte_max_len])
        writer.writerow(["downsample_factor", args.downsample_factor])
        writer.writerow(["encoder_seq_len", args.byte_max_len // args.downsample_factor])
        writer.writerow(["epochs", args.epochs])
        writer.writerow(["lr", args.lr])
        writer.writerow(["note",
                          "trained FROM SCRATCH (not a retrofit of the P1 fine-tuned checkpoint -- "
                          "byte-level GBST input has no correspondence to BERT's WordPiece embedding "
                          "space, see module docstring); smaller than BERT-base (dim=256/4 layers vs "
                          "768/12) and higher LR (1e-4 vs 2e-5) by necessity for from-scratch training "
                          "within a single Kaggle time-box -- not directly accuracy-comparable to the "
                          "P1 baseline or the retrofit-based Linformer/Performer results"])
    print(f"Wrote {results_dir}/charformer_metrics.csv")


if __name__ == "__main__":
    main()
