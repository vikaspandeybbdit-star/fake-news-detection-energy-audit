"""Stage-4 revision driver (JournalPaper2): one dataset, one seed, any subset of jobs.

Addresses the Stage-3 review: multiple seeds, held-out TEST-split evaluation, saved
per-example predictions (for McNemar / confusion matrices), an accuracy check of the
pure-FP16 model actually used for energy, and a mean-attention control.

Jobs (run in this order when listed together):
  bert       fine-tune BERT-base (3 epochs, best val-F1 epoch kept)        -> parent checkpoint
  linformer  in-place Linformer retrofit of the parent (k=64), 3 epochs
  performer  in-place Performer retrofit of the parent (m=64), 3 epochs
  meanattn   control: attention replaced by a uniform masked mean of V, 3 epochs
  grid       pruning {0,25,50,75}% x distillation {no,yes}; the DistilBERT student is
             distilled ONCE from the parent and copied for each pruning ratio
             (earlier runs re-distilled per cell); each pruned model gets 1 recovery epoch

Every trained model is evaluated on val and test, with mixed-precision (autocast) and,
for test, also after .half() (the precision used in the energy session).

Reuses: prune.prune_layer_heads/heads_to_prune_by_magnitude,
combine.prune_distilbert_heads, distill.DistillationTrainer,
adapt_linformer.linformerize, adapt_performer.performerize.

Usage: python run_seed.py --dataset liar --seed 42 --jobs bert,linformer,performer,meanattn,grid [--smoke]
"""
import argparse
import copy
import csv
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd  # before torch (Windows pyarrow/torch DLL clash)
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
for p in (os.path.join(P3, "..", "p1_baseline"), P3, os.path.join(P3, "combined_framework"),
          os.path.join(P3, "linear_attention", "linformer"), os.path.join(P3, "linear_attention", "performer")):
    sys.path.insert(0, p)

from config import BATCH_SIZE, MAX_SEQ_LEN  # noqa: E402
from train import load_split, compute_metrics  # noqa: E402
from prune import heads_to_prune_by_magnitude, prune_layer_heads  # noqa: E402
from combine import prune_distilbert_heads  # noqa: E402
from distill import DistillationTrainer, STUDENT_MODEL  # noqa: E402
from adapt_linformer import linformerize, _PadMaskBox  # noqa: E402
from adapt_performer import performerize  # noqa: E402

BERT = "bert-base-uncased"
TINY_STUDENT = "huawei-noah/TinyBERT_General_4L_312D"  # Jiao et al. 2020 task-specific distillation stage
DATA_DIR = os.path.join(P3, "..", "p1_baseline", "results", "data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MeanSelfAttention(nn.Module):
    """Control: every position receives the masked mean of the value vectors, i.e. uniform
    attention. Keeps the fine-tuned value projection; query/key are unused."""

    def __init__(self, orig, box: _PadMaskBox):
        super().__init__()
        self.value = orig.value
        self.all_head_size = orig.all_head_size
        self.box = box

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        v = self.value(hidden_states)  # (B,S,D)
        pad = self.box.mask
        keep = (~pad).unsqueeze(-1).to(v.dtype) if pad is not None else torch.ones_like(v[..., :1])
        mean = (v * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp(min=1.0)
        return mean.expand_as(v).contiguous(), None


def meanattnize(model):
    box = _PadMaskBox()

    def capture(module, args, kwargs):
        am = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        if am is not None:
            box.mask = am == 0

    model.register_forward_pre_hook(capture, with_kwargs=True)
    for layer in model.bert.encoder.layer:
        layer.attention.self = MeanSelfAttention(layer.attention.self, box)


class Run:
    def __init__(self, a):
        self.a = a
        self.out = os.path.join(a.out_dir, a.dataset, f"s{a.seed}")
        self.work = os.path.join(a.work_dir, f"{a.dataset}_s{a.seed}")
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.work, exist_ok=True)
        self.metrics_path = os.path.join(self.out, "metrics.csv")

    def data(self, tok, split):
        ds = load_split(self.a.dataset, split)
        if self.a.smoke:
            ds = ds.select(range(min(16, len(ds))))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length"),
                      batched=True)

    def train(self, model, tok, name, epochs, trainer_cls=Trainer, **extra):
        """Best-val-F1 epoch kept IN MEMORY and restored with strict=True. Trainer's own
        load_best_model_at_end reloads a saved checkpoint non-strictly; with BERT's legacy
        LayerNorm key names (gamma/beta) that silently kept the LAST epoch's LayerNorm."""
        set_seed(self.a.seed)
        out_dir = os.path.join(self.work, name)
        args = TrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=1 if self.a.smoke else epochs, eval_strategy="epoch", save_strategy="no",
            fp16=torch.cuda.is_available(), seed=self.a.seed, logging_steps=50, report_to="none")
        best = {"f1": -1.0, "state": None, "epoch": None}

        class KeepBest(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kw):
                f1 = metrics.get("eval_f1", -1.0)
                if f1 > best["f1"]:
                    best.update(f1=f1, epoch=state.epoch,
                                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        t0 = time.time()
        trainer_cls(model=model, args=args, train_dataset=self.data(tok, "train"), eval_dataset=self.data(tok, "val"),
                    compute_metrics=compute_metrics, callbacks=[KeepBest()], **extra).train()
        model.load_state_dict(best["state"], strict=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"[{name}] trained in {time.time() - t0:.0f}s; best epoch {best['epoch']} val F1 {best['f1']:.4f}", flush=True)
        return model

    def evaluate(self, model, tok, name, parent):
        df_by = {s: pd.read_parquet(os.path.join(DATA_DIR, f"{self.a.dataset}_{s}.parquet")) for s in ("val", "test")}
        model.eval()
        modes = [("val", "amp"), ("test", "amp")] + ([("test", "fp16")] if torch.cuda.is_available() else [])
        for split, mode in modes:
            df = df_by[split].iloc[:16] if self.a.smoke else df_by[split]
            # fp16 pass IN PLACE (deepcopy would detach retrofits from their pad-mask hook);
            # exact weights restored afterwards
            saved = {k: v.detach().clone() for k, v in model.state_dict().items()} if mode == "fp16" else None
            m = model.half() if mode == "fp16" else model
            probs = []
            with torch.no_grad():
                for i in range(0, len(df), BATCH_SIZE):
                    enc = tok(df["text"].iloc[i:i + BATCH_SIZE].tolist(), truncation=True, max_length=MAX_SEQ_LEN,
                              padding="max_length", return_tensors="pt").to(DEV)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=(mode == "amp" and DEV == "cuda")):
                        logits = m(**enc).logits.float()
                    probs.append(logits.softmax(-1)[:, 1].cpu().numpy())
            p1 = np.concatenate(probs)
            y = df["label"].to_numpy()
            pred = (p1 >= 0.5).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            row = dict(model=name, parent=parent, split=split, precision_mode=mode, seed=self.a.seed,
                       dataset=self.a.dataset, n=len(y), accuracy=accuracy_score(y, pred), precision=pr, recall=rc,
                       f1=f1, mcc=matthews_corrcoef(y, pred), tn=tn, fp=fp, fn=fn, tp=tp,
                       n_params=sum(p.numel() for p in model.parameters()))
            new = not os.path.exists(self.metrics_path)
            with open(self.metrics_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            if mode == "amp":
                pd.DataFrame({"row": np.arange(len(y)), "label": y, "pred": pred, "p_true": p1}).to_csv(
                    os.path.join(self.out, f"preds_{name}_{split}.csv"), index=False)
            print(f"[{name}] {split}/{mode}: acc={row['accuracy']:.4f} mcc={row['mcc']:.4f}", flush=True)
            if saved is not None:
                model.float()
                model.load_state_dict(saved, strict=True)
                del saved
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- jobs
    def parent(self):
        """Fresh copy of the fine-tuned parent: config + raw state_dict, loaded strictly
        (no checkpoint key renaming involved)."""
        path = os.path.join(self.work, "parent.pt")
        if not os.path.exists(path):
            raise RuntimeError("run the 'bert' job first in this session")
        model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(BERT, num_labels=2))
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        return model, AutoTokenizer.from_pretrained(BERT)

    def job_bert(self):
        set_seed(self.a.seed)
        tok = AutoTokenizer.from_pretrained(BERT)
        model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=2)
        ln = model.bert.embeddings.LayerNorm.weight
        assert not torch.allclose(ln, torch.ones_like(ln)), "pretrained LayerNorm not loaded (transformers>=5?)"
        model = self.train(model.to(DEV), tok, "bert", 3)
        torch.save(model.state_dict(), os.path.join(self.work, "parent.pt"))
        self.evaluate(model, tok, "bert_base", "pretrained")

    def _retrofit(self, name, fn):
        model, tok = self.parent()
        fn(model)
        model = model.to(DEV)  # after replacement, so new modules move too
        model = self.train(model, tok, name, 3)
        self.evaluate(model, tok, name, "bert_base")

    def job_linformer(self):
        self._retrofit("linformer", lambda m: linformerize(m, proj_k=64, max_seq_len=MAX_SEQ_LEN))

    def job_performer(self):
        self._retrofit("performer", lambda m: performerize(m, nb_features=64))

    def job_meanattn(self):
        self._retrofit("meanattn", meanattnize)

    def job_tinybert(self):
        """Second published-architecture baseline (Reviewer 1 #2 / Reviewer 2 #5): same
        fixed protocol, same teacher, same train()/evaluate() harness as the DistilBERT
        student in job_grid, swapped to TinyBERT-4L-312D so the comparison in the paper
        isn't limited to one published framework (Rout et al.) under different settings."""
        teacher, _ = self.parent()
        teacher = teacher.to(DEV).eval()
        stok = AutoTokenizer.from_pretrained(TINY_STUDENT)
        set_seed(self.a.seed)
        student = AutoModelForSequenceClassification.from_pretrained(TINY_STUDENT, num_labels=2).to(DEV)
        student = self.train(student, stok, "tinybert", 3, trainer_cls=DistillationTrainer, teacher_model=teacher)
        del teacher
        self.evaluate(student, stok, "tinybert", "bert_base")

    def job_quant(self):
        """Dynamic INT8 quantization (CPU) of the fine-tuned parent, evaluated on the test split (accuracy only; CPU execution is
        not energy-comparable to the GPU measurements)."""
        model, tok = self.parent()
        model.eval()
        q = torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{self.a.dataset}_test.parquet"))
        df = df.iloc[:16] if self.a.smoke else df
        probs = []
        with torch.no_grad():
            for i in range(0, len(df), BATCH_SIZE):
                enc = tok(df["text"].iloc[i:i + BATCH_SIZE].tolist(), truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
                probs.append(q(**enc).logits.float().softmax(-1)[:, 1].numpy())
        p1 = np.concatenate(probs)
        y = df["label"].to_numpy()
        pred = (p1 >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        row = dict(model="quant_int8", parent="bert_base", split="test", precision_mode="cpu_int8", seed=self.a.seed, dataset=self.a.dataset, n=len(y),
                   accuracy=accuracy_score(y, pred), precision=pr, recall=rc, f1=f1, mcc=matthews_corrcoef(y, pred), tn=tn, fp=fp, fn=fn, tp=tp, n_params=0)
        new = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        pd.DataFrame({"row": np.arange(len(y)), "label": y, "pred": pred, "p_true": p1}).to_csv(os.path.join(self.out, "preds_quant_int8_test.csv"), index=False)
        print(f"[quant_int8] test: acc={row['accuracy']:.4f} mcc={row['mcc']:.4f}", flush=True)

    def job_bert64(self):
        """Length control: BERT-base fine-tuned and evaluated at 64 tokens (the sequence length SpikeBERT and the dense control use)."""
        global MAX_SEQ_LEN
        old = MAX_SEQ_LEN
        MAX_SEQ_LEN = 64
        try:
            set_seed(self.a.seed)
            tok = AutoTokenizer.from_pretrained(BERT)
            model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=2)
            model = self.train(model.to(DEV), tok, "bert_len64", 3)
            self.evaluate(model, tok, "bert_len64", "pretrained")
        finally:
            MAX_SEQ_LEN = old

    def job_grid(self):
        for r in (0.25, 0.5, 0.75):  # 0% undistilled is the parent itself (job_bert)
            model, tok = self.parent()
            for layer in model.bert.encoder.layer:
                prune_layer_heads(layer, heads_to_prune_by_magnitude(layer, r))
            model = self.train(model.to(DEV), tok, f"bert_prune{int(r * 100)}", 1)
            self.evaluate(model, tok, f"bert_prune{int(r * 100)}", "bert_base")
            del model
        teacher, _ = self.parent()
        teacher = teacher.to(DEV).eval()
        stok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
        set_seed(self.a.seed)
        student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL, num_labels=2).to(DEV)
        student = self.train(student, stok, "distil", 3, trainer_cls=DistillationTrainer, teacher_model=teacher)
        del teacher
        self.evaluate(student, stok, "distil_prune0", "bert_base")
        for r in (0.25, 0.5, 0.75):
            s = prune_distilbert_heads(copy.deepcopy(student), r)
            s = self.train(s, stok, f"distil_prune{int(r * 100)}", 1)
            self.evaluate(s, stok, f"distil_prune{int(r * 100)}", "distil_prune0")
            del s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["liar", "isot", "fakenewsnet"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--jobs", default="bert,linformer,performer,meanattn,grid")
    ap.add_argument("--out_dir", default=os.path.join(P3, "results_rev"))
    ap.add_argument("--work_dir", default="/kaggle/temp" if os.path.isdir("/kaggle") else os.path.join(P3, "_work_rev"))
    ap.add_argument("--smoke", action="store_true", help="16 examples, 1 epoch per stage")
    a = ap.parse_args()
    run = Run(a)
    for job in a.jobs.split(","):
        t0 = time.time()
        getattr(run, f"job_{job}")()
        print(f"=== job {job} done in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
