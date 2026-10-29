"""Stage 2 / RO2 — knowledge distillation from the P1 baseline (teacher) to
distilbert-base-uncased (student). Loss = KL(student_soft, teacher_soft) + CE(student, labels).

Student choice: distilbert-base-uncased (pretrained, least code) rather than a
truncated warm-started BERT-base — per plan sign-off default.

Usage: python distill.py --dataset liar
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "p1_baseline"))
from config import BATCH_SIZE, MAX_SEQ_LEN, PRECISION, SEED  # noqa: E402
from profile_energy import run_profiled_inference  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
STUDENT_MODEL = "distilbert-base-uncased"
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5  # weight on distillation loss vs. hard-label CE


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        student_logits = outputs.logits

        with torch.no_grad():
            teacher_logits = self.teacher(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).logits

        ce_loss = F.cross_entropy(student_logits, labels)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / KD_TEMPERATURE, dim=-1),
            F.softmax(teacher_logits / KD_TEMPERATURE, dim=-1),
            reduction="batchmean",
        ) * (KD_TEMPERATURE ** 2)
        loss = KD_ALPHA * kd_loss + (1 - KD_ALPHA) * ce_loss
        return (loss, outputs) if return_outputs else loss


def load_split(dataset: str, split: str) -> Dataset:
    df = pd.read_parquet(os.path.join(P1_RESULTS_DIR, "data", f"{dataset}_{split}.parquet"))
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": matthews_corrcoef(labels, preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["liar", "isot", "fakenewsnet"])
    args = parser.parse_args()

    teacher_dir = os.path.join(P1_RESULTS_DIR, args.dataset, "best")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir)

    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL, num_labels=2)

    if torch.cuda.is_available():
        teacher = teacher.to("cuda")
        student = student.to("cuda")

    def tokenize(batch):
        return student_tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    train_ds = load_split(args.dataset, "train").map(tokenize, batched=True)
    val_ds = load_split(args.dataset, "val").map(tokenize, batched=True)

    out_dir = os.path.join(RESULTS_DIR, args.dataset, "distilled")
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=(PRECISION == "fp16"),
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = DistillationTrainer(
        teacher_model=teacher,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(f"{args.dataset} distilled(student={STUDENT_MODEL}) metrics: {metrics}")

    best_dir = os.path.join(out_dir, "best")
    trainer.save_model(best_dir)
    student_tokenizer.save_pretrained(best_dir)

    val_path = os.path.join(P1_RESULTS_DIR, "data", f"{args.dataset}_val.parquet")
    sample_texts = pd.read_parquet(val_path)["text"].tolist()[:BATCH_SIZE]
    if len(sample_texts) < BATCH_SIZE:
        sample_texts = (sample_texts * (BATCH_SIZE // len(sample_texts) + 1))[:BATCH_SIZE]
    encoded = student_tokenizer(sample_texts, truncation=True, max_length=MAX_SEQ_LEN,
                                 padding="max_length", return_tensors="pt")
    result = run_profiled_inference(student, encoded["input_ids"], encoded["attention_mask"])
    print(result)

    results_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "distill_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in ("eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_mcc"):
            writer.writerow([k, metrics.get(k)])
        writer.writerow(["energy_j_total", result["energy_j_total"]])
        writer.writerow(["energy_method", result["energy_method"]])
    print(f"Wrote {results_dir}/distill_metrics.csv")


if __name__ == "__main__":
    main()
