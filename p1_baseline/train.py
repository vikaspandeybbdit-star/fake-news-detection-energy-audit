"""Fine-tune bert-base-uncased for binary fake-news classification on one dataset.

Usage: python train.py --dataset liar
Checkpoints to results/<dataset>/checkpoint-*, which is what profile_energy.py loads.
"""

import argparse
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import BATCH_SIZE, MAX_SEQ_LEN, MODEL_NAME, PRECISION, SEED

DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_split(dataset: str, split: str) -> Dataset:
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{dataset}_{split}.parquet"))
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
    parser.add_argument("--teacher_checkpoint", default=None, help="Reserved for p3_compression/distill.py")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    train_ds = load_split(args.dataset, "train").map(tokenize, batched=True)
    val_ds = load_split(args.dataset, "val").map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    out_dir = os.path.join(RESULTS_DIR, args.dataset)
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(f"{args.dataset} val metrics: {metrics}")

    best_dir = os.path.join(out_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Saved best checkpoint to {best_dir}")


if __name__ == "__main__":
    main()
