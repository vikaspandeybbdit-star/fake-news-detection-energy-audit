"""Performer / FAVOR+ (Choromanski et al. 2020, "Rethinking Attention with Performers",
arXiv:2009.14794) adapted to text classification on LIAR.

M12 of the exploratory (beyond-P1-P4) scope. Real mechanism verified by reading
lucidrains/performer-pytorch's softmax_kernel/linear_attention/FastAttention.forward
directly (performer_pytorch.py, github.com/lucidrains/performer-pytorch -- itself a
transcription of google-research/performer's JAX fast_attention.py), not recalled from
the paper alone:

1. A fixed random projection matrix (gaussian_orthogonal_random_matrix: QR-decomposed
   orthogonal blocks, shape (nb_features, head_dim)) maps Q and K through a positive
   random feature map (softmax_kernel) that in expectation reproduces the softmax
   kernel exp(q.k/sqrt(d)) -- this is FAVOR+'s core trick, letting attention be computed
   without ever forming the (seq x seq) score matrix.
2. Non-causal linear_attention then exploits associativity: instead of
   softmax(QK^T)V (O(seq^2)), compute K'^T V and Q' (K'^T V) (O(seq)), where Q', K' are
   the random-feature-mapped tensors -- normalized by D_inv = 1/(Q' . sum(K')).

Real-source functions (softmax_kernel, orthogonal_matrix_chunk,
gaussian_orthogonal_random_matrix, linear_attention) are ported with the same math,
minus the reference's einops.repeat calls -- torch.einsum's implicit broadcasting
produces an identical result without materializing the projection matrix per batch
element, verified against the reference by having no free `b`/`h` subscript on the
projection matrix's einsum term. Only the non-causal path is ported: BERT's encoder is
bidirectional, so the causal_linear_attention path (which needs a custom CUDA kernel or
an explicit chunked fallback) is out of scope entirely, not merely omitted for time.

Deliberate scope decisions, stated explicitly:

- Retrofit, not from-scratch, same as Linformer (M11): starts from the P1 baseline's
  already-fine-tuned LIAR checkpoint, only `attention.self` is replaced per layer,
  query/key/value Linear weights are inherited unchanged, only the (non-trainable)
  random projection matrix is new.
- nb_features=64 (== head_dim), NOT the reference library's own default
  (int(head_dim * log(head_dim)) = 266 for head_dim=64). That default is tuned for
  regimes where seq_len >> nb_features so the O(seq * nb_features * d) linear path beats
  O(seq^2 * d) by a wide margin. This project's fixed seq_len=256 (p1_baseline/config.py)
  is short enough that nb_features=266 would make the linear path's own FLOPs
  (256*266*64 ~= 4.36M) roughly comparable to, not smaller than, full attention's
  (256*256*64 ~= 4.19M) -- the "efficient" mechanism would not actually be more
  efficient at this sequence length with the library default. nb_features=64 gives a
  real ~4x FLOPs reduction (256*64*64 ~= 1.05M) and puts this on comparable footing with
  Linformer's own proj_k=64 (also a 4x sequence-axis reduction) for a fair side-by-side
  read of the two techniques. This is a single fixed choice, not swept.
- No periodic projection-matrix redraw (the reference's ProjectionUpdater / paper's own
  ablations note redrawing improves stability over long training runs) -- the matrix is
  drawn once at construction and left fixed. Only the core mechanism is being tested
  here, not every stabilization refinement, matching the same scope discipline used for
  EBT/SNN/Linformer.
- Padding is masked on the KERNEL-MAPPED k (post softmax_kernel), not on raw K before
  the feature map: softmax_kernel(0) is NOT 0 (it's ratio*(exp(-max)+eps), strictly
  positive), so zeroing raw K before projecting would still leak a nonzero contribution
  from padded positions into k_cumsum/context. V is masked directly (linear, so pre- or
  post- is equivalent there). This is not mentioned in the reference repo (it doesn't
  handle padding at all) -- worked out from the math above, not copied.
- No fixed-protocol deviations otherwise: trains/evaluates in the same fp16/batch=16/
  seq=256 as the P1 baseline and other P3 techniques.

Time-boxed single go/no-go attempt, matching M9/M10/M11's discipline: if this doesn't
train to a usable, evaluable classifier within one Kaggle run, "evaluated, found
impractical to adapt within scope" is the citable finding, stated plainly in
results/liar/performer_metrics.csv, not forced.

Usage: python adapt_performer.py --dataset liar
"""

import argparse
import csv
import math
import os
import sys

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "p1_baseline"))
from config import BATCH_SIZE, MAX_SEQ_LEN, PRECISION, SEED  # noqa: E402
from profile_energy import run_profiled_inference  # noqa: E402
from train import load_split, compute_metrics  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "p3_compression"))
from quantize import evaluate_accuracy  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


class _PadMaskBox:
    def __init__(self):
        self.mask = None  # (B, S) bool, True = pad position


def orthogonal_matrix_chunk(cols: int, device=None) -> torch.Tensor:
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    return q.t().to(device)


def gaussian_orthogonal_random_matrix(nb_rows: int, nb_columns: int, scaling: int = 0,
                                       device=None) -> torch.Tensor:
    nb_full_blocks = nb_rows // nb_columns
    block_list = [orthogonal_matrix_chunk(nb_columns, device=device) for _ in range(nb_full_blocks)]
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        block_list.append(orthogonal_matrix_chunk(nb_columns, device=device)[:remaining_rows])
    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f"Invalid scaling {scaling}")
    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(data: torch.Tensor, projection_matrix: torch.Tensor, is_query: bool,
                    eps: float = 1e-4) -> torch.Tensor:
    """data: (B,H,S,d). projection_matrix: (m,d), no b/h subscript in the einsum below
    -- broadcasts across batch/heads identically to the reference's explicit repeat."""
    data_normalizer = data.shape[-1] ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5
    projection = projection_matrix.type_as(data)

    data_dash = torch.einsum("bhnd,md->bhnm", data_normalizer * data, projection)
    diag_data = (data ** 2).sum(dim=-1, keepdim=True) / 2.0 * (data_normalizer ** 2)  # (B,H,S,1)

    if is_query:
        stabilizer = data_dash.amax(dim=-1, keepdim=True).detach()
    else:
        stabilizer = data_dash.amax(dim=(-1, -2), keepdim=True).detach()
    return ratio * (torch.exp(data_dash - diag_data - stabilizer) + eps)


def linear_attention(q_prime: torch.Tensor, k_prime: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q_prime, k_prime: (B,H,S,m); v: (B,H,S,d). Associativity trick: sum over S first
    (O(S*m*d)) instead of forming the (S,S) score matrix (O(S^2*d))."""
    k_cumsum = k_prime.sum(dim=-2)  # (B,H,m)
    d_inv = 1.0 / torch.einsum("bhnm,bhm->bhn", q_prime, k_cumsum)  # (B,H,S)
    context = torch.einsum("bhnm,bhnd->bhmd", k_prime, v)  # (B,H,m,d)
    return torch.einsum("bhmd,bhnm,bhn->bhnd", context, q_prime, d_inv)  # (B,H,S,d)


class PerformerSelfAttention(nn.Module):
    """Drop-in replacement for BertSelfAttention. Reuses query/key/value verbatim
    (pretrained/fine-tuned weights preserved); the projection_matrix is the only new
    (non-trainable) tensor."""

    def __init__(self, orig_self_attn, nb_features: int, pad_mask_box: _PadMaskBox):
        super().__init__()
        self.num_attention_heads = orig_self_attn.num_attention_heads
        self.attention_head_size = orig_self_attn.attention_head_size
        self.all_head_size = orig_self_attn.all_head_size
        self.query = orig_self_attn.query
        self.key = orig_self_attn.key
        self.value = orig_self_attn.value
        self.pad_mask_box = pad_mask_box
        projection_matrix = gaussian_orthogonal_random_matrix(nb_features, self.attention_head_size)
        self.register_buffer("projection_matrix", projection_matrix)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.attention_head_size)

        q = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        # FAVOR+'s D_inv divides by a sum of the kernel feature map's outputs -- under
        # fp16/autocast this can underflow to exactly 0 and blow up to inf/NaN (confirmed
        # empirically: the full LIAR run produced eval_loss=nan from epoch 1 onward under
        # HF Trainer's fp16=True, at both nb_features=8 and nb_features=64 -- ruling out
        # "too few features" as the cause). Computed in fp32 regardless of the ambient
        # autocast dtype -- matches PyTorch autocast's own policy of keeping numerically
        # sensitive reductions (softmax, exp, norm) in fp32 even inside an otherwise-fp16
        # forward -- then cast back so downstream fp16 layers see a consistent dtype.
        orig_dtype = hidden_states.dtype
        with torch.autocast(device_type=q.device.type, enabled=False):
            q32, k32, v32 = q.float(), k.float(), v.float()
            proj32 = self.projection_matrix.float()
            q_prime = softmax_kernel(q32, proj32, is_query=True)
            k_prime = softmax_kernel(k32, proj32, is_query=False)

            pad_mask = self.pad_mask_box.mask
            if pad_mask is not None:
                m = pad_mask[:, None, :, None]
                k_prime = k_prime.masked_fill(m, 0.0)
                v32 = v32.masked_fill(m, 0.0)

            out = linear_attention(q_prime, k_prime, v32)  # (B,H,S,d), fp32
        out = out.to(orig_dtype)
        out = out.transpose(1, 2).reshape(*input_shape, self.all_head_size).contiguous()
        return out, None


def performerize(model, nb_features: int) -> _PadMaskBox:
    pad_mask_box = _PadMaskBox()

    def _capture_pad_mask(module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is None and len(args) > 1:
            am = args[1]
        if am is not None:
            pad_mask_box.mask = (am == 0)

    model.register_forward_pre_hook(_capture_pad_mask, with_kwargs=True)

    for layer in model.bert.encoder.layer:
        orig = layer.attention.self
        layer.attention.self = PerformerSelfAttention(orig, nb_features, pad_mask_box)

    return pad_mask_box


def recovery_finetune(model, tokenizer, dataset: str, out_dir: str, epochs: int, smoke_test: bool = False):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    train_ds = load_split(dataset, "train")
    val_ds = load_split(dataset, "val")
    if smoke_test:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        val_ds = val_ds.select(range(min(8, len(val_ds))))
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=(PRECISION == "fp16"),
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                       eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liar", choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--nb_features", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny run (few examples, 1 epoch) to check the pipeline doesn't crash.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    ckpt_dir = os.path.join(P1_RESULTS_DIR, args.dataset, "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    performerize(model, args.nb_features)
    print(f"Replaced attention.self in {len(model.bert.encoder.layer)} layers "
          f"with Performer/FAVOR+ (nb_features={args.nb_features})")
    # performerize() constructs the projection_matrix buffer on CPU (device=None default
    # in gaussian_orthogonal_random_matrix) -- must move the WHOLE model to device in one
    # shot after replacement, not before, or the new buffer stays stranded on CPU.
    if torch.cuda.is_available():
        model = model.to("cuda")

    epochs = 1 if args.smoke_test else args.epochs
    recovery_out = os.path.join(RESULTS_DIR, args.dataset, "performer_recovery_ckpt")
    model = recovery_finetune(model, tokenizer, args.dataset, recovery_out, epochs, smoke_test=args.smoke_test)

    if args.smoke_test:
        metrics = evaluate_accuracy(model, tokenizer, args.dataset)
        print("Smoke test finished without crashing:", metrics)
        return

    metrics = evaluate_accuracy(model, tokenizer, args.dataset)
    print(f"{args.dataset} Performer(nb_features={args.nb_features}) metrics: {metrics}")

    if PRECISION == "fp16" and torch.cuda.is_available():
        model.half()

    val_path = os.path.join(P1_RESULTS_DIR, "data", f"{args.dataset}_val.parquet")
    sample_texts = pd.read_parquet(val_path)["text"].tolist()[:BATCH_SIZE]
    if len(sample_texts) < BATCH_SIZE:
        sample_texts = (sample_texts * (BATCH_SIZE // len(sample_texts) + 1))[:BATCH_SIZE]
    encoded = tokenizer(sample_texts, truncation=True, max_length=MAX_SEQ_LEN,
                         padding="max_length", return_tensors="pt")
    result = run_profiled_inference(model, encoded["input_ids"], encoded["attention_mask"])
    print(result)

    out_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "performer_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    with open(os.path.join(out_dir, "performer_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        writer.writerow(["energy_j_total", result["energy_j_total"]])
        writer.writerow(["energy_method", result["energy_method"]])
        writer.writerow(["nb_features", args.nb_features])
        writer.writerow(["max_seq_len", MAX_SEQ_LEN])
        writer.writerow(["epochs", epochs])
        writer.writerow(["note",
                          "retrofit onto P1's already-fine-tuned checkpoint (Q/K/V inherited, "
                          "only the fixed random projection matrix is new, not trained); "
                          "nb_features=64 chosen (not the reference library's own log-scaled "
                          "default) specifically because seq_len=256 is short -- see module "
                          "docstring for the FLOPs argument"])
    print(f"Wrote {out_dir}/performer_metrics.csv")


if __name__ == "__main__":
    main()
