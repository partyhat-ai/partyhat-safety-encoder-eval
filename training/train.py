#!/usr/bin/env python
"""
train.py: fine-tune a DeBERTa-v3 encoder as a 2-class prompt-safety classifier.

It takes the ToxicChat (toxicchat0124) training split from tc_common.py, optionally appends
extra JSONL training files, trains in FP32 with the Hugging Face Trainer, and writes the model,
the tokenizer and threshold.json.

  * FP32 only. DeBERTa-v3 silently degrades in fp16/bf16, so mixed precision is
    asserted OFF and the flags are never exposed.
  * Class imbalance is handled with a class-weighted loss ("balanced" weights by
    default, or frozen weights from --class_weights).
  * Soft targets (--soft_label_field), a fixed step budget (--max_steps) and the
    bag-of-words correction (--bow_logit_field) are optional.
  * A decision threshold is tuned on the held-out validation split and saved to
    threshold.json. No test set is touched here.

The loss is class-weighted cross-entropy over a 2-logit head (num_labels=2). On a
binary problem this equals a class-weighted BCE on the logit difference, and a
2-class head loads as a plain AutoModelForSequenceClassification with id2label
{0: non_toxic, 1: toxic}. `--loss focal` and `--loss plain` are also available.

Usage:
  python train.py --output_dir checkpoints/toxicchat-deberta-v3-large
  python train.py --model microsoft/deberta-v3-base   --loss focal   # fast fallback
  python train.py --max_samples 400 --epochs 1                       # smoke test
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from tc_common import (
    ID2LABEL,
    INPUT_FIELD,
    LABEL2ID,
    LABEL_FIELD,
    TARGET_F1,
    load_train_val,
    metrics_from_scores,
    softmax_np,
    tune_threshold,
)


class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted CE or focal loss, optional R-Drop."""

    def __init__(self, *args, class_weights=None, loss_type="weighted_ce",
                 focal_gamma=2.0, rdrop_alpha=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.rdrop_alpha = rdrop_alpha

    def _base_loss(self, logits, labels, weight):
        """Per-sample-reduced class-weighted CE or focal loss -> scalar."""
        if self.loss_type == "focal":
            ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
            pt = torch.exp(-ce)
            return ((1.0 - pt) ** self.focal_gamma * ce).mean()
        # weighted_ce (default) / plain (weight is None)
        return F.cross_entropy(logits, labels, weight=weight)

    def _soft_loss(self, logits, soft, weight):
        """Class-weighted CE against soft targets [1-y, y]:
            l_i = -(w1*y*log p1 + w0*(1-y)*log p0);  L = sum_i l_i / sum_i (w1*y + w0*(1-y)).
        The denominator is F.cross_entropy(weight=...)'s weighted-mean normalisation, so this
        equals weighted_ce exactly when every y is 0 or 1."""
        if self.loss_type != "weighted_ce":
            raise ValueError("soft targets are only defined for --loss weighted_ce")
        y = soft.to(device=logits.device, dtype=logits.dtype)
        logp = F.log_softmax(logits, dim=-1)
        w0, w1 = (weight[0], weight[1]) if weight is not None else (1.0, 1.0)
        num = -(w1 * y * logp[:, 1] + w0 * (1.0 - y) * logp[:, 0])
        den = w1 * y + w0 * (1.0 - y)
        return num.sum() / den.sum()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        soft = inputs.pop("soft_labels", None)
        bow = inputs.pop("bow_logit", None)
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(device=logits.device, dtype=logits.dtype)
            if self.class_weights is not None
            else None
        )
        # Bag-of-words correction (recipe variant R2). In TRAINING only, the loss sees
        # logits + [-b/2, +b/2], so the encoder gets gradient only for what the frozen
        # bag-of-words expert cannot explain. Evaluation and inference use the raw logits,
        # and `outputs` always carries the raw logits.
        shift = None
        if bow is not None and model.training:
            b = bow.to(device=logits.device, dtype=logits.dtype)
            shift = torch.stack([-0.5 * b, 0.5 * b], dim=1)
        adj = (lambda lg: lg + shift) if shift is not None else (lambda lg: lg)  # noqa: E731
        if soft is not None:
            base = lambda lg: self._soft_loss(adj(lg), soft, weight)  # noqa: E731
        else:
            base = lambda lg: self._base_loss(adj(lg), labels, weight)  # noqa: E731
        loss = base(logits)

        # R-Drop: a second stochastic-dropout forward + symmetric KL between the
        # two predictive distributions. Zero inference cost (regularizer only);
        # tightens the decision boundary on hard examples -> typically a small
        # recall/AUPRC lift. Only active in training mode (dropout on) and alpha>0.
        if self.rdrop_alpha > 0 and model.training:
            logits2 = model(**inputs).logits
            loss = 0.5 * (loss + base(logits2))
            lp, lq = F.log_softmax(logits, dim=-1), F.log_softmax(logits2, dim=-1)
            kl = 0.5 * (
                F.kl_div(lp, lq, reduction="batchmean", log_target=True)
                + F.kl_div(lq, lp, reduction="batchmean", log_target=True)
            )
            loss = loss + self.rdrop_alpha * kl
        return (loss, outputs) if return_outputs else loss


def split_hash(df):
    """sha256 over a data split's (text, label) rows in order: identical iff the split is."""
    import hashlib
    h = hashlib.sha256()
    for t, y in zip(df[INPUT_FIELD].astype(str), df[LABEL_FIELD].astype(int)):
        h.update(t.encode("utf-8") + b"\x00" + str(y).encode() + b"\n")
    return h.hexdigest()


def make_compute_metrics():
    """Eval-time metrics for checkpoint selection. Tunes the threshold on the
    current val scores and reports F1 at that threshold (model is selected on
    best achievable F1)."""

    def _fn(eval_pred):
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if logits is None:
            logits, labels = eval_pred[0], eval_pred[1]
        if isinstance(logits, tuple):
            logits = logits[0]
        scores = softmax_np(np.asarray(logits))[:, 1]
        thr, _ = tune_threshold(labels, scores)
        m = metrics_from_scores(labels, scores, thr)
        return {
            "f1": m["f1"],
            "auprc": m["auprc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "threshold": m["threshold"],
        }

    return _fn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="microsoft/deberta-v3-large",
                   help="encoder backbone (fallbacks: microsoft/deberta-v3-base, "
                        "answerdotai/ModernBERT-base)")
    p.add_argument("--output_dir", default="checkpoints/toxicchat-deberta-v3-large")
    p.add_argument("--lr", type=float, default=1.5e-5)      # sweep {1e-5,1.5e-5,2e-5}
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation steps. EFFECTIVE batch = batch_size*grad_accum. "
                        "R-Drop runs a 2nd forward pass (≈2x activation memory), so DeBERTa-v3-large "
                        "len512 OOMs an 80GB A100 at bs16; use --batch_size 8 --grad_accum 2 to keep "
                        "the effective batch at 16 with the same math at ~half the peak memory.")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)          # SEEDS = [13, 42, 99]
    p.add_argument("--split_seed", type=int, default=None,
                   help="seed for the stratified train/val split. Defaults to --seed. "
                        "Pin it to give several runs the same held-out validation split.")
    p.add_argument("--head_seed", type=int, default=None,
                   help="seed for the reinitialized 2-class head. Defaults to --seed. "
                        "Pin it to give several runs the same head initialisation.")
    p.add_argument("--loss", choices=["weighted_ce", "focal", "plain"],
                   default="weighted_ce")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--rdrop_alpha", type=float, default=0.0,
                   help="R-Drop weight on the symmetric-KL consistency term (0=off). "
                        "Two dropout forwards per step regularize toward a tighter boundary; "
                        "zero inference cost. ~0.5-1.0 is a reasonable range for a 2-class head.")
    p.add_argument("--extra_train_jsonl", default=None, nargs="+",
                   help="path to a JSONL of EXTRA decontaminated training rows, each "
                        f"{{'{INPUT_FIELD}': str, '{LABEL_FIELD}': 0|1}}. Appended to the ToxicChat "
                        "training split (never val/test). The file MUST be pre-decontaminated "
                        "against BOTH test splits (decontam.py) — this loader does not re-check.")
    p.add_argument("--soft_label_field", default=None,
                   help="optional float field in the --extra_train_jsonl rows holding a soft target "
                        "y in [0,1] (e.g. toxicity_soft). Rows without it use their hard label. "
                        "Enables the soft-target weighted CE.")
    p.add_argument("--bow_logit_field", default=None,
                   help="optional float field in the --extra_train_jsonl rows holding a bag-of-words "
                        "log-odds b (the bag-of-words correction of recipe variant R2). During TRAINING "
                        "the loss is computed "
                        "on logits + [-b/2, +b/2]; evaluation and inference use the raw logits. Rows "
                        "without the field, and the ToxicChat loader rows, use b = 0.")
    p.add_argument("--group_by_length", action="store_true",
                   help="batch rows of similar token length together. Speed only.")
    p.add_argument("--class_weights", default=None,
                   help="frozen class weights 'w0,w1'. Default: computed "
                        "'balanced' from this run's training labels.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="fixed optimizer-step budget (overrides --epochs). Requires --eval_steps.")
    p.add_argument("--eval_steps", type=int, default=None,
                   help="evaluate + save every N steps (used with --max_steps).")
    p.add_argument("--run_meta", default=None,
                   help="write a JSON of step counts, class weights, the chosen checkpoint step "
                        "and the loss log to this path.")
    p.add_argument("--warm_start_model", default=None,
                   help="init the ENCODER BODY from a toxicity checkpoint (a fresh 2-class head "
                        "is reinitialized), e.g. cooperleong00/deberta-v3-large_toxicity-scorer. "
                        "Release 2.0 does not use it.")
    p.add_argument("--metric_for_best", choices=["auprc", "f1"], default="auprc",
                   help="val metric for checkpoint selection. AUPRC is threshold-free and the more "
                        "reliable bar to clear; F1 depends on the operating point.")
    p.add_argument("--allow_tf32", action="store_true",
                   help="allow TF32 matmuls (Ampere+): ~3-5x faster, usually negligible quality loss. "
                        "This is NOT fp16/bf16 — the DeBERTa-v3 pitfall is reduced STORAGE precision, "
                        "not TF32 accumulation. Off by default to stay strictly IEEE-FP32.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="cap rows for a quick smoke test")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", default=None)
    p.add_argument("--private", action=argparse.BooleanOptionalAction, default=True,
                   help="with --push_to_hub, create the Hub repo as private (the default); "
                        "--no-private creates it public.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[precision] TF32 matmuls ENABLED (still no fp16/bf16)")
    init_from = args.warm_start_model or args.model
    # The validation split and the head initialisation have their own seeds, so
    # runs can share them while --seed still varies shuffling and dropout.
    split_seed = args.seed if args.split_seed is None else args.split_seed
    head_seed = args.seed if args.head_seed is None else args.head_seed

    # ---- data ------------------------------------------------------------- #
    train_df, val_df = load_train_val(
        val_fraction=args.val_fraction, seed=split_seed, max_samples=args.max_samples
    )
    val_split_sha256 = split_hash(val_df)
    print(f"[data] val split sha256={val_split_sha256} (split_seed={split_seed})")
    pos = int(train_df[LABEL_FIELD].sum())
    print(f"[data] train={len(train_df)} (pos={pos}, {pos/len(train_df):.2%})  "
          f"val={len(val_df)} (pos={int(val_df[LABEL_FIELD].sum())})")

    # ---- optional: append EXTRA pre-decontaminated training rows ----------- #
    # Extra training rows from other datasets. Appended to TRAIN only; the
    # validation split stays pure ToxicChat, so
    # the frozen threshold and val-AUPRC selection remain on in-distribution data.
    if args.extra_train_jsonl:
        import pandas as pd

        soft_f = args.soft_label_field
        bow_f = args.bow_logit_field
        if soft_f:
            train_df = train_df.copy()
            train_df["soft_labels"] = train_df[LABEL_FIELD].astype(float)
        if bow_f:
            train_df = train_df.copy()
            train_df["bow_logit"] = 0.0  # loader rows carry no bag-of-words score
        before, epos = len(train_df), int(train_df[LABEL_FIELD].sum())
        parts = [train_df[[INPUT_FIELD, LABEL_FIELD] + (["soft_labels"] if soft_f else [])
                          + (["bow_logit"] if bow_f else [])]]
        for path in args.extra_train_jsonl:
            extra = pd.read_json(path, lines=True)
            miss = {INPUT_FIELD, LABEL_FIELD} - set(extra.columns)
            if miss:
                raise SystemExit(f"--extra_train_jsonl {path} missing columns {miss}")
            extra = extra.dropna(subset=[INPUT_FIELD])
            extra[LABEL_FIELD] = extra[LABEL_FIELD].astype(int)
            cols = [INPUT_FIELD, LABEL_FIELD]
            if soft_f:
                if soft_f in extra.columns:
                    if extra[soft_f].isna().any():
                        raise SystemExit(f"--soft_label_field: {path} has rows with no {soft_f} "
                                         "(a missing soft target becomes a NaN loss)")
                    extra["soft_labels"] = extra[soft_f].astype(float)
                    n_soft = int(((extra["soft_labels"] > 0) & (extra["soft_labels"] < 1)).sum())
                else:
                    extra["soft_labels"] = extra[LABEL_FIELD].astype(float)
                    n_soft = 0
                cols.append("soft_labels")
                print(f"[extra] {path}: {n_soft} rows with a fractional soft target")
            if bow_f:
                if bow_f in extra.columns:
                    if extra[bow_f].isna().any():
                        raise SystemExit(f"--bow_logit_field: {path} has rows with no {bow_f}")
                    extra["bow_logit"] = extra[bow_f].astype(float)
                else:
                    extra["bow_logit"] = 0.0
                cols.append("bow_logit")
                print(f"[extra] {path}: bag-of-words log-odds on "
                      f"{int((extra['bow_logit'] != 0).sum())} rows")
            parts.append(extra[cols])
            print(f"[extra] +{len(extra)} rows (pos={int(extra[LABEL_FIELD].sum())}/{len(extra)}) "
                  f"from {path}")
        train_df = pd.concat(parts, ignore_index=True)
        print(f"[extra] train {before}->{len(train_df)} "
              f"(pos {epos}->{int(train_df[LABEL_FIELD].sum())})")
    elif args.soft_label_field or args.bow_logit_field:
        raise SystemExit("--soft_label_field / --bow_logit_field need --extra_train_jsonl")
    use_soft = bool(args.soft_label_field)
    use_bow = bool(args.bow_logit_field)

    tok = AutoTokenizer.from_pretrained(init_from)

    def to_hf(df, soft=False, bow=False):
        cols = ([INPUT_FIELD, LABEL_FIELD] + (["soft_labels"] if soft else [])
                + (["bow_logit"] if bow else []))
        d = df[cols].rename(columns={LABEL_FIELD: "labels"})
        return Dataset.from_pandas(d, preserve_index=False)

    def tokenize(batch):
        return tok(batch[INPUT_FIELD], truncation=True, max_length=args.max_len)

    train_ds = to_hf(train_df, soft=use_soft, bow=use_bow).map(
        tokenize, batched=True, remove_columns=[INPUT_FIELD])
    val_ds = to_hf(val_df).map(tokenize, batched=True, remove_columns=[INPUT_FIELD])

    # ---- model ------------------------------------------------------------ #
    model = AutoModelForSequenceClassification.from_pretrained(
        init_from, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=bool(args.warm_start_model),
        torch_dtype=torch.float32,  # force IEEE FP32 at LOAD. Newer transformers
        # defaults to the checkpoint's stored dtype (often fp16), the DeBERTa-v3
        # pitfall. fp16=False in TrainingArguments governs autocast, NOT load dtype.
    )
    if args.warm_start_model:
        # ignore_mismatched_sizes only re-inits the head on a SHAPE mismatch. A
        # toxicity checkpoint often already has a 2-class head, so it would
        # otherwise transfer verbatim and carry the checkpoint's (possibly
        # inverted) class polarity. Re-init the classifier deterministically so
        # warm-start means "encoder body only".
        # Only the classifier is reinitialised, under head_seed. The ContextPooler is
        # kept from the checkpoint: on a warm-start checkpoint that has a head
        # (e.g. cooperleong00) it loads verbatim, so it is the same for every seed.
        set_seed(head_seed)
        model.classifier.apply(model._init_weights)
        print(f"[warm-start] encoder body <- {args.warm_start_model} "
              f"(2-class head reinitialized @ head_seed={head_seed}; "
              f"encoder is the transferred signal)")

    # ---- class weights ---------------------------------------------------- #
    class_weights = None
    if args.class_weights:
        cw = np.array([float(x) for x in args.class_weights.split(",")], dtype=np.float64)
        if cw.shape != (2,):
            raise SystemExit("--class_weights must be 'w0,w1'")
        class_weights = torch.tensor(cw, dtype=torch.float)
        print(f"[loss] {args.loss}  class_weights={cw.tolist()} (FROZEN via --class_weights)")
    elif args.loss in ("weighted_ce", "focal"):
        cw = compute_class_weight(
            "balanced", classes=np.array([0, 1]), y=train_df[LABEL_FIELD].values
        )
        class_weights = torch.tensor(cw, dtype=torch.float)
        print(f"[loss] {args.loss}  class_weights={cw.round(3).tolist()}")

    # ---- training args (FP32 only, see the module docstring) -------------- #
    if (args.max_steps is None) != (args.eval_steps is None):
        raise SystemExit("--max_steps and --eval_steps go together")
    step_kw = (
        dict(max_steps=args.max_steps, eval_strategy="steps", eval_steps=args.eval_steps,
             save_strategy="steps", save_steps=args.eval_steps)
        if args.max_steps else dict(eval_strategy="epoch", save_strategy="epoch")
    )
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        **step_kw,
        remove_unused_columns=not (use_soft or use_bow),  # keep soft_labels / bow_logit
        # transformers 5.x: the old group_by_length=True is train_sampling_strategy="group_by_length"
        train_sampling_strategy="group_by_length" if args.group_by_length else "random",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best,
        greater_is_better=True,
        save_total_limit=2,  # keep best + last; guards load_best_model_at_end
        fp16=False,   # DO NOT enable. DeBERTa-v3 silently degrades in fp16/bf16.
        bf16=False,   # DO NOT enable.
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    assert not targs.fp16 and not targs.bf16, "DeBERTa-v3 must train in FP32"

    trainer = WeightedLossTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=make_compute_metrics(),
        class_weights=class_weights,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        rdrop_alpha=args.rdrop_alpha,
    )
    if args.rdrop_alpha > 0:
        print(f"[rdrop] symmetric-KL consistency ON (alpha={args.rdrop_alpha})")

    trainer.train()
    st = trainer.state
    best_step = None
    if st.best_model_checkpoint:
        best_step = int(str(st.best_model_checkpoint).rstrip("/").rsplit("-", 1)[-1])
    run_meta = {
        "global_step": int(st.global_step),
        "max_steps": int(st.max_steps),
        "num_train_epochs": float(st.num_train_epochs),
        "steps_per_epoch": (int(st.max_steps // st.num_train_epochs)
                            if not args.max_steps else None),
        "effective_epochs": float(st.epoch or 0.0),
        "train_rows": int(len(train_df)),
        "train_pos": int(train_df[LABEL_FIELD].sum()),
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "soft_targets": use_soft,
        "bow_residual": use_bow,
        "group_by_length": bool(args.group_by_length),
        "best_checkpoint_step": best_step,
        "val_carve_sha256": val_split_sha256,   # key name kept: stored run_meta files use it
        "split_seed": split_seed,
        "best_val_auprc": st.best_metric,
        "log_history": st.log_history,
    }
    print(f"[steps] global_step={run_meta['global_step']} max_steps={run_meta['max_steps']} "
          f"steps_per_epoch={run_meta['steps_per_epoch']} best_step={best_step}")
    if args.run_meta:
        with open(args.run_meta, "w") as f:
            json.dump(run_meta, f, indent=1)

    # ---- freeze the decision threshold on val (NEVER on test) ------------- #
    val_logits = trainer.predict(val_ds).predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    val_scores = softmax_np(np.asarray(val_logits))[:, 1]
    val_labels = val_df[LABEL_FIELD].values
    threshold, val_f1 = tune_threshold(val_labels, val_scores)
    val_metrics = metrics_from_scores(val_labels, val_scores, threshold)
    print(f"\n[val] frozen threshold={threshold:.4f}  "
          f"F1={val_metrics['f1']:.4f}  AUPRC={val_metrics['auprc']:.4f}  "
          f"P={val_metrics['precision']:.4f}  R={val_metrics['recall']:.4f}")

    # ---- save model + tokenizer + frozen threshold ------------------------ #
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    meta = {
        "threshold": threshold,
        "val_f1": val_f1,
        "val_metrics": val_metrics,
        "base_model": args.model,
        "warm_start_model": args.warm_start_model,
        "loss": args.loss,
        "rdrop_alpha": args.rdrop_alpha,
        "extra_train_jsonl": args.extra_train_jsonl,
        "soft_label_field": args.soft_label_field,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_seed": split_seed,
        "head_seed": head_seed,
        "max_len": args.max_len,
        "val_fraction": args.val_fraction,
        "target_f1": TARGET_F1,
    }
    with open(os.path.join(args.output_dir, "threshold.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] model + tokenizer + threshold.json -> {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise SystemExit("--push_to_hub requires --hub_model_id")
        from huggingface_hub import HfApi
        model.push_to_hub(args.hub_model_id, private=args.private)
        tok.push_to_hub(args.hub_model_id, private=args.private)
        HfApi().upload_file(
            path_or_fileobj=os.path.join(args.output_dir, "threshold.json"),
            path_in_repo="threshold.json",
            repo_id=args.hub_model_id,
            repo_type="model",
        )
        print(f"[hub] pushed -> {args.hub_model_id} (private={args.private})")

    print(f"\nFor reference: the published ToxicChat-T5-large test F1 is {TARGET_F1}")


if __name__ == "__main__":
    main()
