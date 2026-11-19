"""
tc_common.py: the shared ToxicChat data and metric code.

Everything that could let a number drift lives here exactly once:
  * dataset id / config / field names      (never inline these elsewhere)
  * the train/val split (stratified, seeded)
  * the metric implementation              (ONE function computes P/R/F1/AUPRC)
  * threshold tuning (on val only)
  * the published ToxicChat reference numbers

train.py imports its data loading, metrics and threshold tuning from here, so every
model is scored by the same code.
"""

import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# `datasets` and sklearn's splitter are imported lazily inside the loaders below
# so the metric/threshold code path stays importable with only numpy + sklearn.

# --------------------------------------------------------------------------- #
# LOCKED dataset identity — do not drift.                                      #
# --------------------------------------------------------------------------- #
DATASET_ID = "lmsys/toxic-chat"
CONFIG_NAME = "toxicchat0124"     # canonical current revision
INPUT_FIELD = "user_input"        # classify the user prompt
LABEL_FIELD = "toxicity"          # binary target (0/1)
JAILBREAK_FIELD = "jailbreaking"  # for the jailbreak-recall sub-metric
OPENAI_FIELD = "openai_moderation"

ID2LABEL = {0: "non_toxic", 1: "toxic"}
LABEL2ID = {"non_toxic": 0, "toxic": 1}

# --------------------------------------------------------------------------- #
# The bar to beat (verbatim from the toxicchat0124 dataset card, June 2026).   #
# --------------------------------------------------------------------------- #
TARGET_F1 = 0.8221
TARGET_AUPRC = 0.8850

# Published reference rows from the ToxicChat dataset card.
PUBLISHED = {
    "ToxicChat-T5-large (published)": dict(
        precision=0.7983, recall=0.8475, f1=0.8221, auprc=0.8850
    ),
    "OpenAI Moderation (published, thr=0.02)": dict(
        precision=0.5476, recall=0.6989, f1=0.6141, auprc=0.6313
    ),
}


# --------------------------------------------------------------------------- #
# Metrics — ONE implementation, used for every model.                         #
# --------------------------------------------------------------------------- #
def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for a 2-D [N, C] logit array."""
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def metrics_from_preds(y_true, y_pred, scores=None) -> dict:
    """The single source of truth for P/R/F1/accuracy (+ AUPRC if scores given).

    P/R/F1 come from the hard predictions `y_pred`. AUPRC is threshold-free and
    is computed from the continuous `scores` when available. The T5 baseline
    supplies y_pred from its own greedy decode; our encoder supplies y_pred from
    a thresholded score. Both land here.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    out["auprc"] = (
        float(average_precision_score(y_true, np.asarray(scores, dtype=float)))
        if scores is not None
        else float("nan")
    )
    return out


def metrics_from_scores(y_true, scores, threshold) -> dict:
    """Threshold a continuous score into preds, then score. For our encoder."""
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    m = metrics_from_preds(y_true, y_pred, scores=scores)
    m["threshold"] = float(threshold)
    return m


def tune_threshold(y_true, scores):
    """Pick the F1-maximizing threshold. MUST be called on val/train only.

    Returns (best_threshold, best_f1). Uses the exact score values as candidate
    cut points via precision_recall_curve (no arbitrary grid).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0 or np.unique(scores).size == 1:
        return 0.5, 0.0
    precision, recall, thr = precision_recall_curve(y_true, scores)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    # precision/recall have one more entry than thr; the trailing point
    # (recall=0) has no threshold, so align by dropping it.
    f1 = f1[:-1]
    if f1.size == 0:
        return 0.5, 0.0
    best = int(np.argmax(f1))
    return float(thr[best]), float(f1[best])


def jailbreak_recall(jb_true, y_pred):
    """Of the jailbreak-flagged prompts, what fraction did we mark toxic?"""
    jb = np.asarray(jb_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    mask = jb == 1
    if mask.sum() == 0:
        return None
    return float((yp[mask] == 1).mean())


# --------------------------------------------------------------------------- #
# Data loading.                                                                #
#   train.py -> load_train_val   (touches train + a held-out val split)       #
#   evaluation -> load_test      (touches test ONLY; never before final eval) #
# --------------------------------------------------------------------------- #
def _load_split(split: str, max_samples=None):
    # If TOXICCHAT_DIR points at a local clone of lmsys/toxic-chat, load the CSVs
    # directly (no gated-HF dependency). The columns match the HF dataset exactly.
    # Otherwise fall back to the gated Hub download.
    local = os.environ.get("TOXICCHAT_DIR")
    if local:
        import pandas as pd

        date_code = CONFIG_NAME[-4:]  # "toxicchat0124" -> "0124"
        path = os.path.join(local, "data", date_code,
                            f"toxic-chat_annotation_{split}.csv")
        df = pd.read_csv(path)
    else:
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, CONFIG_NAME, split=split)
        df = ds.to_pandas()
    if max_samples:
        df = df.iloc[:max_samples].copy()
    return df.reset_index(drop=True)


def load_train_val(val_fraction=0.1, seed=42, max_samples=None):
    """Stratified train -> (train, val) split. The ONLY place val is created,
    so the threshold frozen in train.py is reproducible from the same seed."""
    from sklearn.model_selection import train_test_split

    df = _load_split("train", max_samples=max_samples)
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=df[LABEL_FIELD],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_test(max_samples=None):
    return _load_split("test", max_samples=max_samples)


# --------------------------------------------------------------------------- #
# Pretty results table.                                                        #
# --------------------------------------------------------------------------- #
def format_results_table(rows) -> str:
    """rows: list of dicts with keys name, precision, recall, f1, auprc,
    jb_recall (optional), note (optional)."""
    header = (
        "| Model | Precision | Recall | F1 | AUPRC | JB-recall | Note |\n"
        "|---|---|---|---|---|---|---|"
    )

    def fmt(x):
        if x is None:
            return "—"
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    lines = [header]
    for r in rows:
        lines.append(
            "| {name} | {p} | {r} | {f1} | {auprc} | {jb} | {note} |".format(
                name=r.get("name", ""),
                p=fmt(r.get("precision")),
                r=fmt(r.get("recall")),
                f1=fmt(r.get("f1")),
                auprc=fmt(r.get("auprc")),
                jb=fmt(r.get("jb_recall")),
                note=r.get("note", ""),
            )
        )
    return "\n".join(lines)
