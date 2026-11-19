"""v6_metrics.py: dev-set metrics, the three frozen thresholds, and the two-model average.

Pure numpy/sklearn on per-item scores; no text. Higher score = more unsafe; a prompt is flagged
when score >= threshold. The FPR-budget threshold rule and AUPRC come from metrics_core.py.

The dictionary keys that dev_report() returns (t_gate, t_screen, t_bal, saturation and so on) are
stored in the run results, so they keep their original names; README.md lists what each means.
"""
import math

import numpy as np
from sklearn.metrics import roc_auc_score

import metrics_core as mc

EPS = 1e-12
PINNED_MARGIN = 9.0   # |logit margin| above this means P(unsafe) is within about 1e-4 of 0 or 1


def probs_from_margin(margin):
    """P(unsafe) = softmax([l0, l1])[1] = sigmoid(l1 - l0), in float64 (so it never rounds to 0 or 1)."""
    m = np.asarray(margin, dtype=np.float64)
    out = np.empty_like(m)
    pos = m >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-m[pos]))
    e = np.exp(m[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def two_model_average(margin_a, margin_b):
    """Average two models: the mean of their logit margins; P(unsafe) is its sigmoid."""
    m = (np.asarray(margin_a, np.float64) + np.asarray(margin_b, np.float64)) / 2.0
    return m, probs_from_margin(m)


def auroc(y, s):
    y = np.asarray(y, int)
    return float(roc_auc_score(y, np.asarray(s, float))) if 0 < y.sum() < y.size else None


# ------------------------------------------------------------------------------------------- #
# the three frozen thresholds (set on dev data only)                                          #
# ------------------------------------------------------------------------------------------- #
def strict_threshold(benign, fpr=0.01):
    """Strict threshold (stored as t_gate): the lowest threshold with FPR <= 1%. k = floor(0.01 n);
    the smallest float strictly above the (k+1)-th largest benign score."""
    return mc.threshold_lowest_at_frr(benign, fpr)


def prescreen_threshold(unsafe, sens=0.98):
    """Pre-screen threshold (stored as t_screen): the k-th largest unsafe score, k = ceil(0.98 n).
    Escalate when score >= this threshold, so realised recall >= 98% even with ties (ties at the
    threshold are all escalated)."""
    u = np.sort(np.asarray(unsafe, float))[::-1]
    k = min(max(math.ceil(sens * u.size - EPS), 1), u.size)
    return float(u[k - 1])


def _f1_curve(y, s, thresholds):
    """F1 on the unsafe class for every threshold (flag when s >= t), vectorised."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = pos.size - np.searchsorted(pos, thresholds, side="left")
    fp = neg.size - np.searchsorted(neg, thresholds, side="left")
    fn = pos.size - tp
    denom = 2 * tp + fp + fn
    return np.where(tp > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)


def balanced_threshold(sets):
    """Balanced threshold (stored as t_bal): the threshold maximising the mean F1 over the given
    (labels, scores) sets, searched over the distinct scores of their items; a tie goes to the
    lower threshold. Returns (t, mean_f1)."""
    cand = np.unique(np.concatenate([np.asarray(s, float) for _, s in sets]))
    mean = np.mean([_f1_curve(y, s, cand) for y, s in sets], axis=0)
    i = int(np.flatnonzero(mean == mean.max())[0])   # cand ascending -> first max = lowest t
    return float(cand[i]), float(mean[i])


# ------------------------------------------------------------------------------------------- #
# dev metrics                                                                                  #
# ------------------------------------------------------------------------------------------- #
def pair_accuracy(scores, labels, pair_ids):
    """Share of pairs whose unsafe side scores above its benign side; ties count half."""
    by = {}
    for s, y, p in zip(scores, labels, pair_ids):
        by.setdefault(p, {})[int(y)] = float(s)
    vals = []
    for d in by.values():
        if 0 in d and 1 in d:
            vals.append(1.0 if d[1] > d[0] else 0.5 if d[1] == d[0] else 0.0)
    return float(np.mean(vals)) if vals else None, len(vals)


def pinned_score_share(margins):
    """Share of scores pinned near 0 or 1 (stored as `saturation`): |margin| > PINNED_MARGIN."""
    m = np.abs(np.asarray(margins, float))
    return float((m > PINNED_MARGIN).mean())


def dev_report(sets):
    """sets: {name: {"score": [...], "margin": [...], "label": [...], "pair_id": [...]?}}.
    Needs twins_prompt, dev_pair2, tc_val, aegis_val; the held-back pool rows are used when present.
    `dev_intent` is the intent score: the mean AUROC on twins_prompt and dev_pair2."""
    g = lambda n, k: np.asarray(sets[n][k])  # noqa: E731
    out = {}
    tw = auroc(g("twins_prompt", "label"), g("twins_prompt", "score"))
    pair2 = auroc(g("dev_pair2", "label"), g("dev_pair2", "score"))
    out["twins_auroc"], out["dev_pair2_auroc"] = tw, pair2
    out["dev_intent"] = float((tw + pair2) / 2.0)
    out["twins_pair_acc"], out["twins_pairs"] = pair_accuracy(
        g("twins_prompt", "score"), g("twins_prompt", "label"), sets["twins_prompt"]["pair_id"])
    out["tc_val_auprc"] = mc.auprc(g("tc_val", "label"), g("tc_val", "score"))
    out["aegis_val_auroc"] = auroc(g("aegis_val", "label"), g("aegis_val", "score"))
    for h in ("dev_holdout_p3", "dev_holdout_p4"):
        if h in sets:
            out[f"{h}_auroc"] = auroc(g(h, "label"), g(h, "score"))
    margins = np.concatenate([g(n, "margin") for n in ("twins_prompt", "tc_val", "aegis_val")])
    out["saturation"] = pinned_score_share(margins)

    benign = np.concatenate([g(n, "score")[g(n, "label") == 0] for n in ("tc_val", "aegis_val")])
    unsafe = np.concatenate([g(n, "score")[g(n, "label") == 1] for n in ("tc_val", "aegis_val")])
    strict, prescreen = strict_threshold(benign), prescreen_threshold(unsafe)
    out["t_gate"], out["t_screen"] = strict, prescreen
    out["t_gate_benign_n"], out["t_gate_k"] = int(benign.size), int(math.floor(0.01 * benign.size + EPS))
    bal_sets = [(g(n, "label"), g(n, "score")) for n in ("tc_val", "aegis_val", "dev_holdout_p3")
                if n in sets]
    if len(bal_sets) == 3:
        out["t_bal"], out["t_bal_mean_f1"] = balanced_threshold(bal_sets)
    if "dev_holdout_p5" in sets:
        out["dev_holdout_p5_frr_at_gate"] = float((g("dev_holdout_p5", "score") >= strict).mean())
    return out
