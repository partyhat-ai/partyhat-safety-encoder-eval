#!/usr/bin/env python3
"""metrics_core.py: small score-based metrics that v6_metrics.py builds on.

Pure numpy/sklearn; scores in, numbers out. No text ever.

Conventions: higher score = more unsafe; a decision is `score >= threshold`.
"""
import math

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12


def auprc(y, p):
    return float(average_precision_score(np.asarray(y, int), np.asarray(p, float)))


def threshold_lowest_at_frr(benign, frr=0.05):
    """The lowest threshold with FRR <= frr. k = floor(frr * n); t is the smallest
    float strictly above the (k+1)-th largest benign score."""
    b = np.sort(np.asarray(benign, float))[::-1]
    k = int(math.floor(frr * b.size + EPS))
    if k >= b.size:
        return float(-np.inf)
    return float(np.nextafter(b[k], np.inf))


def f1_at(y, p, t):
    """F1 on the unsafe class when flagging every score >= t."""
    y, d = np.asarray(y, int), (np.asarray(p, float) >= t).astype(int)
    tp = int(((d == 1) & (y == 1)).sum())
    fp = int(((d == 1) & (y == 0)).sum())
    fn = int(((d == 0) & (y == 1)).sum())
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
