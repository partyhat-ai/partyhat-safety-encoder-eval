#!/usr/bin/env python3
"""v6_bow_crossfit.py: cross-fitted bag-of-words log-odds for the training files.

One computation serves two purposes:
  * a shortcut audit: how far vocabulary alone predicts the label in the training mix
    (AUROC overall and per pool file);
  * recipe variant R2 (base recipe + bag-of-words correction): the per-row `bow_logit` that
    train.py --bow_logit_field adds to the logits during training.

For every row of the given JSONL files, a unigram TF-IDF + logistic regression is fitted on the
other folds and its log-odds for the row is written as `bow_logit`, clipped to [-clip, +clip].
Rows with the same normalised text always share a fold, so a duplicate is never scored by a model
that saw its label. CPU only; prints counts and AUROCs, never text.

    python v6_bow_crossfit.py --inputs p1_copy.jsonl p2.jsonl p3.jsonl p4.jsonl p5.jsonl \
        --out_dir r2_pools/ [--report bow_crossfit_report.json]

Each input is rewritten under --out_dir with the same name, the same rows in the same order, and
one extra field. A path given twice is processed once.
"""
import argparse
import hashlib
import json
import os

import numpy as np


def _norm(t):
    return " ".join(str(t).lower().split())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--text_field", default="user_input")
    p.add_argument("--label_field", default="toxicity")
    p.add_argument("--out_field", default="bow_logit")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--clip", type=float, default=6.0)
    p.add_argument("--C", type=float, default=4.0)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--report", default=None, help="optional JSON path for the counts and AUROCs")
    return p.parse_args(argv)


def crossfit(texts, labels, folds=5, seed=13, C=4.0, min_df=2):
    """Out-of-fold log-odds of a unigram TF-IDF + logistic regression. Duplicate texts share a fold."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(labels, dtype=int)
    codes, groups = {}, np.empty(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        h = hashlib.sha1(_norm(t).encode("utf-8")).hexdigest()
        groups[i] = codes.setdefault(h, len(codes))
    arr = np.asarray(texts, dtype=object)
    oof = np.full(len(texts), np.nan)
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(arr, y, groups):
        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=min_df, sublinear_tf=True, lowercase=True)
        clf = LogisticRegression(C=C, max_iter=2000)  # no class weighting: we want calibrated log-odds
        clf.fit(vec.fit_transform(list(arr[tr])), y[tr])
        oof[te] = clf.decision_function(vec.transform(list(arr[te])))
    assert not np.isnan(oof).any(), "a row was left without an out-of-fold score"
    return oof, len(codes)


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    paths = list(dict.fromkeys(a.inputs))  # unique, order kept
    names = [os.path.basename(p) for p in paths]
    if len(set(names)) != len(names):
        raise SystemExit("two inputs share a file name; the outputs would collide")
    rows, owner = [], []
    for k, path in enumerate(paths):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get(a.text_field) is None or r.get(a.label_field) is None:
                        raise SystemExit(f"{path}: a row lacks {a.text_field} or {a.label_field}")
                    rows.append(r)
                    owner.append(k)
    owner = np.asarray(owner)
    texts = [str(r[a.text_field]) for r in rows]
    y = np.asarray([int(r[a.label_field]) for r in rows])
    raw, n_groups = crossfit(texts, y, a.folds, a.seed, a.C, a.min_df)
    b = np.clip(raw, -a.clip, a.clip)

    report = {"rows": int(len(rows)), "unique_texts": int(n_groups), "unsafe": int(y.sum()),
              "folds": a.folds, "seed": a.seed, "clip": a.clip, "C": a.C, "min_df": a.min_df,
              "auroc_overall": _auroc(y, raw), "share_at_clip": float((np.abs(raw) >= a.clip).mean()),
              "per_file": {}}
    for k, (path, name) in enumerate(zip(paths, names)):
        m = owner == k
        report["per_file"][name] = {"rows": int(m.sum()), "unsafe": int(y[m].sum()),
                                    "auroc": _auroc(y[m], raw[m]),
                                    "mean_abs_bow_logit": float(np.abs(b[m]).mean())}
        with open(os.path.join(a.out_dir, name), "w") as f:
            for r, v in zip([rows[i] for i in np.flatnonzero(m)], b[m]):
                r = dict(r)
                r[a.out_field] = round(float(v), 6)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    main()
