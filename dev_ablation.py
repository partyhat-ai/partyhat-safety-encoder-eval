#!/usr/bin/env python3
"""dev_ablation.py: recompute the "Where the gain came from" section of the partyhat-safety-encoder 2.0 card and
the full tables behind it (reports/dev_ablation.md), from the per-item development scorecards in this bundle.

This is a descriptive analysis made after the fact. It was not planned before the training runs, it uses no test
set, and it trains or scores nothing: it only re-reads the development scorecards produced when the release 2.0
recipe was chosen. Read each difference as "on these development sets, these training runs differed by this much",
not as a measured causal effect of one change.

Reads a LABELLED copy of the bundle (run attach_labels.py first):

    pip install numpy scikit-learn
    python3 dev_ablation.py <labelled bundle folder> [--quick] [--write-report]

For every figure in the card section it prints the card's wording, the card's value and the recomputed value; it
then regenerates reports/dev_ablation.md and compares it with the copy in the bundle, line by line. It exits
non-zero on any mismatch. --write-report rewrites reports/dev_ablation.md instead of comparing it. --quick uses
1,000 bootstrap draws (intervals will then not match). With the full draws it runs in about a minute.

Recipe variants (run folders scorecards/dev/<folder>_s<seed>/; the folder names are kept from the training code):
- base recipe (folder R1): ToxicChat train (three copies), Aegis 2.0, WildGuardMix, WildJailbreak (all four
  types) and FalseReject, from microsoft/deberta-v3-large with no warm start.
- base recipe without the alarming-but-benign data (R0): no FalseReject and no WildJailbreak benign rows.
- base recipe without FalseReject (R4).
- base recipe + bag-of-words correction (R2): the same rows; during training the logit of a bag-of-words model is
  added to the encoder's logit.
- base recipe + teacher labels (R3, the released recipe; R3_s13 is the published weights): labels that three open
  guard models (the teachers) confidently disputed are trained toward 0.5, and 100,000 extra prompts are added, most
  of them with teacher targets.
- Each of these has three training runs (seeds 13, 42, 99), all 9,396 optimizer steps at batch 32, so the number of
  passes over the data differs by variant.
- Two-model averages (no training): the mean of two runs' logit margins.
- Release 1.0 trial checkpoints (folders A0_s13, A0_s13_repeat): release 1.0's recipe retrained twice with the same
  seed (the runs differ only by GPU nondeterminism), warm-started from
  cooperleong00/deberta-v3-large_toxicity-scorer, scored on the same development sets.

Metrics (higher score = more unsafe; `score` is P(unsafe)):
- AUROC on each two-class development set; the intent score is the mean AUROC on the paired prompts and on the
  PHTest, HarmfulQA and CatQA set; AUPRC on the ToxicChat validation split (sklearn average_precision_score).
- The FalseReject held-back rows are all benign, so they have no AUROC; their figure is the share flagged at the
  strict threshold: with n benign rows of the ToxicChat validation split and the Aegis 2.0 validation set and
  k = floor(0.01 n), a prompt is flagged when its score is above the (k+1)-th largest of those benign scores. The
  threshold is recomputed inside every bootstrap draw.
- The trial checkpoints were scored on an earlier, 1,199-row Aegis 2.0 validation set; 40 rows that match test items
  were removed before the other runs were scored. They are evaluated here on the 1,159 shared rows (joined by `src`).

Intervals: 95% paired bootstrap, 10,000 draws, built as in reproduce_card.py. Benign and unsafe items are resampled
separately within each set, with the same item draws for every training run; each set has its own random stream,
numpy default_rng([13, zlib.crc32(set file name)]). Training runs are resampled with replacement inside each draw,
independently for each recipe variant (stream "training_runs:<variant key>"; reproduce_card.py resamples one recipe
and uses "training_runs"). A difference's interval is the 2.5th to 97.5th percentile of its 10,000 draws.
"""
import argparse
import json
import math
import pathlib
import sys
import zlib

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_SEED = 13
SEEDS = (13, 42, 99)
EPS = 1e-12

# variant key -> (run-folder prefix, plain name)
TRAINED = {
    "base": ("R1", "base recipe"),
    "no_alarming_benign": ("R0", "base recipe without the alarming-but-benign data"),
    "no_falsereject": ("R4", "base recipe without FalseReject"),
    "bag_of_words": ("R2", "base recipe + bag-of-words correction"),
    "teacher_labels": ("R3", "base recipe + teacher labels (released)"),
}
# two-model averages: key -> (plain name, [(variant, seed), (variant, seed)] per training run)
AVERAGES = {
    "avg_base_base": ("two-model average: base recipe, two seeds",
                      [(("base", 13), ("base", 42)), (("base", 42), ("base", 99)), (("base", 99), ("base", 13))]),
    "avg_base_bow": ("two-model average: base recipe and + bag-of-words correction",
                     [(("base", s), ("bag_of_words", s)) for s in SEEDS]),
    "avg_base_teacher": ("two-model average: base recipe and + teacher labels",
                         [(("base", s), ("teacher_labels", s)) for s in SEEDS]),
    "avg_bow_teacher": ("two-model average: + bag-of-words correction and + teacher labels",
                        [(("bag_of_words", s), ("teacher_labels", s)) for s in SEEDS]),
}
TRIAL_RUNS = ["A0_s13", "A0_s13_repeat"]
TRIAL = ("release_1_trial", "release 1.0 trial checkpoints")
DISPLAY_ORDER = [TRIAL[0], "base", "no_alarming_benign", "no_falsereject", "bag_of_words", "teacher_labels",
                 "avg_base_base", "avg_base_bow", "avg_base_teacher", "avg_bow_teacher"]

# set file name -> plain name
SETS = {
    "twins_prompt": "paired prompts (generated in-house)",
    "dev_pair2": "PHTest, HarmfulQA and CatQA set",
    "tc_val": "ToxicChat validation split",
    "aegis_val": "Aegis 2.0 validation set",
    "dev_holdout_p3": "WildGuardMix held-back rows",
    "dev_holdout_p4": "WildJailbreak held-back rows",
    "dev_holdout_p5": "FalseReject held-back rows",
}
TWO_CLASS_SETS = ["twins_prompt", "dev_pair2", "tc_val", "aegis_val", "dev_holdout_p3", "dev_holdout_p4"]
BENIGN_ONLY_SET = "dev_holdout_p5"
REFERENCE_RUN = "R1_s13"

# metric key, column label, kind ("auc": 3 decimals; "flagged": percent)
METRICS = [("intent_score", "intent score", "auc")]
METRICS += [(f"{s}.auroc", SETS[s], "auc") for s in TWO_CLASS_SETS]
METRICS += [(f"{BENIGN_ONLY_SET}.flagged", "FalseReject held-back rows flagged at the strict threshold", "flagged"),
            ("tc_val.auprc", "ToxicChat validation AUPRC", "auc")]

# contrast: (plain name, variant X, variant Y, what differs). The difference is X minus Y.
CONTRASTS = [
    ("Adding teacher labels (with 100,000 extra prompts)", "teacher_labels", "base",
     "two changes together. (1) 2,182 of the base recipe's WildGuardMix, WildJailbreak and FalseReject rows are "
     "trained toward 0.5 where the teachers confidently disputed the label. (2) 100,000 prompts are added: 30,000 more "
     "WildJailbreak vanilla rows (their own labels, unless disputed: 425 were, which with the 2,182 above makes the "
     "card's 2,607 disputed rows), 30,000 instructions from the HarmAug generated "
     "dataset and 40,000 WildChat-1M first turns (teacher targets only); 35,761 of the 100,000 are unsafe. "
     "250,316 rows seen 1.20 times against 150,316 rows seen 2.00 times"),
    ("Adding FalseReject", "base", "no_falsereject",
     "the FalseReject pool alone: 14,185 benign prompts. 150,316 rows seen 2.00 times against 136,131 seen 2.21 times"),
    ("Adding the alarming-but-benign data", "base", "no_alarming_benign",
     "a combined change: the FalseReject pool (14,185 rows) and WildJailbreak's benign rows (29,100: 14,550 "
     "vanilla_benign and 14,550 adversarial_benign) together. 150,316 rows seen 2.00 times against 107,031 seen 2.81 "
     "times"),
    ("Adding WildJailbreak's benign rows", "no_falsereject", "no_alarming_benign",
     "WildJailbreak's benign rows alone (29,100), FalseReject absent from both. 136,131 rows seen 2.21 times against "
     "107,031 seen 2.81 times. The variant without FalseReject was added after the first development results of the "
     "base recipe, the variant without the alarming-but-benign data and the bag-of-words variant had been seen"),
    ("Adding the bag-of-words correction", "bag_of_words", "base",
     "the bag-of-words correction alone: the same rows, seen the same number of times (2.00)"),
    ("Base recipe against the release 1.0 trial checkpoints", "base", TRIAL[0],
     "everything between release 1.0's recipe and the base recipe at once: the training data, the starting weights "
     "(a toxicity-scorer checkpoint against the plain base model) and the training settings"),
    ("Released recipe against the release 1.0 trial checkpoints", "teacher_labels", TRIAL[0],
     "the two previous steps together: base recipe against the trial checkpoints, plus adding teacher labels"),
]


# ----------------------------------------------------------------------------------------------- data
def probability_from_margin(margin):
    m = np.asarray(margin, dtype=np.float64)
    out = np.empty_like(m)
    pos = m >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-m[pos]))
    e = np.exp(m[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def run_folder(variant, seed):
    return f"{TRAINED[variant][0]}_s{seed}"


class DevScorecards:
    """Per-run score vectors on every development set, aligned to the reference run's item order."""

    def __init__(self, root):
        self.root = pathlib.Path(root) / "scorecards" / "dev"
        self.labels, self.row_ids, self.score, self.margin = {}, {}, {}, {}
        self.trial_full_aegis = {}
        all_sets = TWO_CLASS_SETS + [BENIGN_ONLY_SET]
        for s in all_sets:
            card = self._card(REFERENCE_RUN, s)
            self.labels[s] = np.array([int(r["label"]) for r in card["rows"]], int)
            self.row_ids[s] = [r["src"] for r in card["rows"]]
        folders = [run_folder(v, seed) for v in TRAINED for seed in SEEDS] + TRIAL_RUNS
        for run in folders:
            for s in all_sets:
                rows = self._card(run, s)["rows"]
                if run in TRIAL_RUNS and s == "aegis_val":
                    self.trial_full_aegis[run] = rows
                    by_id = {r["src"]: r for r in rows}
                    rows = [by_id[x] for x in self.row_ids[s]]
                assert [r["src"] for r in rows] == self.row_ids[s], (run, s, "item order differs")
                assert all(int(r["label"]) == y for r, y in zip(rows, self.labels[s])), (run, s, "labels differ")
                self.score[(run, s)] = np.array([r["score"] for r in rows], float)
                self.margin[(run, s)] = np.array([r["margin"] for r in rows], float)
        self.runs_of = {v: [run_folder(v, seed) for seed in SEEDS] for v in TRAINED}
        self.runs_of[TRIAL[0]] = list(TRIAL_RUNS)
        for key, (_, pairs) in AVERAGES.items():
            self.runs_of[key] = []
            for (va, sa), (vb, sb) in pairs:
                ra, rb = run_folder(va, sa), run_folder(vb, sb)
                name = f"{ra} + {rb}"
                for s in all_sets:
                    m = (self.margin[(ra, s)] + self.margin[(rb, s)]) / 2.0
                    self.margin[(name, s)], self.score[(name, s)] = m, probability_from_margin(m)
                self.runs_of[key].append(name)

    def _card(self, run, s):
        p = self.root / run / f"{s}.json"
        if not p.exists():
            raise SystemExit(f"missing scorecard {p}")
        card = json.loads(p.read_text())
        if "labels_removed" in card:
            raise SystemExit(f"{p} has no labels: run attach_labels.py and pass the labelled copy")
        return card

    def all_runs(self):
        return [r for v in self.runs_of for r in self.runs_of[v]]


def plain_name(variant):
    if variant == TRIAL[0]:
        return TRIAL[1]
    if variant in TRAINED:
        return TRAINED[variant][1]
    return AVERAGES[variant][0]


# ----------------------------------------------------------------------------------------------- point metrics
def strict_cut(benign_scores):
    """The (k+1)-th largest benign score, k = floor(0.01 n); a prompt is flagged when its score is above it."""
    b = np.sort(np.asarray(benign_scores, float))[::-1]
    return float(b[int(math.floor(0.01 * b.size + EPS))])


def point_metrics(dev, run):
    out = {}
    for s in TWO_CLASS_SETS:
        out[f"{s}.auroc"] = float(roc_auc_score(dev.labels[s], dev.score[(run, s)]))
    out["intent_score"] = (out["twins_prompt.auroc"] + out["dev_pair2.auroc"]) / 2.0
    out["tc_val.auprc"] = float(average_precision_score(dev.labels["tc_val"], dev.score[(run, "tc_val")]))
    benign = np.concatenate([dev.score[(run, s)][dev.labels[s] == 0] for s in ("tc_val", "aegis_val")])
    out[f"{BENIGN_ONLY_SET}.flagged"] = float((dev.score[(run, BENIGN_ONLY_SET)] > strict_cut(benign)).mean())
    return out


# ----------------------------------------------------------------------------------------------- bootstrap
def set_rng(name):
    return np.random.default_rng([BASE_SEED, zlib.crc32(name.encode("utf-8"))])


def counts(indices, n):
    """Per-draw multiplicities of each item (rows = draws)."""
    r = indices.shape[0]
    flat = (indices + (np.arange(r) * n)[:, None]).ravel()
    return np.bincount(flat, minlength=r * n).reshape(r, n)


def bootstrap_auroc(benign, unsafe, benign_counts, unsafe_counts):
    order = np.argsort(benign, kind="mergesort")
    sorted_benign = benign[order]
    left = np.searchsorted(sorted_benign, unsafe, "left")
    right = np.searchsorted(sorted_benign, unsafe, "right")
    cum = np.concatenate([np.zeros((benign_counts.shape[0], 1)), np.cumsum(benign_counts[:, order], 1)], 1)
    below, at_or_below = cum[:, left], cum[:, right]
    return (unsafe_counts * (below + 0.5 * (at_or_below - below))).sum(1) / (len(benign) * len(unsafe))


def bootstrap_auprc(labels, scores, weights):
    """Average precision with per-draw item multiplicities (rows = draws); row by row it equals sklearn's
    average_precision_score(labels, scores, sample_weight=row)."""
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, y_sorted = scores[order], labels[order].astype(float)
    group_ends = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]
    w = weights[:, order].astype(float)
    tp = np.cumsum(w * y_sorted, 1)[:, group_ends]
    fp = np.cumsum(w * (1 - y_sorted), 1)[:, group_ends]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / tp[:, -1:]
    return (np.diff(np.concatenate([np.zeros((w.shape[0], 1)), recall], 1), axis=1) * precision).sum(1)


def run_bootstrap(dev, draws, chunk=500):
    runs = dev.all_runs()
    per_run = {r: {} for r in runs}
    benign_counts_for_threshold = {}
    for s in TWO_CLASS_SETS:
        y = dev.labels[s]
        benign_idx, unsafe_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        nb, nu = len(benign_idx), len(unsafe_idx)
        rng = set_rng(s)
        benign_draws = np.vstack([np.arange(nb)[None], rng.integers(0, nb, (draws, nb))])
        unsafe_draws = np.vstack([np.arange(nu)[None], rng.integers(0, nu, (draws, nu))])
        auroc_parts = {r: [] for r in runs}
        auprc_parts = {r: [] for r in runs}
        kept_benign = []
        for c in range(0, draws + 1, chunk):
            cb, cu = counts(benign_draws[c:c + chunk], nb), counts(unsafe_draws[c:c + chunk], nu)
            if s in ("tc_val", "aegis_val"):
                kept_benign.append(cb.astype(np.int32))
            if s == "tc_val":
                weights = np.zeros((cb.shape[0], len(y)), np.int64)
                weights[:, benign_idx], weights[:, unsafe_idx] = cb, cu
            for r in runs:
                sc = dev.score[(r, s)]
                auroc_parts[r].append(bootstrap_auroc(sc[benign_idx], sc[unsafe_idx], cb, cu))
                if s == "tc_val":
                    auprc_parts[r].append(bootstrap_auprc(y, sc, weights))
        for r in runs:
            per_run[r][f"{s}.auroc"] = np.concatenate(auroc_parts[r])
            if s == "tc_val":
                per_run[r]["tc_val.auprc"] = np.concatenate(auprc_parts[r])
        if kept_benign:
            benign_counts_for_threshold[s] = np.concatenate(kept_benign)
        del benign_draws, unsafe_draws
    n5 = len(dev.labels[BENIGN_ONLY_SET])
    p5_counts = counts(np.vstack([np.arange(n5)[None], set_rng(BENIGN_ONLY_SET).integers(0, n5, (draws, n5))]), n5)
    pooled_counts = np.hstack([benign_counts_for_threshold["tc_val"], benign_counts_for_threshold["aegis_val"]])
    k = int(math.floor(0.01 * pooled_counts.shape[1] + EPS))
    for r in runs:
        benign = np.concatenate([dev.score[(r, s)][dev.labels[s] == 0] for s in ("tc_val", "aegis_val")])
        order = np.argsort(-benign, kind="mergesort")
        cum = np.cumsum(pooled_counts[:, order], 1)
        cut = benign[order][np.argmax(cum > k, axis=1)]
        p5 = dev.score[(r, BENIGN_ONLY_SET)]
        per_run[r][f"{BENIGN_ONLY_SET}.flagged"] = (p5_counts * (p5[None, :] > cut[:, None])).sum(1) / n5
        per_run[r]["intent_score"] = (per_run[r]["twins_prompt.auroc"] + per_run[r]["dev_pair2.auroc"]) / 2.0
    per_variant = {}
    cols = np.arange(draws + 1)
    for v, rs in dev.runs_of.items():
        n_runs = len(rs)
        picks = np.vstack([np.arange(n_runs)[None], set_rng(f"training_runs:{v}").integers(0, n_runs, (draws, n_runs))])
        per_variant[v] = {m: np.stack([per_run[r][m] for r in rs])[picks, cols[:, None]].mean(1) for m, _, _ in METRICS}
    return per_run, per_variant


# ----------------------------------------------------------------------------------------------- formatting
def signed(x, dp=3):
    s = f"{abs(x):.{dp}f}"
    if float(s) == 0:
        return f"+{s}" if x >= 0 else f"−{s}"
    return ("+" if x > 0 else "−") + s


def fmt_value(x, kind):
    return f"{100 * x:.1f}%" if kind == "flagged" else f"{x:.3f}"


def fmt_difference(d, kind):
    lo, hi = np.nanpercentile(d[1:], [2.5, 97.5])
    if kind == "flagged":
        return f"{signed(100 * d[0], 1)} [{signed(100 * lo, 1)}, {signed(100 * hi, 1)}]"
    return f"{signed(d[0])} [{signed(lo)}, {signed(hi)}]"


# ----------------------------------------------------------------------------------------------- report
def build_report(dev, points, per_variant):
    trial_aegis_full = ", ".join(
        fmt_value(float(roc_auc_score([int(x["label"]) for x in dev.trial_full_aegis[r]],
                                      [x["score"] for x in dev.trial_full_aegis[r]])), "auc") for r in TRIAL_RUNS)
    lines = [
        "# Where the gain came from: development-set analysis", "",
        "Generated by `dev_ablation.py` from the development scorecards in `scorecards/dev/` (labelled copy). Every "
        "number below is recomputed by that script, which exits non-zero if this file or the card section differs "
        "from what it computes.", "",
        "## Read this first", "",
        "- **Descriptive and after the fact.** This analysis was not planned before the training runs. It re-reads the "
        "development scorecards produced when the release 2.0 recipe was chosen; no model was trained or scored for "
        "it, and no test set is used. The differences describe these training runs on these sets; they are not "
        "measured causal effects.",
        "- **The recipe variants were not built to isolate one change each.** Only adding FalseReject and adding the "
        "bag-of-words correction change one thing. Adding teacher labels also adds 100,000 prompts; adding the "
        "alarming-but-benign data adds two pools at once; the base recipe against the release 1.0 trial checkpoints "
        "changes data, starting weights and training settings at once. No comparison separates the warm start from "
        "the data.",
        "- **Same number of steps, different passes over the data.** Every training run of the recipe variants took "
        "9,396 optimizer steps at batch 32, so a variant with fewer rows sees each row more often: without the "
        "alarming-but-benign data 2.81 times, without FalseReject 2.21, the base recipe and the bag-of-words "
        "variant 2.00, with teacher labels 1.20. This is part of every comparison except the bag-of-words one. The "
        "release 1.0 trial checkpoints trained for up to 5 epochs on 13,132 rows at batch 16 and each kept the "
        "checkpoint with the best ToxicChat validation AUPRC, which favours them on that split.",
        "- **The held-back rows are in-domain for the variants that train on their source.** The WildGuardMix, "
        "WildJailbreak and FalseReject held-back rows were held back from the same sources as the training pools. So "
        "adding FalseReject, and adding the alarming-but-benign data, compare on the FalseReject held-back rows a "
        "variant that trained on FalseReject with one that did not; adding the alarming-but-benign data likewise on "
        "WildJailbreak's benign held-back rows; and every variant against the release 1.0 trial checkpoints on all "
        "three (the trial checkpoints trained on none of these sources).",
        "- **The paired prompts were generated in-house** for this project: 602 pairs of a harmful and a benign "
        "prompt on the same topic. No training run saw them, but they are not an independent public benchmark.",
        "- **The teachers were calibrated on two development sets.** Each teacher's score was Platt-scaled (two "
        "parameters) on the labels of the ToxicChat validation split and the Aegis 2.0 validation set before it set "
        "the teacher-label variant's targets, so that variant's figures on those two sets are not fully independent "
        "of its training targets.",
        "- **Small sets and few runs.** The ToxicChat validation split has 38 unsafe prompts, so its AUPRC is noisy. "
        "The release 1.0 trial checkpoints are two runs with the same seed (they differ only by GPU "
        "nondeterminism), not three seeds, so their run-to-run spread is understated.",
        "- **The trial checkpoints' Aegis 2.0 validation set.** They were scored on an earlier 1,199-row version; 40 "
        "rows that match test items were removed before the other runs were scored. They are evaluated here on the "
        "1,159 shared rows.",
        "",
        "## The recipe variants", "",
        "| Recipe variant | Run folders | Training data | Rows | Times each row is seen |",
        "|---|---|---|---|---|",
        "| release 1.0 trial checkpoints | A0_s13, A0_s13_repeat | release 1.0's recipe: ToxicChat train plus "
        "relabelled response rows, warm-started from `cooperleong00/deberta-v3-large_toxicity-scorer` | 13,132 | up "
        "to 5 (best ToxicChat validation checkpoint kept) |",
        "| base recipe | R1_s13, R1_s42, R1_s99 | ToxicChat train (three copies), Aegis 2.0, WildGuardMix, "
        "WildJailbreak (all four types), FalseReject; from the base model, no warm start | 150,316 | 2.00 |",
        "| base recipe without the alarming-but-benign data | R0_s13, R0_s42, R0_s99 | the base recipe without "
        "FalseReject and without WildJailbreak's benign rows | 107,031 | 2.81 |",
        "| base recipe without FalseReject | R4_s13, R4_s42, R4_s99 | the base recipe without FalseReject | 136,131 | "
        "2.21 |",
        "| base recipe + bag-of-words correction | R2_s13, R2_s42, R2_s99 | the base recipe's rows; during training a "
        "bag-of-words model's logit is added to the encoder's | 150,316 | 2.00 |",
        "| base recipe + teacher labels (released; R3_s13 is the published weights) | R3_s13, R3_s42, R3_s99 | the "
        "base recipe's rows with 2,182 teacher-disputed labels trained toward 0.5, plus 100,000 extra prompts "
        "(30,000 WildJailbreak vanilla, 425 of them disputed; 30,000 HarmAug generated instructions, 40,000 WildChat-1M "
        "first turns; teacher targets for the 70,000 without labels) | 250,316 | 1.20 |",
        "| two-model averages | built from the runs above | no training: the mean logit margin of two training runs "
        "(base + base with different seeds; base + bag-of-words; base + teacher labels; bag-of-words + teacher "
        "labels) | n/a | n/a |",
        "",
        "The variant without the alarming-but-benign data trains on a WildJailbreak file (`p4_r0` in the training "
        "code) that keeps only the `vanilla_harmful` and `adversarial_harmful` rows (29,100) and drops the 29,100 "
        "`vanilla_benign` and `adversarial_benign` rows; that variant also has no FalseReject pool. The three teachers "
        "are Qwen3Guard-Gen-8B, granite-guardian-4.1-8b and Shieldstral-1.0-3B.",
        "",
        "## Results by recipe variant (mean over training runs [lowest, highest])", "",
        "AUROC unless marked. The FalseReject held-back rows are all benign, so they have no AUROC: their column is "
        "the share flagged at the strict threshold (1% false positives on the benign rows of the ToxicChat "
        "validation split and the Aegis 2.0 validation set; lower is better). The intent score is the mean AUROC on "
        "the paired prompts and on the PHTest, HarmfulQA and CatQA set.", ""]
    lines += ["| Recipe variant | Training runs | " + " | ".join(lab for _, lab, _ in METRICS) + " |",
              "|" + "---|" * (len(METRICS) + 2)]
    for v in DISPLAY_ORDER:
        rs = dev.runs_of[v]
        cells = []
        for m, _, kind in METRICS:
            vals = [points[r][m] for r in rs]
            cells.append(f"{fmt_value(np.mean(vals), kind)} [{fmt_value(min(vals), kind)}, {fmt_value(max(vals), kind)}]")
        lines.append(f"| {plain_name(v)} | {len(rs)} | " + " | ".join(cells) + " |")
    lines += ["", "Every training run (two-model averages: the two run folders averaged):", "",
              "| Recipe variant | Run folder(s) | " + " | ".join(lab for _, lab, _ in METRICS) + " |",
              "|" + "---|" * (len(METRICS) + 2)]
    for v in DISPLAY_ORDER:
        for r in dev.runs_of[v]:
            lines.append(f"| {plain_name(v)} | {r} | " +
                         " | ".join(fmt_value(points[r][m], kind) for m, _, kind in METRICS) + " |")
    lines += ["", "Missing training runs: none. Every recipe variant has all three training runs (seeds 13, 42 and "
              "99) scored on all seven development sets; the release 1.0 trial checkpoints have both their runs "
              "scored on all seven (on the Aegis 2.0 validation set, the 1,159 shared rows). Their Aegis 2.0 "
              f"validation AUROC as first recorded, on the 1,199-row version: {trial_aegis_full}. Their intent score, "
              "as first recorded, is the same as here.", ""]
    lines += ["## Comparisons (difference of variant means [95% paired-bootstrap interval])", "",
              "AUROC and AUPRC differences are in AUROC units; the FalseReject held-back rows in percentage points of "
              "rows flagged (negative = fewer false flags). An interval that includes zero is a tie at this sample "
              "size.", ""]
    lines += ["| Comparison | " + " | ".join(lab for _, lab, _ in METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    for name, x, y, _ in CONTRASTS:
        lines.append(f"| {name} | " + " | ".join(
            fmt_difference(per_variant[x][m] - per_variant[y][m], kind) for m, _, kind in METRICS) + " |")
    lines += ["", "What each comparison changes (first variant minus second):", ""]
    for name, x, y, what in CONTRASTS:
        lines.append(f"- **{name}** ({plain_name(x)} minus {plain_name(y)}): {what}.")
    lines += ["", "## Method", "",
              "- Point estimates: AUROC (sklearn `roc_auc_score`) and AUPRC (`average_precision_score`) on "
              "P(unsafe); a variant's value is the mean over its training runs. A two-model average's score is the "
              "sigmoid of the two runs' mean logit margin.",
              "- Intervals: 10,000 paired bootstrap draws. Benign and unsafe items are resampled separately within "
              "each set, with the same item draws for every training run; each set has its own stream, "
              "`numpy.random.default_rng([13, zlib.crc32(set file name)])`. Inside each draw a variant's training "
              "runs are resampled with replacement (stream `training_runs:<variant key>`). For the FalseReject "
              "held-back rows the strict threshold is recomputed in every draw from that draw's benign rows of the "
              "ToxicChat validation split and the Aegis 2.0 validation set.",
              "- An interval is the 2.5th to 97.5th percentile of the 10,000 draws of the difference.", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- card section
def card_checks(per_run, per_variant, runs_of):
    """(card wording, value printed on the card, recomputed value). Edit together with the card."""
    def diff(x, y, m, kind="auc"):
        return fmt_difference(per_variant[x][m] - per_variant[y][m], kind)

    def value(v, m):
        return f"{per_variant[v][m][0]:.3f}"

    return [
        ("Intent score of the release 1.0 trial checkpoints (mean of their two runs)", "0.715",
         value(TRIAL[0], "intent_score")),
        ("Intent score of release 2.0's recipe (mean of three training runs)", "0.884",
         value("teacher_labels", "intent_score")),
        ("Release 1.0's recipe to the base recipe (base recipe minus trial checkpoints) [95% CI]",
         "+0.128 [+0.088, +0.170]", diff("base", TRIAL[0], "intent_score")),
        ("Teacher labels, added with 100,000 extra prompts: intent score [95% CI]", "+0.041 [+0.029, +0.051]",
         diff("teacher_labels", "base", "intent_score")),
        ("Same step on ToxicChat validation AUPRC: a tie [95% CI]", "+0.032 [−0.009, +0.080]",
         diff("teacher_labels", "base", "tc_val.auprc")),
        ("A bag-of-words correction lowered the intent score [95% CI]", "−0.053 [−0.072, −0.034]",
         diff("bag_of_words", "base", "intent_score")),
        ("The alarming-but-benign data lowered it too [95% CI]", "−0.054 [−0.067, −0.041]",
         diff("base", "no_alarming_benign", "intent_score")),
        ("Same step on ToxicChat validation AUPRC, with the data minus without [95% CI]", "+0.072 [−0.001, +0.150]",
         diff("base", "no_alarming_benign", "tc_val.auprc")),
        ("Intent score of the variant without the alarming-but-benign data (mean of three runs)", "0.897",
         value("no_alarming_benign", "intent_score")),
        ("That variant trails the base recipe's ToxicChat validation AUPRC by (the selection rule's check)", "0.072",
         f"{per_variant['base']['tc_val.auprc'][0] - per_variant['no_alarming_benign']['tc_val.auprc'][0]:.3f}"),
        ("Base recipe's run-to-run range in ToxicChat validation AUPRC", "0.055",
         f"{np.ptp([per_run[r]['tc_val.auprc'][0] for r in runs_of['base']]):.3f}"),
    ]


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()
    draws = 1000 if args.quick else 10000
    dev = DevScorecards(args.bundle)
    points = {r: point_metrics(dev, r) for r in dev.all_runs()}
    per_run, per_variant = run_bootstrap(dev, draws)
    mismatches = 0
    for r in dev.all_runs():   # draw 0 is the original sample: it must equal the direct computation
        for m, _, _ in METRICS:
            if abs(per_run[r][m][0] - points[r][m]) > 1e-9:
                print(f"INTERNAL MISMATCH {r} {m}: bootstrap {per_run[r][m][0]!r} != direct {points[r][m]!r}")
                mismatches += 1
    print("## Card section: Where the gain came from (development sets)")
    checks = card_checks(per_run, per_variant, dev.runs_of)
    for wording, card, got in checks:
        ok = card == got
        mismatches += not ok
        print(f"  [{'ok ' if ok else 'BAD'}] {wording}: card {card} | recomputed {got}")
    report = build_report(dev, points, per_variant) + "\n"
    path = pathlib.Path(args.bundle) / "reports" / "dev_ablation.md"
    if args.write_report:
        path.write_text(report)
        print(f"wrote {path}")
    else:
        have = path.read_text().splitlines() if path.exists() else []
        want = report.splitlines()
        differing = [(i + 1, h, w) for i, (h, w) in enumerate(zip(have, want)) if h != w]
        if len(have) != len(want):
            differing.append((min(len(have), len(want)) + 1, f"{len(have)} lines", f"{len(want)} lines"))
        for line_no, h, w in differing[:10]:
            print(f"  [BAD] reports/dev_ablation.md line {line_no}:\n        bundle:     {h}\n        recomputed: {w}")
        print(f"## reports/dev_ablation.md: {len(want)} lines recomputed, {len(differing)} differing")
        mismatches += len(differing)
    print(f"\n{len(checks)} card figures and the full report checked; {mismatches} mismatches.")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
