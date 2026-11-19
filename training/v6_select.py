#!/usr/bin/env python3
"""v6_select.py: choose the release candidate from per-item dev scorecards (no text).

Reads v6_results/runs/<run>/scorecards/<set>.json (fetched with `v6_modal.py --action fetch`) and
builds every eligible candidate: each recipe variant on its own (R1, R0, R2, R3, R4) and the
two-model averages E11, E12, E13, E23, made from the single models' margins. It then applies the
selection rule and the release bars, computes the three frozen thresholds for each training run
of the selected candidate, and writes V6_SELECTION.md and v6_results/selection.json.

    python3 v6_select.py [--results v6_results] [--out V6_SELECTION.md]

To re-run the selection from the evidence bundle's development scorecards (labelled copy, made by
attach_labels.py), point --dev-scorecards at its scorecards/dev folder:

    python3 v6_select.py --dev-scorecards ../evidence_with_labels/scorecards/dev --results rerun --out rerun/V6_SELECTION.md

The keys written to selection.json (B, G1-G3, replicates, t_gate and so on) keep their original
names, because v6_modal.py and the stored results read them; README.md lists what each means.
"""
import argparse
import json
import pathlib

import numpy as np

import v6_metrics as vm
from v6_score import DEV_SETS, RECIPE_NAMES, dev_sets_for_report

SEEDS = (13, 42, 99)
SINGLE_MODELS = ("R1", "R0", "R2", "R3", "R4")   # R4 was added later and is only used on its own
# Two-model averages: each candidate pairs two runs, (variant, seed, variant, seed), per seed.
TWO_MODEL_AVERAGES = {
    "E11": [("R1", 13, "R1", 42), ("R1", 42, "R1", 99), ("R1", 99, "R1", 13)],
    "E12": [("R1", s, "R2", s) for s in SEEDS],
    "E13": [("R1", s, "R3", s) for s in SEEDS],
    "E23": [("R2", s, "R3", s) for s in SEEDS],
}
CANDIDATE_NAMES = RECIPE_NAMES | {
    "E11": "two-model average: base recipe + base recipe (different seeds)",
    "E12": "two-model average: base recipe + bag-of-words correction",
    "E13": "two-model average: base recipe + teacher labels",
    "E23": "two-model average: bag-of-words correction + teacher labels",
}

# The intent score of the release 1.0 trial checkpoints scored on the v6 dev sets
# (v6_modal.py --action baseline, saved as v6_results/baseline_B.json).
RELEASE_1_TRIAL_INTENT_SCORE = 0.7147105385513868
# The three release bars, checked on the selected candidate's mean over its training runs.
INTENT_SCORE_BAR = RELEASE_1_TRIAL_INTENT_SCORE + 0.05    # intent score must reach this
TOXICCHAT_AUPRC_BAR = 0.80                               # tc_val AUPRC must reach this
PINNED_SCORE_LIMIT = 0.50                                # share of pinned scores must stay below
# A candidate may trail the base recipe's tc_val AUPRC by at most this much.
TOXICCHAT_AUPRC_MARGIN = 0.02
GATE_NAMES = {"G1": "intent score bar", "G2": "ToxicChat AUPRC bar",
              "G3": "share of scores pinned near 0 or 1"}


def name_of(candidate):
    return f"{candidate} ({CANDIDATE_NAMES[candidate]})" if candidate in CANDIDATE_NAMES else candidate


def load_run(results, run, dev_scorecards=None):
    """One run's dev scorecards, from v6_results/runs/<run>/scorecards/ or, if dev_scorecards is
    given, from <dev_scorecards>/<run>/ (the evidence bundle's layout)."""
    d = (pathlib.Path(dev_scorecards) / run if dev_scorecards
         else pathlib.Path(results) / "runs" / run / "scorecards")
    if not all((d / f"{n}.json").exists() for n in DEV_SETS):
        return None
    return {n: json.load(open(d / f"{n}.json")) for n in DEV_SETS}


def combine(ca, cb):
    """Two-model average scorecards: mean margin, sigmoid score; items must align by src."""
    out = {}
    for n in DEV_SETS:
        ra, rb = ca[n]["rows"], cb[n]["rows"]
        assert [r["src"] for r in ra] == [r["src"] for r in rb], f"{n}: item order differs"
        m, p = vm.two_model_average([r["margin"] for r in ra], [r["margin"] for r in rb])
        out[n] = {"rows": [dict(r, margin=float(mm), score=float(pp)) for r, mm, pp in zip(ra, m, p)]}
    return out


def summarise(reps):
    """reps: list of dev reports, one per training run. Mean and range of every numeric key."""
    keys = [k for k, v in reps[0].items() if isinstance(v, (int, float)) and v is not None]
    out = {}
    for k in keys:
        vals = [r[k] for r in reps if r.get(k) is not None]
        if len(vals) == len(reps):
            out[k] = {"mean": float(np.mean(vals)), "range": float(max(vals) - min(vals)),
                      "values": vals}
    return out


def choose_candidate(cands, trace):
    """Pick the leading candidate by intent score. A single model must beat the base recipe (R1)
    by more than the larger of their run-to-run ranges, and a two-model average must beat that
    single model the same way. Returns the chosen candidate."""
    di = lambda c: cands[c]["dev_intent"]["mean"]           # noqa: E731
    rg = lambda c: cands[c]["dev_intent"]["range"]          # noqa: E731
    singles = sorted([c for c in cands if c in SINGLE_MODELS], key=di, reverse=True)
    best = singles[0]
    if best != "R1" and di(best) - di("R1") < max(rg(best), rg("R1")):
        trace.append(f"best single model: {best} leads R1 by {di(best) - di('R1'):.4f}, less than "
                     f"the larger run-to-run range {max(rg(best), rg('R1')):.4f}, so R1 stays")
        best = "R1"
    else:
        trace.append(f"best single model: {best} (mean intent score {di(best):.4f})")
    averages = sorted([c for c in cands if c in TWO_MODEL_AVERAGES], key=di, reverse=True)
    if averages:
        t = averages[0]
        lead, need = di(t) - di(best), max(rg(t), rg(best))
        if lead > need:
            trace.append(f"best two-model average: {t} leads {best} by {lead:.4f} > {need:.4f}, "
                         f"so {t} replaces it")
            return t
        trace.append(f"best two-model average: {t} leads {best} by {lead:.4f}, not > {need:.4f}; "
                     f"{best} stands")
    return best


def select(cands):
    """Choose a candidate, then check it against the base recipe's tc_val AUPRC. A candidate that
    trails R1 by more than TOXICCHAT_AUPRC_MARGIN is removed and the choice is made again from
    the rest, until one passes. R1 always passes. Returns (selected, trace)."""
    if "R1" not in cands:
        raise SystemExit("R1 does not have three dev-scored training runs; nothing can be selected")
    trace, remaining = [], dict(cands)
    r1 = cands["R1"]["tc_val_auprc"]["mean"]
    while True:
        c = choose_candidate(remaining, trace)
        gap = r1 - cands[c]["tc_val_auprc"]["mean"]
        if c == "R1" or gap <= TOXICCHAT_AUPRC_MARGIN:
            trace.append(f"ToxicChat check: {c} tc_val AUPRC {cands[c]['tc_val_auprc']['mean']:.4f} "
                         f"(R1 {r1:.4f}; gap {gap:.4f} <= {TOXICCHAT_AUPRC_MARGIN}): selected")
            return c, trace
        trace.append(f"ToxicChat check: {c} fails (tc_val AUPRC gap {gap:.4f} > "
                     f"{TOXICCHAT_AUPRC_MARGIN}); removed, and the choice is made again")
        del remaining[c]


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--results", default="v6_results")
    a.add_argument("--out", default="V6_SELECTION.md")
    a.add_argument("--dev-scorecards", default=None,
                   help="read <dir>/<run>/<set>.json (the evidence bundle's scorecards/dev, labelled)")
    a = a.parse_args(argv)
    rerun = a.dev_scorecards is not None

    runs = {}
    for variant in SINGLE_MODELS:
        for s in SEEDS:
            c = load_run(a.results, f"{variant}_s{s}", a.dev_scorecards)
            if c is not None:
                runs[(variant, s)] = c
    reports, cands, runs_of = {}, {}, {}
    for (variant, s), c in runs.items():
        reports[f"{variant}_s{s}"] = vm.dev_report(dev_sets_for_report(c))
    for variant in SINGLE_MODELS:
        if all((variant, s) in runs for s in SEEDS):
            runs_of[variant] = [f"{variant}_s{s}" for s in SEEDS]
            cands[variant] = summarise([reports[r] for r in runs_of[variant]])
    for name, pairs in TWO_MODEL_AVERAGES.items():
        if all((x, s) in runs and (y, t) in runs for x, s, y, t in pairs):
            reps = []
            for x, s, y, t in pairs:
                key = f"{name}[{x}_s{s}+{y}_s{t}]"
                reports[key] = vm.dev_report(dev_sets_for_report(combine(runs[(x, s)], runs[(y, t)])))
                reps.append(key)
            runs_of[name] = reps
            cands[name] = summarise([reports[r] for r in reps])

    sel, trace = select(cands)
    s = cands[sel]
    gates = {"G1": {"value": s["dev_intent"]["mean"], "bar": INTENT_SCORE_BAR,
                    "pass": s["dev_intent"]["mean"] >= INTENT_SCORE_BAR},
             "G2": {"value": s["tc_val_auprc"]["mean"], "bar": TOXICCHAT_AUPRC_BAR,
                    "pass": s["tc_val_auprc"]["mean"] >= TOXICCHAT_AUPRC_BAR},
             "G3": {"value": s["saturation"]["mean"], "bar": PINNED_SCORE_LIMIT,
                    "pass": s["saturation"]["mean"] < PINNED_SCORE_LIMIT}}
    shipped = runs_of[sel][0]
    thr = {r: {k: reports[r][k] for k in ("t_gate", "t_screen", "t_bal")} for r in runs_of[sel]}
    out = {"B": RELEASE_1_TRIAL_INTENT_SCORE, "candidates": cands, "replicates": runs_of,
           "run_reports": reports, "selected": sel, "trace": trace, "gates": gates,
           "all_gates_pass": all(g["pass"] for g in gates.values()),
           "shipped_replicate": shipped, "thresholds": thr}
    pathlib.Path(a.results).mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.results, "selection.json").write_text(json.dumps(out, indent=1))

    source = ("Recomputed by `v6_select.py` from the evidence bundle's development scorecards."
              if rerun else
              "Computed by `v6_select.py` from per-item dev scorecards only. No test set has been scored "
              "for any v6 checkpoint.")
    L = ["# V6_SELECTION: dev-set selection", "",
         f"{source} Full numbers: `{pathlib.Path(a.results, 'selection.json')}`.", "",
         f"Release 1.0 trial intent score = {RELEASE_1_TRIAL_INTENT_SCORE:.4f}; intent score bar = "
         f"{INTENT_SCORE_BAR:.4f}; ToxicChat AUPRC bar = {TOXICCHAT_AUPRC_BAR}; share of scores "
         f"pinned near 0 or 1 must stay below {PINNED_SCORE_LIMIT}.", "",
         "## Candidates (mean over three training runs [range])", "",
         "| Candidate | What it is | Training runs | Intent score | Twins AUROC | dev_pair2 AUROC | Pair acc | tc_val AUPRC | aegis_val AUROC | Pinned scores |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda c, k: f"{cands[c][k]['mean']:.4f} [{cands[c][k]['range']:.4f}]"  # noqa: E731
    for c in cands:
        L.append(f"| {c} | {CANDIDATE_NAMES.get(c, '')} | {', '.join(runs_of[c])} | {f(c, 'dev_intent')} | "
                 f"{f(c, 'twins_auroc')} | {f(c, 'dev_pair2_auroc')} | {f(c, 'twins_pair_acc')} | "
                 f"{f(c, 'tc_val_auprc')} | {f(c, 'aegis_val_auroc')} | {f(c, 'saturation')} |")
    L += ["", "## Every run", "",
          "| Run | Intent score | Twins | dev_pair2 | tc_val AUPRC | aegis_val AUROC | Held-back WildGuardMix AUROC | Held-back WildJailbreak AUROC | Held-back FalseReject FRR at strict | Pinned scores | Strict | Pre-screen | Balanced |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, d in reports.items():
        L.append(f"| {r} | {d['dev_intent']:.4f} | {d['twins_auroc']:.4f} | {d['dev_pair2_auroc']:.4f} | "
                 f"{d['tc_val_auprc']:.4f} | {d['aegis_val_auroc']:.4f} | {d['dev_holdout_p3_auroc']:.4f} | "
                 f"{d['dev_holdout_p4_auroc']:.4f} | {d['dev_holdout_p5_frr_at_gate']:.4f} | {d['saturation']:.4f} | "
                 f"{d['t_gate']!r} | {d['t_screen']!r} | {d['t_bal']!r} |")
    L += ["", "## Selection", ""] + [f"- {t}" for t in trace]
    L += ["", f"**Selected: {name_of(sel)}.** Shipped model (fixed before training): the first "
          f"training run, `{shipped}`.", "",
          "## Release bars (on the selected candidate's mean over its training runs)", ""]
    for k, g in gates.items():
        L.append(f"- {GATE_NAMES[k]}: {g['value']:.4f} vs {g['bar']:.4f}: **{'PASS' if g['pass'] else 'FAIL'}**")
    L += ["", f"**All bars: {'PASS: sealing this file unlocks the test sets' if out['all_gates_pass'] else 'FAIL: the test sets stay locked'}.**",
          "", "## Frozen thresholds (per training run, on P(unsafe) = sigmoid(margin))", "",
          "| Training run | Strict | Pre-screen | Balanced |", "|---|---|---|---|"]
    for r, t in thr.items():
        L.append(f"| {r}{' (shipped)' if r == shipped else ''} | {t['t_gate']!r} | {t['t_screen']!r} | {t['t_bal']!r} |")
    if rerun:
        L += [""]
    else:
        # The Record states how the selection was made. Those statements are written, not computed;
        # R4's figures are computed like every other number in this file.
        r4 = cands.get("R4")
        r4_line = (f"- R4 was added after the first round's dev results had been seen and before any v6 "
                   f"test score. Its outcome here: intent score {r4['dev_intent']['mean']:.4f} (range "
                   f"{r4['dev_intent']['range']:.4f}), tc_val AUPRC {r4['tc_val_auprc']['mean']:.4f}. "
                   f"**R4 was {'' if sel == 'R4' else 'not '}selected.**" if r4 else
                   "- R4 has no three dev-scored training runs.")
        L += ["", "## Record", "",
              "- One selection, made after every launched run was dev-scored: the first round of "
              "training runs (R1, R0, R2), then R3 and R4. The decision to train R3 used no dev result.",
              r4_line,
              "- Dev results could be seen as soon as each run finished. None of them could change this "
              "selection, since the selection rule is mechanical.",
              "- No v6 checkpoint has been scored on any test set.", "",
              "The selection is not revisited after any test set is seen.", ""]
    pathlib.Path(a.out).write_text("\n".join(L))
    print("\n".join(trace))
    print(json.dumps({"selected": sel, "selected_name": CANDIDATE_NAMES.get(sel, sel),
                      "gates": gates, "shipped": shipped}, indent=1))


if __name__ == "__main__":
    main()
