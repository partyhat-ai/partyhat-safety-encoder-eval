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
    "E12": [("R1", seed, "R2", seed) for seed in SEEDS],
    "E13": [("R1", seed, "R3", seed) for seed in SEEDS],
    "E23": [("R2", seed, "R3", seed) for seed in SEEDS],
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
    folder = (pathlib.Path(dev_scorecards) / run if dev_scorecards
              else pathlib.Path(results) / "runs" / run / "scorecards")
    if not all((folder / f"{set_name}.json").exists() for set_name in DEV_SETS):
        return None
    return {set_name: json.load(open(folder / f"{set_name}.json")) for set_name in DEV_SETS}


def combine(scorecards_a, scorecards_b):
    """Two-model average scorecards: mean margin, sigmoid score; items must align by src."""
    out = {}
    for set_name in DEV_SETS:
        rows_a, rows_b = scorecards_a[set_name]["rows"], scorecards_b[set_name]["rows"]
        assert [row["src"] for row in rows_a] == [row["src"] for row in rows_b], \
            f"{set_name}: item order differs"
        margins, scores = vm.two_model_average([row["margin"] for row in rows_a],
                                               [row["margin"] for row in rows_b])
        out[set_name] = {"rows": [dict(row, margin=float(margin), score=float(score))
                                  for row, margin, score in zip(rows_a, margins, scores)]}
    return out


def summarise(run_reports):
    """run_reports: list of dev reports, one per training run. Mean and range of every numeric key."""
    keys = [key for key, value in run_reports[0].items()
            if isinstance(value, (int, float)) and value is not None]
    out = {}
    for key in keys:
        values = [report[key] for report in run_reports if report.get(key) is not None]
        if len(values) == len(run_reports):
            out[key] = {"mean": float(np.mean(values)), "range": float(max(values) - min(values)),
                        "values": values}
    return out


def mean_intent(cands, candidate):
    """The candidate's intent score, averaged over its training runs."""
    return cands[candidate]["dev_intent"]["mean"]


def intent_range(cands, candidate):
    """The spread (max - min) of the candidate's intent score across its training runs."""
    return cands[candidate]["dev_intent"]["range"]


def choose_candidate(cands, trace):
    """Pick the leading candidate by intent score. A single model must beat the base recipe (R1)
    by more than the larger of their run-to-run ranges, and a two-model average must beat that
    single model the same way. Returns the chosen candidate."""
    def intent(candidate):
        return mean_intent(cands, candidate)

    def spread(candidate):
        return intent_range(cands, candidate)

    singles = sorted([cand for cand in cands if cand in SINGLE_MODELS], key=intent, reverse=True)
    best = singles[0]
    if best != "R1" and intent(best) - intent("R1") < max(spread(best), spread("R1")):
        trace.append(f"best single model: {best} leads R1 by {intent(best) - intent('R1'):.4f}, less than "
                     f"the larger run-to-run range {max(spread(best), spread('R1')):.4f}, so R1 stays")
        best = "R1"
    else:
        trace.append(f"best single model: {best} (mean intent score {intent(best):.4f})")
    averages = sorted([cand for cand in cands if cand in TWO_MODEL_AVERAGES], key=intent, reverse=True)
    if averages:
        top_average = averages[0]
        lead, needed = intent(top_average) - intent(best), max(spread(top_average), spread(best))
        if lead > needed:
            trace.append(f"best two-model average: {top_average} leads {best} by {lead:.4f} > {needed:.4f}, "
                         f"so {top_average} replaces it")
            return top_average
        trace.append(f"best two-model average: {top_average} leads {best} by {lead:.4f}, not > {needed:.4f}; "
                     f"{best} stands")
    return best


def select(cands):
    """Choose a candidate, then check it against the base recipe's tc_val AUPRC. A candidate that
    trails R1 by more than TOXICCHAT_AUPRC_MARGIN is removed and the choice is made again from
    the rest, until one passes. R1 always passes. Returns (selected, trace)."""
    if "R1" not in cands:
        raise SystemExit("R1 does not have three dev-scored training runs; nothing can be selected")
    trace, remaining = [], dict(cands)
    r1_auprc = cands["R1"]["tc_val_auprc"]["mean"]
    while True:
        candidate = choose_candidate(remaining, trace)
        candidate_auprc = cands[candidate]["tc_val_auprc"]["mean"]
        gap = r1_auprc - candidate_auprc
        if candidate == "R1" or gap <= TOXICCHAT_AUPRC_MARGIN:
            trace.append(f"ToxicChat check: {candidate} tc_val AUPRC {candidate_auprc:.4f} "
                         f"(R1 {r1_auprc:.4f}; gap {gap:.4f} <= {TOXICCHAT_AUPRC_MARGIN}): selected")
            return candidate, trace
        trace.append(f"ToxicChat check: {candidate} fails (tc_val AUPRC gap {gap:.4f} > "
                     f"{TOXICCHAT_AUPRC_MARGIN}); removed, and the choice is made again")
        del remaining[candidate]


def build_candidates(results, dev_scorecards):
    """Load every training run's dev scorecards and summarise each eligible candidate.

    Returns (run_reports, cands, runs_of): the dev report of every training run (and of every
    two-model pairing), each candidate's summary over its runs, and the runs behind each candidate.
    """
    scorecards = {}
    for variant in SINGLE_MODELS:
        for seed in SEEDS:
            run_scorecards = load_run(results, f"{variant}_s{seed}", dev_scorecards)
            if run_scorecards is not None:
                scorecards[(variant, seed)] = run_scorecards
    run_reports, cands, runs_of = {}, {}, {}
    for (variant, seed), run_scorecards in scorecards.items():
        run_reports[f"{variant}_s{seed}"] = vm.dev_report(dev_sets_for_report(run_scorecards))
    for variant in SINGLE_MODELS:
        if all((variant, seed) in scorecards for seed in SEEDS):
            runs_of[variant] = [f"{variant}_s{seed}" for seed in SEEDS]
            cands[variant] = summarise([run_reports[run] for run in runs_of[variant]])
    for name, pairs in TWO_MODEL_AVERAGES.items():
        if all((variant_a, seed_a) in scorecards and (variant_b, seed_b) in scorecards
               for variant_a, seed_a, variant_b, seed_b in pairs):
            pair_runs = []
            for variant_a, seed_a, variant_b, seed_b in pairs:
                key = f"{name}[{variant_a}_s{seed_a}+{variant_b}_s{seed_b}]"
                averaged = combine(scorecards[(variant_a, seed_a)], scorecards[(variant_b, seed_b)])
                run_reports[key] = vm.dev_report(dev_sets_for_report(averaged))
                pair_runs.append(key)
            runs_of[name] = pair_runs
            cands[name] = summarise([run_reports[run] for run in pair_runs])
    return run_reports, cands, runs_of


def check_gates(summary):
    """The three release bars, checked on a candidate's summary (means over its training runs)."""
    intent = summary["dev_intent"]["mean"]
    auprc = summary["tc_val_auprc"]["mean"]
    pinned = summary["saturation"]["mean"]
    return {"G1": {"value": intent, "bar": INTENT_SCORE_BAR, "pass": intent >= INTENT_SCORE_BAR},
            "G2": {"value": auprc, "bar": TOXICCHAT_AUPRC_BAR, "pass": auprc >= TOXICCHAT_AUPRC_BAR},
            "G3": {"value": pinned, "bar": PINNED_SCORE_LIMIT, "pass": pinned < PINNED_SCORE_LIMIT}}


def mean_and_range(cands, candidate, metric):
    """Table cell: the candidate's mean for a metric, with its run-to-run range in brackets."""
    stats = cands[candidate][metric]
    return f"{stats['mean']:.4f} [{stats['range']:.4f}]"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="v6_results")
    parser.add_argument("--out", default="V6_SELECTION.md")
    parser.add_argument("--dev-scorecards", default=None,
                        help="read <dir>/<run>/<set>.json (the evidence bundle's scorecards/dev, labelled)")
    args = parser.parse_args(argv)
    rerun = args.dev_scorecards is not None

    run_reports, cands, runs_of = build_candidates(args.results, args.dev_scorecards)
    selected, trace = select(cands)
    gates = check_gates(cands[selected])
    shipped = runs_of[selected][0]
    thresholds = {run: {key: run_reports[run][key] for key in ("t_gate", "t_screen", "t_bal")}
                  for run in runs_of[selected]}
    out = {"B": RELEASE_1_TRIAL_INTENT_SCORE, "candidates": cands, "replicates": runs_of,
           "run_reports": run_reports, "selected": selected, "trace": trace, "gates": gates,
           "all_gates_pass": all(gate["pass"] for gate in gates.values()),
           "shipped_replicate": shipped, "thresholds": thresholds}
    pathlib.Path(args.results).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results, "selection.json").write_text(json.dumps(out, indent=1))

    source = ("Recomputed by `v6_select.py` from the evidence bundle's development scorecards."
              if rerun else
              "Computed by `v6_select.py` from per-item dev scorecards only. No test set has been scored "
              "for any v6 checkpoint.")
    lines = ["# V6_SELECTION: dev-set selection", "",
             f"{source} Full numbers: `{pathlib.Path(args.results, 'selection.json')}`.", "",
             f"Release 1.0 trial intent score = {RELEASE_1_TRIAL_INTENT_SCORE:.4f}; intent score bar = "
             f"{INTENT_SCORE_BAR:.4f}; ToxicChat AUPRC bar = {TOXICCHAT_AUPRC_BAR}; share of scores "
             f"pinned near 0 or 1 must stay below {PINNED_SCORE_LIMIT}.", "",
             "## Candidates (mean over three training runs [range])", "",
             "| Candidate | What it is | Training runs | Intent score | Twins AUROC | dev_pair2 AUROC | Pair acc | tc_val AUPRC | aegis_val AUROC | Pinned scores |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cand in cands:
        cells = [mean_and_range(cands, cand, metric)
                 for metric in ("dev_intent", "twins_auroc", "dev_pair2_auroc", "twins_pair_acc",
                                "tc_val_auprc", "aegis_val_auroc", "saturation")]
        lines.append(f"| {cand} | {CANDIDATE_NAMES.get(cand, '')} | {', '.join(runs_of[cand])} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## Every run", "",
              "| Run | Intent score | Twins | dev_pair2 | tc_val AUPRC | aegis_val AUROC | Held-back WildGuardMix AUROC | Held-back WildJailbreak AUROC | Held-back FalseReject FRR at strict | Pinned scores | Strict | Pre-screen | Balanced |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, report in run_reports.items():
        lines.append(f"| {run} | {report['dev_intent']:.4f} | {report['twins_auroc']:.4f} | "
                     f"{report['dev_pair2_auroc']:.4f} | {report['tc_val_auprc']:.4f} | "
                     f"{report['aegis_val_auroc']:.4f} | {report['dev_holdout_p3_auroc']:.4f} | "
                     f"{report['dev_holdout_p4_auroc']:.4f} | {report['dev_holdout_p5_frr_at_gate']:.4f} | "
                     f"{report['saturation']:.4f} | {report['t_gate']!r} | {report['t_screen']!r} | "
                     f"{report['t_bal']!r} |")
    lines += ["", "## Selection", ""] + [f"- {step}" for step in trace]
    lines += ["", f"**Selected: {name_of(selected)}.** Shipped model (fixed before training): the first "
              f"training run, `{shipped}`.", "",
              "## Release bars (on the selected candidate's mean over its training runs)", ""]
    for gate_id, gate in gates.items():
        verdict = "PASS" if gate["pass"] else "FAIL"
        lines.append(f"- {GATE_NAMES[gate_id]}: {gate['value']:.4f} vs {gate['bar']:.4f}: **{verdict}**")
    all_bars = ("PASS: sealing this file unlocks the test sets" if out["all_gates_pass"]
                else "FAIL: the test sets stay locked")
    lines += ["", f"**All bars: {all_bars}.**",
              "", "## Frozen thresholds (per training run, on P(unsafe) = sigmoid(margin))", "",
              "| Training run | Strict | Pre-screen | Balanced |", "|---|---|---|---|"]
    for run, run_thresholds in thresholds.items():
        label = f"{run} (shipped)" if run == shipped else run
        lines.append(f"| {label} | {run_thresholds['t_gate']!r} | {run_thresholds['t_screen']!r} | "
                     f"{run_thresholds['t_bal']!r} |")
    if rerun:
        lines += [""]
    else:
        # The Record states how the selection was made. Those statements are written, not computed;
        # R4's figures are computed like every other number in this file.
        r4 = cands.get("R4")
        r4_line = (f"- R4 was added after the first round's dev results had been seen and before any v6 "
                   f"test score. Its outcome here: intent score {r4['dev_intent']['mean']:.4f} (range "
                   f"{r4['dev_intent']['range']:.4f}), tc_val AUPRC {r4['tc_val_auprc']['mean']:.4f}. "
                   f"**R4 was {'' if selected == 'R4' else 'not '}selected.**" if r4 else
                   "- R4 has no three dev-scored training runs.")
        lines += ["", "## Record", "",
                  "- One selection, made after every launched run was dev-scored: the first round of "
                  "training runs (R1, R0, R2), then R3 and R4. The decision to train R3 used no dev result.",
                  r4_line,
                  "- Dev results could be seen as soon as each run finished. None of them could change this "
                  "selection, since the selection rule is mechanical.",
                  "- No v6 checkpoint has been scored on any test set.", "",
                  "The selection is not revisited after any test set is seen.", ""]
    pathlib.Path(args.out).write_text("\n".join(lines))
    print("\n".join(trace))
    print(json.dumps({"selected": selected, "selected_name": CANDIDATE_NAMES.get(selected, selected),
                      "gates": gates, "shipped": shipped}, indent=1))


if __name__ == "__main__":
    main()
