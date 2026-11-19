#!/usr/bin/env python3
"""v6_committee_targets.py: turns the teacher committee's verdicts into training labels for recipe
variant R3 (base recipe + teacher labels). This is the CPU half; the Modal half scores the rows.

Turns committee margins into training targets. It does no model inference: the committee scorer
(r3_committee/r3_committee_modal.py) scores rows on Modal and hands this script one margin file per member.

Inputs (JSONL):
  --rows   R3's rows: row_id, user_input, pool, toxicity (0/1, or null when the row has no source
           label), toxicity_soft (the pool's soft target, or null when there is no source label)
  --member NAME=path   repeated; rows of {row_id, margin}. The margin is the verdict-token margin,
           unsafe minus safe (for Qwen3Guard: Unsafe against Safe-or-Controversial). Rows the member
           did not score are simply absent.
  --calib  NAME=path   repeated; rows of {margin, y} for that member on tc_val + aegis_val.

Rules:
  * each member is Platt-scaled on its calibration rows; the committee log-odds is the mean of the
    calibrated log-odds over the members that scored the row; p_c is its sigmoid;
  * a row needs at least --min_members (default 2) members to count as scored;
  * source label kept, unless p_c >= 0.9 on a row labelled safe or p_c <= 0.1 on a row labelled
    unsafe: then the label is disputed and the target is 0.5;
  * no source label: target = p_c clipped to [0.02, 0.98], toxicity = 1 when p_c >= 0.5;
  * a source-labelled row that was not scored keeps its target; an unlabelled row that was not
    scored is dropped (there is nothing to train it on).

Output: --out (JSONL: every kept row with toxicity, toxicity_soft, p_c, target_rule) and
--report (JSON: counts per pool, Platt parameters). Prints the report; never prints text.
A disputed label is recorded as target_rule "clash" and counted in clash_rows; those names are
kept because the stored files use them.
A member whose Platt slope is not positive stops the run: its margin has the wrong sign.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def platt(margins, y):
    """Return (a, b) with calibrated log-odds = a * margin + b."""
    from sklearn.linear_model import LogisticRegression
    m, y = np.asarray(margins, float).reshape(-1, 1), np.asarray(y, int)
    if not (np.isfinite(m).all() and 0 < y.sum() < len(y)):
        raise SystemExit("calibration rows need finite margins and both classes")
    # Fit on the standardised margin, then convert back: the fit is then the same whatever scale a
    # member's logits live on.
    mu, sd = float(m.mean()), float(m.std()) or 1.0
    clf = LogisticRegression(C=1e6, max_iter=5000, tol=1e-8).fit((m - mu) / sd, y)
    a = float(clf.coef_[0, 0]) / sd
    return a, float(clf.intercept_[0]) - a * mu


def build_targets(rows, member_margins, calib, min_members=2, hi=0.9, lo=0.1, clip=(0.02, 0.98)):
    """rows: list of dicts. member_margins: {name: {row_id: margin}}. calib: {name: (margins, y)}."""
    params = {}
    for name, (m, y) in calib.items():
        a, b = platt(m, y)
        if a <= 0:
            raise SystemExit(f"member {name}: Platt slope {a:.4f} is not positive; the margin's sign is wrong")
        params[name] = {"a": a, "b": b, "n_calib": int(len(y))}
    missing = set(member_margins) - set(params)
    if missing:
        raise SystemExit(f"members without calibration rows: {sorted(missing)}")
    out, counts = [], defaultdict(Counter)
    for r in rows:
        rid, pool = r["row_id"], r.get("pool", "?")
        lods = [params[n]["a"] * mm[rid] + params[n]["b"] for n, mm in member_margins.items() if rid in mm]
        scored = len(lods) >= min_members
        p_c = float(1.0 / (1.0 + np.exp(-np.mean(lods)))) if scored else None
        y = r.get("toxicity")
        new = dict(r)
        if y is None:
            if not scored:
                counts[pool]["dropped_unlabelled_unscored"] += 1
                continue
            new["toxicity"] = int(p_c >= 0.5)
            new["toxicity_soft"] = float(min(max(p_c, clip[0]), clip[1]))
            rule = "committee"
            counts[pool]["committee_unsafe" if p_c >= 0.5 else "committee_safe"] += 1
        else:
            y = int(y)
            base = r.get("toxicity_soft")
            new["toxicity"] = y
            new["toxicity_soft"] = float(y) if base is None else float(base)
            rule = "source" if scored else "source_unscored"
            if scored and ((y == 0 and p_c >= hi) or (y == 1 and p_c <= lo)):
                new["toxicity_soft"] = 0.5
                rule = "clash"
            counts[pool][rule] += 1
        new["p_c"], new["target_rule"] = p_c, rule
        if not 0.0 <= new["toxicity_soft"] <= 1.0:
            raise SystemExit(f"row {rid}: target {new['toxicity_soft']} is outside [0, 1]")
        out.append(new)
    report = {"members": params, "min_members": min_members, "rows_in": len(rows), "rows_out": len(out),
              "per_pool": {p: dict(c) for p, c in sorted(counts.items())}}
    unl = sum(c["committee_unsafe"] + c["committee_safe"] for c in counts.values())
    report["unlabelled_share_unsafe"] = (sum(c["committee_unsafe"] for c in counts.values()) / unl) if unl else None
    report["clash_rows"] = int(sum(c["clash"] for c in counts.values()))
    return out, report


def _pairs(items):
    out = {}
    for it in items or []:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=path, got {it}")
        out[name] = path
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True)
    p.add_argument("--member", action="append", required=True)
    p.add_argument("--calib", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--min_members", type=int, default=2)
    a = p.parse_args(argv)
    rows = read_jsonl(a.rows)
    ids = [r["row_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("row_id values are not unique")
    members = {n: {r["row_id"]: float(r["margin"]) for r in read_jsonl(path)}
               for n, path in _pairs(a.member).items()}
    calib = {}
    for n, path in _pairs(a.calib).items():
        c = read_jsonl(path)
        calib[n] = ([float(r["margin"]) for r in c], [int(r["y"]) for r in c])
    out, report = build_targets(rows, members, calib, a.min_members)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
