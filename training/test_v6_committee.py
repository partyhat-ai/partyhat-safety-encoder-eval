"""Tests for v6_committee_targets.py (the teacher-committee targets for R3). CPU only:  python3 -m pytest -q test_v6_committee.py"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import v6_committee_targets as ct  # noqa: E402


def calib(rng, n=400, scale=1.0, shift=0.0):
    y = rng.integers(0, 2, n)
    return (list(scale * (rng.normal(0, 1, n) + 2.5 * (2 * y - 1)) + shift), list(y))


def test_rules_and_counts():
    rng = np.random.default_rng(0)
    cal = {"A": calib(rng), "B": calib(rng, scale=3.0, shift=1.0), "C": calib(rng)}
    rows = [
        {"row_id": 1, "user_input": "t1", "pool": "P4", "toxicity": 0, "toxicity_soft": 0.05},   # agrees
        {"row_id": 2, "user_input": "t2", "pool": "P4", "toxicity": 0, "toxicity_soft": 0.05},   # disputed label
        {"row_id": 3, "user_input": "t3", "pool": "P5", "toxicity": 1, "toxicity_soft": 0.95},   # disputed label
        {"row_id": 4, "user_input": "t4", "pool": "wildchat", "toxicity": None, "toxicity_soft": None},
        {"row_id": 5, "user_input": "t5", "pool": "wildchat", "toxicity": None, "toxicity_soft": None},
        {"row_id": 6, "user_input": "t6", "pool": "wildchat", "toxicity": None, "toxicity_soft": None},  # unscored
        {"row_id": 7, "user_input": "t7", "pool": "P1", "toxicity": 1},                           # never scored
        {"row_id": 8, "user_input": "t8", "pool": "P4", "toxicity": 1, "toxicity_soft": 0.95},   # one member only
    ]
    m = {"A": {1: -6.0, 2: 7.0, 3: -7.0, 4: 6.0, 5: -0.2, 8: -9.0},
         "B": {1: -18.0, 2: 21.0, 3: -21.0, 4: 18.0, 5: -0.5},
         "C": {1: -6.0, 2: 7.0, 3: -7.0, 4: 6.0, 5: 0.1}}
    out, rep = ct.build_targets(rows, m, cal)
    by = {r["row_id"]: r for r in out}
    assert by[1]["target_rule"] == "source" and by[1]["toxicity_soft"] == 0.05
    assert by[2]["target_rule"] == "clash" and by[2]["toxicity_soft"] == 0.5 and by[2]["toxicity"] == 0
    assert by[3]["target_rule"] == "clash" and by[3]["toxicity_soft"] == 0.5 and by[3]["toxicity"] == 1
    assert by[4]["target_rule"] == "committee" and by[4]["toxicity"] == 1 and by[4]["toxicity_soft"] == 0.98
    assert by[5]["target_rule"] == "committee" and 0.02 <= by[5]["toxicity_soft"] <= 0.98
    assert 6 not in by, "an unlabelled row nobody scored is dropped"
    assert by[7]["target_rule"] == "source_unscored" and by[7]["toxicity_soft"] == 1.0
    assert by[8]["target_rule"] == "source_unscored", "one member is below min_members: no clash call"
    assert rep["clash_rows"] == 2 and rep["rows_out"] == 7
    assert rep["per_pool"]["wildchat"]["dropped_unlabelled_unscored"] == 1
    assert all(p["a"] > 0 for p in rep["members"].values())
    assert all(0.0 <= r["toxicity_soft"] <= 1.0 for r in out)


def test_wrong_sign_margin_is_refused():
    rng = np.random.default_rng(1)
    m, y = calib(rng)
    with pytest.raises(SystemExit):
        ct.build_targets([], {"A": {}}, {"A": ([-v for v in m], y)})


def test_scale_invariance_of_calibration():
    """A member whose margins are on a different scale must give the same calibrated verdicts."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    m = rng.normal(0, 1, 2000) + 1.0 * (2 * y - 1)  # overlapping classes, as real guard margins are
    a1, b1 = ct.platt(list(m), list(y))
    a2, b2 = ct.platt(list(3 * m + 7), list(y))
    x = np.linspace(-3, 3, 13)
    assert np.allclose(a1 * x + b1, a2 * (3 * x + 7) + b2, atol=1e-3)
    assert 1.6 < a1 < 2.4, "for unit-variance classes at +/-1 the true slope is 2"


def test_cli_round_trip(tmp_path):
    rng = np.random.default_rng(3)

    def w(name, rows):
        path = tmp_path / name
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return str(path)

    rows = [{"row_id": i, "user_input": f"t{i}", "pool": "P4", "toxicity": i % 2,
             "toxicity_soft": 0.95 if i % 2 else 0.05} for i in range(20)]
    rows += [{"row_id": 100 + i, "user_input": f"u{i}", "pool": "harmaug", "toxicity": None,
              "toxicity_soft": None} for i in range(10)]
    args = ["--rows", w("rows.jsonl", rows), "--out", str(tmp_path / "out.jsonl"),
            "--report", str(tmp_path / "rep.json")]
    for n in ("A", "B"):
        cm, cy = calib(rng)
        args += ["--calib", f"{n}=" + w(f"cal_{n}.jsonl", [{"margin": a, "y": int(b)} for a, b in zip(cm, cy)])]
        args += ["--member", f"{n}=" + w(f"m_{n}.jsonl", [{"row_id": r["row_id"], "margin": float(rng.normal(0, 3))}
                                                          for r in rows])]
    rep = ct.main(args)
    out = [json.loads(l) for l in open(tmp_path / "out.jsonl")]
    assert len(out) == 30 and rep["rows_out"] == 30
    assert all(r["toxicity"] in (0, 1) and r["toxicity_soft"] is not None for r in out)
