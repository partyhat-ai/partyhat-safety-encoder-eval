"""Unit tests for v6_metrics.py: dev metrics, the frozen thresholds and the two-model average. CPU, numpy only."""
import math

import numpy as np
import pytest

import v6_metrics as vm


def test_probs_from_margin_matches_softmax_and_does_not_saturate():
    m = np.array([-50.0, -3.0, 0.0, 2.5, 40.0])
    l0, l1 = np.zeros_like(m), m
    sm = np.exp(l1) / (np.exp(l0) + np.exp(l1))
    p = vm.probs_from_margin(m)
    assert np.allclose(p, sm, rtol=0, atol=1e-15)
    assert p[0] > 0 and p[-1] <= 1.0
    assert np.all(np.diff(vm.probs_from_margin(np.array([-40.0, -39.0, -38.0]))) > 0)


def test_two_model_average_is_mean_margin():
    m, p = vm.two_model_average([2.0, -4.0], [0.0, 2.0])
    assert np.allclose(m, [1.0, -1.0])
    assert np.allclose(p, 1 / (1 + np.exp(-m)))


def test_strict_threshold_budget_and_ties():
    rng = np.random.default_rng(0)
    b = rng.random(1000)
    t = vm.strict_threshold(b)
    assert (b >= t).mean() <= 0.01
    # lowest such threshold: the next-lower candidate (the (k+1)-th largest itself) breaks budget
    k = math.floor(0.01 * b.size)
    assert (b >= np.sort(b)[::-1][k]).mean() > 0.01
    # ties at the boundary are never flagged past the budget
    b2 = np.r_[np.full(50, 0.9), np.zeros(950)]
    t2 = vm.strict_threshold(b2)
    assert (b2 >= t2).mean() <= 0.01 and t2 > 0.9


def test_prescreen_threshold_recall_and_ties():
    u = np.r_[np.linspace(0.1, 0.9, 98), [0.05, 0.01]]
    t = vm.prescreen_threshold(u)
    assert (u >= t).mean() >= 0.98
    assert t == pytest.approx(0.1)   # the 98th largest
    tied = np.r_[np.full(10, 0.5), np.ones(90)]
    ts = vm.prescreen_threshold(tied)
    assert ts == 0.5 and (tied >= ts).mean() == 1.0
    # k = ceil(0.98 * n): n = 51 -> k = 50, so one item may fall below
    u51 = np.arange(51, dtype=float)
    assert vm.prescreen_threshold(u51) == 1.0


def test_balanced_threshold_maximises_mean_f1_and_breaks_ties_low():
    y1, s1 = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    y2, s2 = np.array([0, 1]), np.array([0.3, 0.7])
    t, f = vm.balanced_threshold([(y1, s1), (y2, s2)])
    assert f == pytest.approx(1.0)
    assert t == pytest.approx(0.7)   # lowest threshold reaching mean F1 = 1 is 0.7 (0.3 < t <= 0.7)
    # brute force check on random data
    rng = np.random.default_rng(1)
    sets = [(rng.integers(0, 2, 60), rng.random(60)) for _ in range(3)]
    t, f = vm.balanced_threshold(sets)
    import metrics_core as mc
    cand = np.unique(np.concatenate([s for _, s in sets]))
    brute = [np.mean([mc.f1_at(y, s, c) for y, s in sets]) for c in cand]
    assert f == pytest.approx(max(brute))
    assert t == pytest.approx(cand[int(np.argmax(brute))])


def test_pair_accuracy_ties_half():
    acc, n = vm.pair_accuracy([0.9, 0.1, 0.5, 0.5, 0.2, 0.3], [1, 0, 1, 0, 1, 0],
                              ["a", "a", "b", "b", "c", "c"])
    assert n == 3 and acc == pytest.approx((1 + 0.5 + 0) / 3)


def test_pinned_score_share_is_strict():
    assert vm.pinned_score_share([9.0, -9.0, 9.01, -12, 0]) == pytest.approx(2 / 5)


def test_dev_report_runs():
    rng = np.random.default_rng(2)

    def mk(n, pairs=False):
        y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
        m = rng.normal(0, 3, n) + 2 * y
        d = {"label": y.tolist(), "margin": m.tolist(), "score": vm.probs_from_margin(m).tolist()}
        if pairs:
            d["pair_id"] = [f"p{i % (n // 2)}" for i in range(n)]
        return d

    sets = {"twins_prompt": mk(40, True), "dev_pair2": mk(50), "tc_val": mk(60),
            "aegis_val": mk(60), "dev_holdout_p3": mk(30), "dev_holdout_p4": mk(30),
            "dev_holdout_p5": {"label": [0] * 10, "margin": [0.0] * 10, "score": [0.5] * 10}}
    r = vm.dev_report(sets)
    assert r["dev_intent"] == pytest.approx((r["twins_auroc"] + r["dev_pair2_auroc"]) / 2)
    for k in ("t_gate", "t_screen", "t_bal", "saturation", "dev_holdout_p5_frr_at_gate"):
        assert k in r
