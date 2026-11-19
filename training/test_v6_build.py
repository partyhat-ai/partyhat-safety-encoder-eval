"""Tests for v6_build.py: de-duplication, decontamination matching, caps and held-back rows. CPU only; synthetic text."""
import numpy as np
import pandas as pd

import v6_build as vb
from decontam import decontaminate

WORDS = [f"w{i}" for i in range(60)]


def _texts(n, seed):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=rng.integers(2, 15))) for _ in range(n)]


def test_lexical_index_matches_decontaminate_exact_and_jaccard():
    refs = _texts(150, 1)
    cands = _texts(300, 2)
    # plant exact (case/punct changes) and near-duplicates
    cands += [r.upper() + "!" for r in refs[:20]]
    cands += [r + " w1" for r in refs[20:40] if len(r.split()) >= 12]
    idx = vb.RefIndex({"a": refs[:75], "b": refs[75:]})
    mine = {i for i, t in enumerate(cands) if idx.lexical_one(t)}
    ref = set(decontaminate(cands, refs, embedding=False)["removed_idx"])
    assert mine == ref and len(ref) >= 20


def test_per_set_attribution_counts_every_matching_set():
    idx = vb.RefIndex({"a": ["alpha beta gamma delta"], "b": ["Alpha beta, gamma delta"]})
    assert idx.lexical_one("alpha beta gamma delta") == {0: "exact", 1: "exact"}


def test_dedup_within_drops_conflicts_and_duplicates():
    df = pd.DataFrame({"row_id": list("abcde"), "user_input": ["Hi there", "hi there!", "x y", "X Y", ""],
                       "toxicity": [0, 0, 0, 1, 0]})
    out, rep = vb.dedup_within(df)
    assert list(out["row_id"]) == ["a"]
    assert rep["conflict_texts"] == 1 and rep["dup_rows"] == 1 and rep["empty"] == 1


def test_dedup_across_p1_wins_and_p2_p5_conflicts_drop_all():
    mk = lambda ids, texts, ys: vb.dedup_within(pd.DataFrame(  # noqa: E731
        {"row_id": ids, "user_input": texts, "toxicity": ys}))[0]
    pools = {"P2": mk(["p2a", "p2b", "p2c"], ["in p1", "disputed", "shared"], [1, 0, 1]),
             "P3": mk(["p3a", "p3b"], ["disputed", "shared"], [1, 1])}
    out, rep = vb.dedup_across({"in p1": 0}, pools)
    assert list(out["P2"]["row_id"]) == ["p2c"] and list(out["P3"]["row_id"]) == []
    assert rep["P2"]["dup_or_conflict_with_P1"] == 1 and rep["P3"]["conflict_P2_P5"] == 1
    assert rep["P3"]["dup_of_earlier_pool"] == 1


def test_stratified_cap_and_holdout_are_proportional_and_deterministic():
    df = pd.DataFrame({"type": ["a"] * 700 + ["b"] * 300, "toxicity": [0, 1] * 500,
                       "row_id": range(1000), "user_input": [str(i) for i in range(1000)]})
    s1, _ = vb.stratified_cap(df, ["type", "toxicity"], 100, np.random.default_rng(13))
    s2, _ = vb.stratified_cap(df, ["type", "toxicity"], 100, np.random.default_rng(13))
    assert len(s1) == 100 and list(s1["row_id"]) == list(s2["row_id"])
    assert (s1["type"] == "a").sum() == 70
    tr, ho = vb.holdout(df, ["type", "toxicity"], np.random.default_rng(13))
    # 3% per stratum, Python round (half to even): 350 -> 10, 150 -> 4
    assert len(ho) == 2 * round(0.03 * 350) + 2 * round(0.03 * 150) == 28
    assert len(tr) + len(ho) == 1000 and not set(tr["row_id"]) & set(ho["row_id"])
