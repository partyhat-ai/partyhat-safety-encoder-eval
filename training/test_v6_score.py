"""Tests for v6_score.py: the train.py command for each recipe variant, and margin-storing scorecards."""
import numpy as np
import pytest

import v6_score as vs


def _cmd(variant, **kw):
    return vs.build_train_cmd("py", variant, 13, "/vol/v6", "/vol/v6/base/deberta-v3-large",
                              "/tmp/m", "/tmp/rm.json", 1234, **kw)


def test_r4_is_r1_without_p5():
    c, r1 = _cmd("R4"), _cmd("R1")
    files = lambda cmd: [e.rsplit("/", 1)[1] for e in cmd if e.endswith(".jsonl")]  # noqa: E731
    assert files(c) == ["p1_copy.jsonl", "p1_copy.jsonl", "p2.jsonl", "p3.jsonl", "p4.jsonl"]
    assert files(r1) == files(c) + ["p5.jsonl"]
    assert all("/v6/pools/" in e for e in c if e.endswith(".jsonl"))
    strip = lambda cmd: [x for x in cmd if not x.endswith(".jsonl")]  # noqa: E731
    assert strip(c) == strip(r1)          # identical recipe, flags and S
    # R4 row count = 3 ToxicChat copies + Aegis 2.0 + WildGuardMix + WildJailbreak (release file counts)
    assert 3 * 4573 + 21132 + 43080 + 58200 == 136131


def _val(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_r1_command_matches_the_recipe():
    c = _cmd("R1")
    assert _val(c, "--lr") == "2e-05" and _val(c, "--batch_size") == "32"
    assert _val(c, "--grad_accum") == "1" and _val(c, "--max_len") == "512"
    assert _val(c, "--rdrop_alpha") == "0" and _val(c, "--class_weights") == "1.0,1.0"
    assert _val(c, "--loss") == "weighted_ce" and _val(c, "--soft_label_field") == "toxicity_soft"
    assert _val(c, "--max_steps") == "1234" and _val(c, "--eval_steps") == "1234"
    assert _val(c, "--seed") == "13" and _val(c, "--head_seed") == "13"
    assert _val(c, "--split_seed") == "13" and _val(c, "--val_fraction") == "0.10"
    assert "--allow_tf32" in c and "--group_by_length" in c
    assert "--warm_start_model" not in c and "--push_to_hub" not in c
    assert "--bow_logit_field" not in c
    i = c.index("--extra_train_jsonl")
    extra = c[i + 1:i + 7]
    assert [e.rsplit("/", 1)[1] for e in extra] == ["p1_copy.jsonl", "p1_copy.jsonl", "p2.jsonl",
                                                     "p3.jsonl", "p4.jsonl", "p5.jsonl"]


def test_r0_drops_p5_and_benign_p4():
    c = _cmd("R0")
    i = c.index("--extra_train_jsonl")
    assert [e.rsplit("/", 1)[1] for e in c[i + 1:i + 6]] == [
        "p1_copy.jsonl", "p1_copy.jsonl", "p2.jsonl", "p3.jsonl", "p4_r0.jsonl"]
    assert "p5.jsonl" not in " ".join(c)


def test_r2_reads_bow_files_and_sets_flag():
    c = _cmd("R2")
    assert _val(c, "--bow_logit_field") == "bow_logit"
    assert all("/r2/" in e for e in c if e.endswith(".jsonl"))


def test_oom_fallback_batch():
    c = _cmd("R1", batch_size=16, grad_accum=2)
    assert _val(c, "--batch_size") == "16" and _val(c, "--grad_accum") == "2"


def test_scorecard_has_margin_and_no_text():
    rows = [{"row_id": "x:1", "user_input": "secret text", "toxicity": 1},
            {"row_id": "x:2", "user_input": "other", "toxicity": 0}]
    m = np.array([3.0, -40.0])
    from v6_metrics import probs_from_margin
    card = vs.scorecard("R1_s13", "set", rows, probs_from_margin(m), m)
    assert set(card["rows"][0]) == {"i", "src", "label", "score", "margin", "decision"}
    assert "secret" not in str(card) and card["rows"][1]["margin"] == -40.0
    assert card["rows"][1]["score"] > 0


# ---- the test lock ------------------------------------------------------------------------ #
def _seal(tmp_path, text="selection v1"):
    md = tmp_path / vs.SELECTION
    md.write_text(text)
    vs.seal_selection(tmp_path)
    return md


def _never(*a, **k):
    raise AssertionError("data or model touched while the test sets are locked")


@pytest.mark.parametrize("name", vs.TEST_SETS)
def test_every_test_set_is_locked_without_a_selection_seal(tmp_path, name):
    with pytest.raises(vs.TestSetLocked):
        vs.assert_test_unlocked(name, tmp_path)
    with pytest.raises(vs.TestSetLocked):
        vs.score_test_sets("R1_s13", [name], tmp_path, _never, _never)


def test_an_unsealed_record_keeps_the_lock(tmp_path):
    (tmp_path / vs.SELECTION).write_text("selection v1")
    with pytest.raises(vs.TestSetLocked):
        vs.assert_test_unlocked("xstest", tmp_path)


def test_lock_checks_all_sets_before_loading_any(tmp_path):
    with pytest.raises(vs.TestSetLocked):
        vs.score_test_sets("R1_s13", ["tc_val", "orbench"], tmp_path, _never, _never)


def test_unknown_set_names_are_refused_even_when_sealed(tmp_path):
    _seal(tmp_path)
    with pytest.raises(vs.TestSetLocked):
        vs.assert_test_unlocked("orbench_renamed", tmp_path)


def test_drifted_or_partial_seal_keeps_the_lock(tmp_path):
    md = _seal(tmp_path)
    vs.assert_test_unlocked("xstest", tmp_path)
    md.write_text("selection v1, edited after sealing")
    with pytest.raises(vs.TestSetLocked):
        vs.assert_test_unlocked("xstest", tmp_path)
    (tmp_path / vs.SELECTION_SEAL).unlink()
    with pytest.raises(vs.TestSetLocked):
        vs.assert_test_unlocked("xstest", tmp_path)


def test_a_seal_cannot_be_replaced_for_different_content(tmp_path):
    md = _seal(tmp_path)
    assert vs.seal_selection(tmp_path)["sha256"] == vs.record_digest(md)[0]   # resealing the same is a no-op
    md.write_text("selection v2")
    with pytest.raises(vs.TestSetLocked):
        vs.seal_selection(tmp_path)


def test_dev_sets_are_never_locked(tmp_path):
    for n in vs.DEV_SETS:
        vs.assert_test_unlocked(n, tmp_path)


def test_sealed_selection_unlocks_and_scores(tmp_path):
    _seal(tmp_path)
    rows = [{"row_id": "jbb:0", "toxicity": 1}, {"row_id": "jbb:1", "toxicity": 0}]
    m = np.array([2.0, -2.0])
    from v6_metrics import probs_from_margin
    out = vs.score_test_sets("R1_s13", ["jbb"], tmp_path, lambda n: rows,
                             lambda r: (probs_from_margin(m), m), threshold=0.5)
    assert [r["decision"] for r in out["jbb"]["rows"]] == [1, 0]
    assert out["jbb"]["rows"][0]["margin"] == 2.0


def test_twins_pair_id_parsing():
    card = {"rows": [{"src": "calm:abc:def:1", "score": 0.9, "margin": 2.0, "label": 1},
                     {"src": "calm:abc:def:0", "score": 0.1, "margin": -2.0, "label": 0}]}
    s = vs.dev_sets_for_report({"twins_prompt": card})
    assert s["twins_prompt"]["pair_id"] == ["abc:def", "abc:def"]
