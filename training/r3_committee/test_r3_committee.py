#!/usr/bin/env python3
"""Unit tests for the pure functions of r3_committee_modal.py. Synthetic server output only."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_committee_modal as rc  # noqa: E402


def tok(t, lp, top):
    return {"token": t, "logprob": lp, "top_logprobs": [{"token": a, "logprob": b} for a, b in top]}


def test_qwen_loose_margin():
    cfg = rc.MEMBERS["qwen3guard8b"]
    top = [(" Unsafe", math.log(0.2)), (" Safe", math.log(0.5)), (" Controversial", math.log(0.25)),
           (" The", math.log(0.05))]
    content = [tok("Safety", -0.01, [("Safety", -0.01)]), tok(":", -0.01, [(":", -0.01)]),
               tok(" Safe", math.log(0.5), top), tok("\n", -0.1, [("\n", -0.1)])]
    r = rc.parse_choice(content, cfg)
    assert r["parse"] == "ok" and r["gen"] == "safe" and not r["censored"]
    assert abs(r["margin"] - (math.log(0.2) - math.log(0.75))) < 1e-9   # Unsafe vs Safe-or-Controversial


def test_qwen_controversial_is_safe_side_and_unsafe_not_read_as_safe():
    cfg = rc.MEMBERS["qwen3guard8b"]
    top = [(" Controversial", math.log(0.6)), (" Unsafe", math.log(0.3)), (" Safe", math.log(0.1))]
    r = rc.parse_choice([tok("Safety", 0, []), tok(":", 0, []), tok(" Controversial", math.log(0.6), top)], cfg)
    assert r["gen"] == "controversial" and r["margin"] < 0
    top = [(" Unsafe", math.log(0.9)), (" Safe", math.log(0.1))]
    r = rc.parse_choice([tok("Safety", 0, []), tok(":", 0, []), tok(" Unsafe", math.log(0.9), top)], cfg)
    assert r["gen"] == "unsafe" and abs(r["margin"] - math.log(9)) < 1e-9


def test_granite_score_tag_and_censoring():
    cfg = rc.MEMBERS["granite41_8b"]
    top = [(" yes", -0.001), ("<", -9.0), (" maybe", -12.0)]
    content = [tok("<think>", 0, []), tok("\n", 0, []), tok("</think>", 0, []), tok("\n", 0, []),
               tok("<score>", 0, []), tok(" yes", -0.001, top), tok(" </score>", 0, [])]
    r = rc.parse_choice(content, cfg)
    assert r["parse"] == "ok" and r["gen"] == "yes" and r["censored"]
    assert abs(r["margin"] - (-0.001 - -12.0)) < 1e-9   # the missing class is bounded by the last logprob


def test_shieldstral_first_token_and_sign():
    cfg = rc.MEMBERS["shieldstral3b"]
    r = rc.parse_choice([tok("no", math.log(0.8), [("no", math.log(0.8)), ("yes", math.log(0.2))])], cfg)
    assert r["gen"] == "no" and abs(r["margin"] - math.log(0.25)) < 1e-9
    r = rc.parse_choice([tok("Sorry", -0.1, [("Sorry", -0.1), ("yes", -3.0)])], cfg)
    assert r["parse"] == "fail" and r["margin"] is None


def test_templates():
    q = rc.build_messages("qwen3guard8b", "hello")
    assert q == [{"role": "user", "content": "hello"}]
    g = rc.build_messages("granite41_8b", "hello")
    assert [m["role"] for m in g] == ["user", "user"] and g[0]["content"] == "hello"
    assert g[1]["content"].startswith("<guardian><no-think>As a judge agent")
    assert "<think>\\n</think>" in g[1]["content"]          # a literal backslash-n, as in the cookbook
    assert g[1]["content"].endswith("### Scoring Schema: If the last user's text meets the criteria, "
                                    "return 'yes'; otherwise, return 'no'.")
    s = rc.build_messages("shieldstral3b", "hello")
    assert s[0]["role"] == "system" and s[1]["content"].endswith("<Document>: [User] hello")
    assert rc.build_messages("qwen3guard8b", "")[0]["content"] == " "


def test_tier_plan():
    rows = ([{"row_id": f"h{i}", "pool": "HARMAUG", "toxicity": None} for i in range(5)]
            + [{"row_id": f"w{i}", "pool": "WILDCHAT", "toxicity": None} for i in range(5)]
            + [{"row_id": "a", "pool": "P1", "toxicity": 1}, {"row_id": "b", "pool": "P2", "toxicity": 0},
               {"row_id": "c", "pool": "P3", "toxicity": 0}, {"row_id": "d", "pool": "P4", "toxicity": 1},
               {"row_id": "e", "pool": "P5", "toxicity": 0}, {"row_id": "f", "pool": "WJB_EXTRA", "toxicity": 1}])
    tiers, counts = rc.tier_plan(rows)
    assert sorted(tiers[1]) == sorted([f"h{i}" for i in range(5)] + [f"w{i}" for i in range(5)])
    assert sorted(tiers[2]) == ["d", "e", "f"] and tiers[3] == ["c"]          # P1 and P2 are never scored
    assert rc.tier_plan(rows)[0] == tiers                                    # deterministic
    for bad in ({"row_id": "x", "pool": "P4", "toxicity": None}, {"row_id": "x", "pool": "WILDCHAT", "toxicity": 0},
                {"row_id": "x", "pool": "P9", "toxicity": 0}):
        try:
            rc.tier_plan(rows + [bad])
        except SystemExit:
            continue
        raise AssertionError("expected a stop")


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
