"""Tests for v6_select.select (the selection rule and the ToxicChat check) on synthetic candidate summaries."""
import v6_select as s


def c(intent, rng, auprc):
    return {"dev_intent": {"mean": intent, "range": rng}, "tc_val_auprc": {"mean": auprc, "range": 0.0}}


def test_r1_wins_when_lead_is_within_range():
    sel, _ = s.select({"R1": c(0.80, 0.03, 0.85), "R0": c(0.82, 0.01, 0.85)})
    assert sel == "R1"


def test_clear_single_model_winner():
    sel, _ = s.select({"R1": c(0.80, 0.01, 0.85), "R2": c(0.85, 0.02, 0.85)})
    assert sel == "R2"


def test_two_model_average_needs_to_beat_range():
    base = {"R1": c(0.80, 0.01, 0.85)}
    assert s.select(base | {"E11": c(0.81, 0.02, 0.85)})[0] == "R1"
    assert s.select(base | {"E11": c(0.84, 0.02, 0.85)})[0] == "E11"


def test_guardrail_falls_back_in_order_r1_last():
    cands = {"R1": c(0.80, 0.01, 0.86), "R2": c(0.86, 0.01, 0.83), "E12": c(0.90, 0.01, 0.80),
             "R0": c(0.82, 0.005, 0.855)}
    sel, trace = s.select(cands)
    # E12 replaces R2, fails the ToxicChat check; R2 fails it too; R0 passes
    assert sel == "R0" and any("E12 fails" in t for t in trace)
    cands["R0"] = c(0.82, 0.005, 0.70)
    assert s.select(cands)[0] == "R1"


def test_after_a_failed_single_model_the_next_must_still_beat_r1_by_its_range():
    # R2 fails the ToxicChat check; R0 leads R1 by 0.02 < R1's range 0.03, so the choice is R1
    cands = {"R1": c(0.80, 0.03, 0.86), "R2": c(0.90, 0.01, 0.80), "R0": c(0.82, 0.005, 0.86)}
    sel, trace = s.select(cands)
    assert sel == "R1" and any("R2 fails" in t for t in trace)


def test_after_a_failed_two_model_average_the_other_averages_are_reconsidered():
    cands = {"R1": c(0.80, 0.01, 0.86), "E12": c(0.90, 0.01, 0.80), "E11": c(0.86, 0.01, 0.86)}
    sel, trace = s.select(cands)
    assert sel == "E11" and any("E12 fails" in t for t in trace)
    cands["E11"] = c(0.805, 0.01, 0.86)    # no longer leads R1 by more than the range
    assert s.select(cands)[0] == "R1"
