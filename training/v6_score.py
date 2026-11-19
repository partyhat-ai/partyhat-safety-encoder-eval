"""v6_score.py: the recipe variants, the train.py command for each run, and the dev scorer.

Plain Python (no Modal), unit-tested in test_v6_score.py.
"""
import datetime
import hashlib
import json
import os
import pathlib

import numpy as np

import v6_metrics as vm

BASE_MODEL = "microsoft/deberta-v3-large"
BASE_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
SEEDS = (13, 42, 99)
# Training files per recipe variant: (folder, files, uses the bag-of-words correction).
# The ToxicChat file (p1_copy) is listed twice; train.py's own ToxicChat loader adds the third copy.
# The keys R0-R4 are part of run folder names (for example R3_s13), so they stay as they are.
RECIPE_VARIANTS = {
    "R1": ("pools", ["p1_copy", "p1_copy", "p2", "p3", "p4", "p5"], False),
    "R0": ("pools", ["p1_copy", "p1_copy", "p2", "p3", "p4_r0"], False),
    "R2": ("r2", ["p1_copy", "p1_copy", "p2", "p3", "p4", "p5"], True),
    "R3": ("r3/pools", ["p1_copy", "p1_copy", "p2", "p3", "p4", "p5", "r3_extra"], False),
    # R4 was added after the first training runs: the base recipe without FalseReject (P5).
    "R4": ("pools", ["p1_copy", "p1_copy", "p2", "p3", "p4"], False),
}
# Readable names for the recipe-variant keys, used where results are printed.
RECIPE_NAMES = {
    "R1": "base recipe",
    "R0": "base recipe without the alarming-but-benign data",
    "R2": "base recipe + bag-of-words correction",
    "R3": "base recipe + teacher labels (the released recipe)",
    "R4": "base recipe without FalseReject",
}
RECIPE = dict(lr="2e-05", max_len="512", loss="weighted_ce", class_weights="1.0,1.0",
              rdrop_alpha="0", val_fraction="0.10", split_seed="13",
              soft_label_field="toxicity_soft")
DEV_SETS = ("twins_prompt", "dev_pair2", "tc_val", "aegis_val",
            "dev_holdout_p3", "dev_holdout_p4", "dev_holdout_p5")


def run_name(variant, seed):
    return f"{variant}_s{seed}"


def build_train_cmd(py, variant, seed, v6_root, base_path, out_dir, run_meta, steps,
                    batch_size=32, grad_accum=1):
    """The exact train.py command line for one training run of a recipe variant."""
    sub, files, bow = RECIPE_VARIANTS[variant]
    r = RECIPE
    extra = [os.path.join(v6_root, sub, f"{f}.jsonl") for f in files]
    cmd = [py, "train.py", "--output_dir", out_dir, "--model", base_path,
           "--extra_train_jsonl", *extra,
           "--soft_label_field", r["soft_label_field"], "--loss", r["loss"],
           "--class_weights", r["class_weights"], "--lr", r["lr"],
           "--batch_size", str(batch_size), "--grad_accum", str(grad_accum),
           "--max_len", r["max_len"], "--rdrop_alpha", r["rdrop_alpha"], "--allow_tf32",
           "--group_by_length", "--max_steps", str(int(steps)), "--eval_steps", str(int(steps)),
           "--seed", str(seed), "--head_seed", str(seed), "--split_seed", r["split_seed"],
           "--val_fraction", r["val_fraction"], "--run_meta", run_meta]
    if bow:
        cmd += ["--bow_logit_field", "bow_logit"]
    assert "--push_to_hub" not in cmd and "--warm_start_model" not in cmd
    return cmd


def score_texts(model, tok, texts, device, max_len=512, batch_size=64):
    """(P(unsafe), margin) per item in the original order. margin = logit_unsafe - logit_safe
    from the FP32 logits; P = softmax(logits)[:, 1] computed as sigmoid(margin) in float64."""
    import torch
    order = np.argsort([len(t) for t in texts], kind="stable")
    margin = np.empty(len(texts), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            lg = model(**enc).logits.float()
            margin[idx] = (lg[:, 1] - lg[:, 0]).double().cpu().numpy()
    return vm.probs_from_margin(margin), margin


def scorecard(run, set_name, rows, scores, margins, threshold=0.5, meta=None):
    """One scorecard: per item {i, src, label, score, margin, decision}; no text."""
    return {"run": run, "set": set_name, "n": len(rows), "decision_threshold": threshold,
            "rows": [{"i": i, "src": r["row_id"], "label": int(r["toxicity"]), "score": float(s),
                      "margin": float(m), "decision": int(s >= threshold)}
                     for i, (r, s, m) in enumerate(zip(rows, scores, margins))],
            "meta": meta or {}}


def load_dev(v6_root, names=DEV_SETS):
    out = {}
    for n in names:
        p = pathlib.Path(v6_root) / "dev" / f"{n}.jsonl"
        with open(p, encoding="utf-8") as f:
            out[n] = [json.loads(x) for x in f if x.strip()]
    return out


def dev_sets_for_report(cards):
    """scorecards {name: card} -> the dict v6_metrics.dev_report expects."""
    sets = {}
    for n, c in cards.items():
        d = {"score": [r["score"] for r in c["rows"]], "margin": [r["margin"] for r in c["rows"]],
             "label": [r["label"] for r in c["rows"]]}
        if n == "twins_prompt":
            # src = "calm:<pair_id>:<label>"
            d["pair_id"] = [r["src"][len("calm:"):].rsplit(":", 1)[0] for r in c["rows"]]
        sets[n] = d
    return sets


# ------------------------------------------------------------------------------------------- #
# the test lock: test sets stay closed until the selection record has been sealed            #
# ------------------------------------------------------------------------------------------- #
TEST_SETS = ("xstest", "jbb", "orbench", "openai_mod", "harmbench_std", "sst", "toxicchat_test",
             "toxicchat_dedup", "toxicchat_human", "aegis2_test_prompt", "wildguardtest_prompt")
SELECTION = "V6_SELECTION.md"
SELECTION_SEAL = "V6_SELECTION.sha256"   # {"sha256", "utc_timestamp", "size"} of the sealed record


class TestSetLocked(PermissionError):
    pass


def record_digest(path):
    """sha256 and size of a file with line endings normalised to LF and exactly one trailing newline."""
    raw = pathlib.Path(path).read_bytes()
    canon = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n") + b"\n"
    return hashlib.sha256(canon).hexdigest(), len(canon)


def seal_selection(root):
    """Seal V6_SELECTION.md by writing its sha256 and size to V6_SELECTION.sha256. Refuses to replace a
    seal of different content, so a sealed selection cannot be rewritten without it showing."""
    md, seal = pathlib.Path(root) / SELECTION, pathlib.Path(root) / SELECTION_SEAL
    digest, size = record_digest(md)
    if seal.exists():
        old = json.loads(seal.read_text())
        if old.get("sha256") != digest:
            raise TestSetLocked(f"{SELECTION_SEAL} seals different content; refusing to reseal")
        return old
    rec = {"sha256": digest,
           "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "size": size}
    seal.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def selection_sealed(root):
    """True once V6_SELECTION.md has been sealed and V6_SELECTION.sha256 still matches it."""
    md, seal = pathlib.Path(root) / SELECTION, pathlib.Path(root) / SELECTION_SEAL
    if not (md.is_file() and seal.is_file()):
        return False
    try:
        rec = json.loads(seal.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (rec.get("sha256"), rec.get("size")) == record_digest(md)


def assert_test_unlocked(set_name, root):
    """Refuse to score a v6 checkpoint on any test set until the selection record is sealed, so the
    recipe and thresholds are chosen on dev sets before any test result is seen. Unknown set names
    are refused too, so a renamed test set cannot slip past the lock."""
    if set_name in DEV_SETS:
        return
    if set_name not in TEST_SETS:
        raise TestSetLocked(f"{set_name}: not a known v6 dev or test set; refusing")
    if not selection_sealed(root):
        raise TestSetLocked(f"{set_name} is a test set: locked until {SELECTION} is sealed and "
                            f"{SELECTION_SEAL} matches it")


def score_test_sets(run, set_names, root, load_set, score, threshold=0.5):
    """The only v6 test-scoring path. Every set is checked against the lock before ANY set is
    loaded or any model is touched. load_set(name) -> rows; score(rows) -> (p, margin)."""
    for n in set_names:
        assert_test_unlocked(n, root)
    out = {}
    for n in set_names:
        rows = load_set(n)
        p, m = score(rows)
        out[n] = scorecard(run, n, rows, p, m, threshold)
    return out


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
