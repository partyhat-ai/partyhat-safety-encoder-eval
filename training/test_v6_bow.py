"""Tests for the bag-of-words correction (recipe variant R2) in train.py and its cross-fit script.

Run from the directory that holds train.py and v6_bow_crossfit.py:
    python3 -m pytest -q test_v6_bow.py
CPU only. Builds a tiny random DeBERTa; downloads nothing.
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for i in range(40):
        vocab[f"w{i}"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]")


def tiny_model(seed=0):
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
    torch.manual_seed(seed)
    cfg = DebertaV2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=2, intermediate_size=64,
                          max_position_embeddings=64, num_labels=2, pad_token_id=0)
    return DebertaV2ForSequenceClassification(cfg)


def tiny_batch(n=16, seed=0, soft=True, bow=None):
    from datasets import Dataset
    from transformers import DataCollatorWithPadding
    tok = tiny_tokenizer()
    rng = np.random.default_rng(seed)
    texts = [" ".join(f"w{j}" for j in rng.integers(0, 40, rng.integers(3, 12))) for _ in range(n)]
    labels = rng.integers(0, 2, n).tolist()
    d = {"user_input": texts, "labels": labels}
    if soft:
        d["soft_labels"] = [0.95 if y else 0.05 for y in labels]
    if bow is not None:
        d["bow_logit"] = [float(v) for v in bow]
    ds = Dataset.from_dict(d).map(lambda b: tok(b["user_input"], truncation=True, max_length=32),
                                  batched=True, remove_columns=["user_input"])
    return DataCollatorWithPadding(tok)([ds[i] for i in range(n)])


def bare_trainer(weights=(1.0, 1.0), rdrop=0.0):
    from train import WeightedLossTrainer
    tr = WeightedLossTrainer.__new__(WeightedLossTrainer)
    tr.loss_type, tr.focal_gamma, tr.rdrop_alpha = "weighted_ce", 2.0, rdrop
    tr.class_weights = torch.tensor(list(weights))
    tr.model_accepts_loss_kwargs = False
    return tr


def train_mode_no_dropout(model):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or "Dropout" in type(m).__name__:
            m.p = 0.0
            if hasattr(m, "drop_prob"):
                m.drop_prob = 0.0
    return model


def clone(batch, drop=()):
    return {k: v.clone() for k, v in batch.items() if k not in drop}


def test_zero_bow_equals_no_bow_soft_and_hard_paths():
    """With b = 0 everywhere the loss equals the base recipe's (R1) to 1e-6."""
    from train import WeightedLossTrainer
    model = train_mode_no_dropout(tiny_model(1))
    for soft in (True, False):
        batch = tiny_batch(soft=soft, bow=np.zeros(16))
        assert batch["bow_logit"].dtype.is_floating_point
        tr = bare_trainer()
        with torch.no_grad():
            with_b = WeightedLossTrainer.compute_loss(tr, model, clone(batch))
            without = WeightedLossTrainer.compute_loss(tr, model, clone(batch, drop=("bow_logit",)))
        assert torch.allclose(with_b, without, atol=1e-6, rtol=0), (soft, with_b.item(), without.item())


def test_bow_shift_matches_manual_loss_and_outputs_stay_raw():
    from train import WeightedLossTrainer
    model = train_mode_no_dropout(tiny_model(2))
    rng = np.random.default_rng(3)
    b = rng.normal(0, 2.5, 16)
    batch = tiny_batch(soft=True, bow=b)
    tr = bare_trainer(weights=(0.57, 4.07))
    with torch.no_grad():
        loss, out = WeightedLossTrainer.compute_loss(tr, model, clone(batch), return_outputs=True)
        feats = clone(batch, drop=("labels", "soft_labels", "bow_logit"))
        raw = model(**feats).logits
        bt = batch["bow_logit"].to(raw.dtype)
        manual = tr._soft_loss(raw + torch.stack([-0.5 * bt, 0.5 * bt], dim=1),
                               batch["soft_labels"], tr.class_weights)
    assert torch.allclose(loss, manual, atol=1e-6, rtol=0), (loss.item(), manual.item())
    assert torch.allclose(out.logits, raw, atol=1e-6), "outputs must carry the raw logits"
    with torch.no_grad():
        plain = WeightedLossTrainer.compute_loss(tr, model, clone(batch, drop=("bow_logit",)))
    assert abs(loss.item() - plain.item()) > 1e-4, "a non-zero b must change the training loss"


def test_bow_is_ignored_in_eval_mode():
    from train import WeightedLossTrainer
    model = tiny_model(4).eval()
    batch = tiny_batch(soft=False, bow=np.full(16, 5.0))
    tr = bare_trainer()
    with torch.no_grad():
        with_b = WeightedLossTrainer.compute_loss(tr, model, clone(batch))
        without = WeightedLossTrainer.compute_loss(tr, model, clone(batch, drop=("bow_logit",)))
    assert torch.allclose(with_b, without, atol=1e-7, rtol=0)


def test_gradient_shrinks_where_the_bag_of_words_already_explains_the_label():
    """The point of the correction: rows the keyword expert gets right push the encoder less."""
    from train import WeightedLossTrainer
    grads = {}
    for name, sign in (("agrees", 1.0), ("zero", 0.0), ("disagrees", -1.0)):
        model = train_mode_no_dropout(tiny_model(5))
        batch = tiny_batch(soft=False, bow=np.zeros(16), seed=7)
        y = batch["labels"].float()
        batch["bow_logit"] = sign * 4.0 * (2 * y - 1)  # +4 toward the true class, or away from it
        tr = bare_trainer()
        model.zero_grad()
        WeightedLossTrainer.compute_loss(tr, model, clone(batch)).backward()
        grads[name] = float(model.classifier.weight.grad.norm())
    assert grads["agrees"] < grads["zero"] < grads["disagrees"], grads


def test_rdrop_path_still_runs_with_bow():
    from train import WeightedLossTrainer
    model = train_mode_no_dropout(tiny_model(6))
    batch = tiny_batch(soft=True, bow=np.linspace(-3, 3, 16))
    tr = bare_trainer(rdrop=0.5)
    with torch.no_grad():
        loss = WeightedLossTrainer.compute_loss(tr, model, clone(batch))
    assert torch.isfinite(loss)


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_crossfit_script(tmp_path):
    import v6_bow_crossfit as cf
    rng = np.random.default_rng(0)
    scary, bland = [f"s{i}" for i in range(30)], [f"b{i}" for i in range(30)]

    def row(y):
        # unsafe rows lean on the "scary" vocabulary, so a unigram model can predict the label
        pool = scary if (rng.random() < (0.85 if y else 0.15)) else bland
        return {"user_input": " ".join(rng.choice(pool, 8)), "toxicity": int(y), "keep": "x"}

    a = [row(i % 2) for i in range(400)]
    a[1] = dict(a[0])  # an exact duplicate, to check that duplicates share a fold
    b = [row(0) for _ in range(120)]  # a single-class pool, like FalseReject
    pa, pb = tmp_path / "pa.jsonl", tmp_path / "pb.jsonl"
    _write(pa, a)
    _write(pb, b)
    out = tmp_path / "out"
    rep = cf.main(["--inputs", str(pa), str(pb), str(pa), "--out_dir", str(out), "--clip", "6"])
    ra = [json.loads(l) for l in open(out / "pa.jsonl")]
    rb = [json.loads(l) for l in open(out / "pb.jsonl")]
    assert len(ra) == len(a) and len(rb) == len(b), "a path given twice is processed once"
    assert all("bow_logit" in r and r["keep"] == "x" for r in ra + rb)
    assert [r["user_input"] for r in ra] == [r["user_input"] for r in a], "row order is kept"
    vals = np.array([r["bow_logit"] for r in ra + rb])
    assert np.isfinite(vals).all() and np.abs(vals).max() <= 6.0
    assert ra[0]["bow_logit"] == ra[1]["bow_logit"], "duplicates must share a fold and so a score"
    assert rep["auroc_overall"] > 0.8 and rep["per_file"]["pb.jsonl"]["auroc"] is None
    ya = np.array([r["toxicity"] for r in ra])
    va = np.array([r["bow_logit"] for r in ra])
    assert va[ya == 1].mean() > va[ya == 0].mean()


def test_main_end_to_end_with_the_v6_flags(tmp_path, monkeypatch):
    """The release training command line (plus the R2 field) runs end to end on a tiny model:
    loader rows + a file listed twice + a file with soft targets and bow_logit, grouped by length,
    one evaluation at the last step, final checkpoint kept."""
    import types

    import pandas as pd

    import train
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    rng = np.random.default_rng(0)

    def text():
        return " ".join(f"w{j}" for j in rng.integers(0, 40, rng.integers(3, 12)))

    def frame(n):
        y = np.arange(n) % 2
        return pd.DataFrame({"user_input": [text() for _ in range(n)], "toxicity": y})

    monkeypatch.setattr(train, "load_train_val",
                        lambda val_fraction=0.1, seed=13, max_samples=None: (frame(24), frame(16)))
    monkeypatch.setattr(train, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tokenizer()))
    monkeypatch.setattr(train, "AutoModelForSequenceClassification",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_model(0)))
    plain = [{"user_input": text(), "toxicity": int(i % 2)} for i in range(10)]
    rich = [{"user_input": text(), "toxicity": int(i % 2),
             "toxicity_soft": 0.95 if i % 2 else 0.05, "bow_logit": float(rng.normal(0, 2))}
            for i in range(20)]
    p_plain, p_rich = tmp_path / "p1_copy.jsonl", tmp_path / "p3.jsonl"
    _write(p_plain, plain)
    _write(p_rich, rich)
    out, meta_path = tmp_path / "out", tmp_path / "meta.json"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--model", "tiny", "--output_dir", str(out),
        "--extra_train_jsonl", str(p_plain), str(p_plain), str(p_rich),
        "--soft_label_field", "toxicity_soft", "--bow_logit_field", "bow_logit",
        "--loss", "weighted_ce", "--class_weights", "1.0,1.0",
        "--lr", "1e-3", "--batch_size", "4", "--grad_accum", "1", "--max_len", "32",
        "--rdrop_alpha", "0", "--allow_tf32", "--group_by_length",
        "--max_steps", "6", "--eval_steps", "6",
        "--seed", "13", "--head_seed", "13", "--split_seed", "13", "--run_meta", str(meta_path)])
    train.main()
    meta = json.load(open(meta_path))
    assert meta["global_step"] == 6 and meta["best_checkpoint_step"] == 6
    assert meta["train_rows"] == 24 + 2 * len(plain) + len(rich)
    assert meta["soft_targets"] and meta["bow_residual"] and meta["group_by_length"]
    assert all(np.isfinite(h["loss"]) for h in meta["log_history"] if "loss" in h)
    assert (out / "model.safetensors").exists() or (out / "pytorch_model.bin").exists()


def test_missing_soft_target_is_refused(tmp_path, monkeypatch):
    """A row without its soft target must stop the run instead of becoming a NaN loss."""
    import types

    import pandas as pd
    import pytest

    import train
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    df = pd.DataFrame({"user_input": ["w1 w2 w3"] * 8, "toxicity": [0, 1] * 4})
    monkeypatch.setattr(train, "load_train_val",
                        lambda val_fraction=0.1, seed=13, max_samples=None: (df, df))
    monkeypatch.setattr(train, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tokenizer()))
    bad = tmp_path / "bad.jsonl"
    _write(bad, [{"user_input": "w1 w2", "toxicity": 1, "toxicity_soft": 0.95},
                 {"user_input": "w3 w4", "toxicity": 0}])
    monkeypatch.setattr(sys, "argv", ["train.py", "--output_dir", str(tmp_path / "o"),
                                      "--extra_train_jsonl", str(bad),
                                      "--soft_label_field", "toxicity_soft"])
    with pytest.raises(SystemExit):
        train.main()


def test_private_flag_defaults_on_and_can_be_turned_off(monkeypatch):
    import train
    for argv, want in (([], True), (["--private"], True), (["--no-private"], False)):
        monkeypatch.setattr(sys, "argv", ["train.py", *argv])
        assert train.parse_args().private is want
