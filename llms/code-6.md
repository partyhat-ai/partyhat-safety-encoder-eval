# partyhat-safety-encoder-eval: source and docs, part 6 of 7

Generated from commit `6e4922573e47` (2026-11-20) of github.com/partyhat-ai/partyhat-safety-encoder-eval by `scripts/llms_digest.py`, which `.github/workflows/llms-digest.yml` reruns on every push to `main`, so this text matches the code at that commit.
Files `training/test_v6_bow.py` to `training/v6_build.py`. Lines are numbered as in the repository.

## `training/test_v6_bow.py`

```python
    1| """Tests for the bag-of-words correction (recipe variant R2) in train.py and its cross-fit script.
    2| 
    3| Run from the directory that holds train.py and v6_bow_crossfit.py:
    4|     python3 -m pytest -q test_v6_bow.py
    5| CPU only. Builds a tiny random DeBERTa; downloads nothing.
    6| """
    7| import json
    8| import pathlib
    9| import sys
   10| 
   11| import numpy as np
   12| import torch
   13| import torch.nn.functional as F
   14| 
   15| ROOT = pathlib.Path(__file__).parent
   16| sys.path.insert(0, str(ROOT))
   17| 
   18| 
   19| def tiny_tokenizer():
   20|     from tokenizers import Tokenizer, models, pre_tokenizers
   21|     from transformers import PreTrainedTokenizerFast
   22|     vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
   23|     for i in range(40):
   24|         vocab[f"w{i}"] = len(vocab)
   25|     tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
   26|     tk.pre_tokenizer = pre_tokenizers.Whitespace()
   27|     return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
   28|                                    cls_token="[CLS]", sep_token="[SEP]")
   29| 
   30| 
   31| def tiny_model(seed=0):
   32|     from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
   33|     torch.manual_seed(seed)
   34|     cfg = DebertaV2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2,
   35|                           num_attention_heads=2, intermediate_size=64,
   36|                           max_position_embeddings=64, num_labels=2, pad_token_id=0)
   37|     return DebertaV2ForSequenceClassification(cfg)
   38| 
   39| 
   40| def tiny_batch(n=16, seed=0, soft=True, bow=None):
   41|     from datasets import Dataset
   42|     from transformers import DataCollatorWithPadding
   43|     tok = tiny_tokenizer()
   44|     rng = np.random.default_rng(seed)
   45|     texts = [" ".join(f"w{j}" for j in rng.integers(0, 40, rng.integers(3, 12))) for _ in range(n)]
   46|     labels = rng.integers(0, 2, n).tolist()
   47|     d = {"user_input": texts, "labels": labels}
   48|     if soft:
   49|         d["soft_labels"] = [0.95 if y else 0.05 for y in labels]
   50|     if bow is not None:
   51|         d["bow_logit"] = [float(v) for v in bow]
   52|     ds = Dataset.from_dict(d).map(lambda b: tok(b["user_input"], truncation=True, max_length=32),
   53|                                   batched=True, remove_columns=["user_input"])
   54|     return DataCollatorWithPadding(tok)([ds[i] for i in range(n)])
   55| 
   56| 
   57| def bare_trainer(weights=(1.0, 1.0), rdrop=0.0):
   58|     from train import WeightedLossTrainer
   59|     tr = WeightedLossTrainer.__new__(WeightedLossTrainer)
   60|     tr.loss_type, tr.focal_gamma, tr.rdrop_alpha = "weighted_ce", 2.0, rdrop
   61|     tr.class_weights = torch.tensor(list(weights))
   62|     tr.model_accepts_loss_kwargs = False
   63|     return tr
   64| 
   65| 
   66| def train_mode_no_dropout(model):
   67|     model.train()
   68|     for m in model.modules():
   69|         if isinstance(m, torch.nn.Dropout) or "Dropout" in type(m).__name__:
   70|             m.p = 0.0
   71|             if hasattr(m, "drop_prob"):
   72|                 m.drop_prob = 0.0
   73|     return model
   74| 
   75| 
   76| def clone(batch, drop=()):
   77|     return {k: v.clone() for k, v in batch.items() if k not in drop}
   78| 
   79| 
   80| def test_zero_bow_equals_no_bow_soft_and_hard_paths():
   81|     """With b = 0 everywhere the loss equals the base recipe's (R1) to 1e-6."""
   82|     from train import WeightedLossTrainer
   83|     model = train_mode_no_dropout(tiny_model(1))
   84|     for soft in (True, False):
   85|         batch = tiny_batch(soft=soft, bow=np.zeros(16))
   86|         assert batch["bow_logit"].dtype.is_floating_point
   87|         tr = bare_trainer()
   88|         with torch.no_grad():
   89|             with_b = WeightedLossTrainer.compute_loss(tr, model, clone(batch))
   90|             without = WeightedLossTrainer.compute_loss(tr, model, clone(batch, drop=("bow_logit",)))
   91|         assert torch.allclose(with_b, without, atol=1e-6, rtol=0), (soft, with_b.item(), without.item())
   92| 
   93| 
   94| def test_bow_shift_matches_manual_loss_and_outputs_stay_raw():
   95|     from train import WeightedLossTrainer
   96|     model = train_mode_no_dropout(tiny_model(2))
   97|     rng = np.random.default_rng(3)
   98|     b = rng.normal(0, 2.5, 16)
   99|     batch = tiny_batch(soft=True, bow=b)
  100|     tr = bare_trainer(weights=(0.57, 4.07))
  101|     with torch.no_grad():
  102|         loss, out = WeightedLossTrainer.compute_loss(tr, model, clone(batch), return_outputs=True)
  103|         feats = clone(batch, drop=("labels", "soft_labels", "bow_logit"))
  104|         raw = model(**feats).logits
  105|         bt = batch["bow_logit"].to(raw.dtype)
  106|         manual = tr._soft_loss(raw + torch.stack([-0.5 * bt, 0.5 * bt], dim=1),
  107|                                batch["soft_labels"], tr.class_weights)
  108|     assert torch.allclose(loss, manual, atol=1e-6, rtol=0), (loss.item(), manual.item())
  109|     assert torch.allclose(out.logits, raw, atol=1e-6), "outputs must carry the raw logits"
  110|     with torch.no_grad():
  111|         plain = WeightedLossTrainer.compute_loss(tr, model, clone(batch, drop=("bow_logit",)))
  112|     assert abs(loss.item() - plain.item()) > 1e-4, "a non-zero b must change the training loss"
  113| 
  114| 
  115| def test_bow_is_ignored_in_eval_mode():
  116|     from train import WeightedLossTrainer
  117|     model = tiny_model(4).eval()
  118|     batch = tiny_batch(soft=False, bow=np.full(16, 5.0))
  119|     tr = bare_trainer()
  120|     with torch.no_grad():
  121|         with_b = WeightedLossTrainer.compute_loss(tr, model, clone(batch))
  122|         without = WeightedLossTrainer.compute_loss(tr, model, clone(batch, drop=("bow_logit",)))
  123|     assert torch.allclose(with_b, without, atol=1e-7, rtol=0)
  124| 
  125| 
  126| def test_gradient_shrinks_where_the_bag_of_words_already_explains_the_label():
  127|     """The point of the correction: rows the keyword expert gets right push the encoder less."""
  128|     from train import WeightedLossTrainer
  129|     grads = {}
  130|     for name, sign in (("agrees", 1.0), ("zero", 0.0), ("disagrees", -1.0)):
  131|         model = train_mode_no_dropout(tiny_model(5))
  132|         batch = tiny_batch(soft=False, bow=np.zeros(16), seed=7)
  133|         y = batch["labels"].float()
  134|         batch["bow_logit"] = sign * 4.0 * (2 * y - 1)  # +4 toward the true class, or away from it
  135|         tr = bare_trainer()
  136|         model.zero_grad()
  137|         WeightedLossTrainer.compute_loss(tr, model, clone(batch)).backward()
  138|         grads[name] = float(model.classifier.weight.grad.norm())
  139|     assert grads["agrees"] < grads["zero"] < grads["disagrees"], grads
  140| 
  141| 
  142| def test_rdrop_path_still_runs_with_bow():
  143|     from train import WeightedLossTrainer
  144|     model = train_mode_no_dropout(tiny_model(6))
  145|     batch = tiny_batch(soft=True, bow=np.linspace(-3, 3, 16))
  146|     tr = bare_trainer(rdrop=0.5)
  147|     with torch.no_grad():
  148|         loss = WeightedLossTrainer.compute_loss(tr, model, clone(batch))
  149|     assert torch.isfinite(loss)
  150| 
  151| 
  152| def _write(path, rows):
  153|     with open(path, "w") as f:
  154|         for r in rows:
  155|             f.write(json.dumps(r) + "\n")
  156| 
  157| 
  158| def test_crossfit_script(tmp_path):
  159|     import v6_bow_crossfit as cf
  160|     rng = np.random.default_rng(0)
  161|     scary, bland = [f"s{i}" for i in range(30)], [f"b{i}" for i in range(30)]
  162| 
  163|     def row(y):
  164|         # unsafe rows lean on the "scary" vocabulary, so a unigram model can predict the label
  165|         pool = scary if (rng.random() < (0.85 if y else 0.15)) else bland
  166|         return {"user_input": " ".join(rng.choice(pool, 8)), "toxicity": int(y), "keep": "x"}
  167| 
  168|     a = [row(i % 2) for i in range(400)]
  169|     a[1] = dict(a[0])  # an exact duplicate, to check that duplicates share a fold
  170|     b = [row(0) for _ in range(120)]  # a single-class pool, like FalseReject
  171|     pa, pb = tmp_path / "pa.jsonl", tmp_path / "pb.jsonl"
  172|     _write(pa, a)
  173|     _write(pb, b)
  174|     out = tmp_path / "out"
  175|     rep = cf.main(["--inputs", str(pa), str(pb), str(pa), "--out_dir", str(out), "--clip", "6"])
  176|     ra = [json.loads(l) for l in open(out / "pa.jsonl")]
  177|     rb = [json.loads(l) for l in open(out / "pb.jsonl")]
  178|     assert len(ra) == len(a) and len(rb) == len(b), "a path given twice is processed once"
  179|     assert all("bow_logit" in r and r["keep"] == "x" for r in ra + rb)
  180|     assert [r["user_input"] for r in ra] == [r["user_input"] for r in a], "row order is kept"
  181|     vals = np.array([r["bow_logit"] for r in ra + rb])
  182|     assert np.isfinite(vals).all() and np.abs(vals).max() <= 6.0
  183|     assert ra[0]["bow_logit"] == ra[1]["bow_logit"], "duplicates must share a fold and so a score"
  184|     assert rep["auroc_overall"] > 0.8 and rep["per_file"]["pb.jsonl"]["auroc"] is None
  185|     ya = np.array([r["toxicity"] for r in ra])
  186|     va = np.array([r["bow_logit"] for r in ra])
  187|     assert va[ya == 1].mean() > va[ya == 0].mean()
  188| 
  189| 
  190| def test_main_end_to_end_with_the_v6_flags(tmp_path, monkeypatch):
  191|     """The release training command line (plus the R2 field) runs end to end on a tiny model:
  192|     loader rows + a file listed twice + a file with soft targets and bow_logit, grouped by length,
  193|     one evaluation at the last step, final checkpoint kept."""
  194|     import types
  195| 
  196|     import pandas as pd
  197| 
  198|     import train
  199|     monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
  200|     rng = np.random.default_rng(0)
  201| 
  202|     def text():
  203|         return " ".join(f"w{j}" for j in rng.integers(0, 40, rng.integers(3, 12)))
  204| 
  205|     def frame(n):
  206|         y = np.arange(n) % 2
  207|         return pd.DataFrame({"user_input": [text() for _ in range(n)], "toxicity": y})
  208| 
  209|     monkeypatch.setattr(train, "load_train_val",
  210|                         lambda val_fraction=0.1, seed=13, max_samples=None: (frame(24), frame(16)))
  211|     monkeypatch.setattr(train, "AutoTokenizer",
  212|                         types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tokenizer()))
  213|     monkeypatch.setattr(train, "AutoModelForSequenceClassification",
  214|                         types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_model(0)))
  215|     plain = [{"user_input": text(), "toxicity": int(i % 2)} for i in range(10)]
  216|     rich = [{"user_input": text(), "toxicity": int(i % 2),
  217|              "toxicity_soft": 0.95 if i % 2 else 0.05, "bow_logit": float(rng.normal(0, 2))}
  218|             for i in range(20)]
  219|     p_plain, p_rich = tmp_path / "p1_copy.jsonl", tmp_path / "p3.jsonl"
  220|     _write(p_plain, plain)
  221|     _write(p_rich, rich)
  222|     out, meta_path = tmp_path / "out", tmp_path / "meta.json"
  223|     monkeypatch.setattr(sys, "argv", [
  224|         "train.py", "--model", "tiny", "--output_dir", str(out),
  225|         "--extra_train_jsonl", str(p_plain), str(p_plain), str(p_rich),
  226|         "--soft_label_field", "toxicity_soft", "--bow_logit_field", "bow_logit",
  227|         "--loss", "weighted_ce", "--class_weights", "1.0,1.0",
  228|         "--lr", "1e-3", "--batch_size", "4", "--grad_accum", "1", "--max_len", "32",
  229|         "--rdrop_alpha", "0", "--allow_tf32", "--group_by_length",
  230|         "--max_steps", "6", "--eval_steps", "6",
  231|         "--seed", "13", "--head_seed", "13", "--split_seed", "13", "--run_meta", str(meta_path)])
  232|     train.main()
  233|     meta = json.load(open(meta_path))
  234|     assert meta["global_step"] == 6 and meta["best_checkpoint_step"] == 6
  235|     assert meta["train_rows"] == 24 + 2 * len(plain) + len(rich)
  236|     assert meta["soft_targets"] and meta["bow_residual"] and meta["group_by_length"]
  237|     assert all(np.isfinite(h["loss"]) for h in meta["log_history"] if "loss" in h)
  238|     assert (out / "model.safetensors").exists() or (out / "pytorch_model.bin").exists()
  239| 
  240| 
  241| def test_missing_soft_target_is_refused(tmp_path, monkeypatch):
  242|     """A row without its soft target must stop the run instead of becoming a NaN loss."""
  243|     import types
  244| 
  245|     import pandas as pd
  246|     import pytest
  247| 
  248|     import train
  249|     monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
  250|     df = pd.DataFrame({"user_input": ["w1 w2 w3"] * 8, "toxicity": [0, 1] * 4})
  251|     monkeypatch.setattr(train, "load_train_val",
  252|                         lambda val_fraction=0.1, seed=13, max_samples=None: (df, df))
  253|     monkeypatch.setattr(train, "AutoTokenizer",
  254|                         types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tokenizer()))
  255|     bad = tmp_path / "bad.jsonl"
  256|     _write(bad, [{"user_input": "w1 w2", "toxicity": 1, "toxicity_soft": 0.95},
  257|                  {"user_input": "w3 w4", "toxicity": 0}])
  258|     monkeypatch.setattr(sys, "argv", ["train.py", "--output_dir", str(tmp_path / "o"),
  259|                                       "--extra_train_jsonl", str(bad),
  260|                                       "--soft_label_field", "toxicity_soft"])
  261|     with pytest.raises(SystemExit):
  262|         train.main()
```

## `training/test_v6_build.py`

```python
    1| """Tests for v6_build.py: de-duplication, decontamination matching, caps and held-back rows. CPU only; synthetic text."""
    2| import numpy as np
    3| import pandas as pd
    4| 
    5| import v6_build as vb
    6| from decontam import decontaminate
    7| 
    8| WORDS = [f"w{i}" for i in range(60)]
    9| 
   10| 
   11| def _texts(n, seed):
   12|     rng = np.random.default_rng(seed)
   13|     return [" ".join(rng.choice(WORDS, size=rng.integers(2, 15))) for _ in range(n)]
   14| 
   15| 
   16| def test_lexical_index_matches_decontaminate_exact_and_jaccard():
   17|     refs = _texts(150, 1)
   18|     cands = _texts(300, 2)
   19|     # plant exact (case/punct changes) and near-duplicates
   20|     cands += [r.upper() + "!" for r in refs[:20]]
   21|     cands += [r + " w1" for r in refs[20:40] if len(r.split()) >= 12]
   22|     idx = vb.RefIndex({"a": refs[:75], "b": refs[75:]})
   23|     mine = {i for i, t in enumerate(cands) if idx.lexical_one(t)}
   24|     ref = set(decontaminate(cands, refs, embedding=False)["removed_idx"])
   25|     assert mine == ref and len(ref) >= 20
   26| 
   27| 
   28| def test_per_set_attribution_counts_every_matching_set():
   29|     idx = vb.RefIndex({"a": ["alpha beta gamma delta"], "b": ["Alpha beta, gamma delta"]})
   30|     assert idx.lexical_one("alpha beta gamma delta") == {0: "exact", 1: "exact"}
   31| 
   32| 
   33| def test_dedup_within_drops_conflicts_and_duplicates():
   34|     df = pd.DataFrame({"row_id": list("abcde"), "user_input": ["Hi there", "hi there!", "x y", "X Y", ""],
   35|                        "toxicity": [0, 0, 0, 1, 0]})
   36|     out, rep = vb.dedup_within(df)
   37|     assert list(out["row_id"]) == ["a"]
   38|     assert rep["conflict_texts"] == 1 and rep["dup_rows"] == 1 and rep["empty"] == 1
   39| 
   40| 
   41| def test_dedup_across_p1_wins_and_p2_p5_conflicts_drop_all():
   42|     mk = lambda ids, texts, ys: vb.dedup_within(pd.DataFrame(  # noqa: E731
   43|         {"row_id": ids, "user_input": texts, "toxicity": ys}))[0]
   44|     pools = {"P2": mk(["p2a", "p2b", "p2c"], ["in p1", "disputed", "shared"], [1, 0, 1]),
   45|              "P3": mk(["p3a", "p3b"], ["disputed", "shared"], [1, 1])}
   46|     out, rep = vb.dedup_across({"in p1": 0}, pools)
   47|     assert list(out["P2"]["row_id"]) == ["p2c"] and list(out["P3"]["row_id"]) == []
   48|     assert rep["P2"]["dup_or_conflict_with_P1"] == 1 and rep["P3"]["conflict_P2_P5"] == 1
   49|     assert rep["P3"]["dup_of_earlier_pool"] == 1
   50| 
   51| 
   52| def test_stratified_cap_and_holdout_are_proportional_and_deterministic():
   53|     df = pd.DataFrame({"type": ["a"] * 700 + ["b"] * 300, "toxicity": [0, 1] * 500,
   54|                        "row_id": range(1000), "user_input": [str(i) for i in range(1000)]})
   55|     s1, _ = vb.stratified_cap(df, ["type", "toxicity"], 100, np.random.default_rng(13))
   56|     s2, _ = vb.stratified_cap(df, ["type", "toxicity"], 100, np.random.default_rng(13))
   57|     assert len(s1) == 100 and list(s1["row_id"]) == list(s2["row_id"])
   58|     assert (s1["type"] == "a").sum() == 70
   59|     tr, ho = vb.holdout(df, ["type", "toxicity"], np.random.default_rng(13))
   60|     # 3% per stratum, Python round (half to even): 350 -> 10, 150 -> 4
   61|     assert len(ho) == 2 * round(0.03 * 350) + 2 * round(0.03 * 150) == 28
   62|     assert len(tr) + len(ho) == 1000 and not set(tr["row_id"]) & set(ho["row_id"])
```

## `training/test_v6_committee.py`

```python
    1| """Tests for v6_committee_targets.py (the teacher-committee targets for R3). CPU only:  python3 -m pytest -q test_v6_committee.py"""
    2| import json
    3| import pathlib
    4| import sys
    5| 
    6| import numpy as np
    7| import pytest
    8| 
    9| sys.path.insert(0, str(pathlib.Path(__file__).parent))
   10| import v6_committee_targets as ct  # noqa: E402
   11| 
   12| 
   13| def calib(rng, n=400, scale=1.0, shift=0.0):
   14|     y = rng.integers(0, 2, n)
   15|     return (list(scale * (rng.normal(0, 1, n) + 2.5 * (2 * y - 1)) + shift), list(y))
   16| 
   17| 
   18| def test_rules_and_counts():
   19|     rng = np.random.default_rng(0)
   20|     cal = {"A": calib(rng), "B": calib(rng, scale=3.0, shift=1.0), "C": calib(rng)}
   21|     rows = [
   22|         {"row_id": 1, "user_input": "t1", "pool": "P4", "toxicity": 0, "toxicity_soft": 0.05},   # agrees
   23|         {"row_id": 2, "user_input": "t2", "pool": "P4", "toxicity": 0, "toxicity_soft": 0.05},   # disputed label
   24|         {"row_id": 3, "user_input": "t3", "pool": "P5", "toxicity": 1, "toxicity_soft": 0.95},   # disputed label
   25|         {"row_id": 4, "user_input": "t4", "pool": "wildchat", "toxicity": None, "toxicity_soft": None},
   26|         {"row_id": 5, "user_input": "t5", "pool": "wildchat", "toxicity": None, "toxicity_soft": None},
   27|         {"row_id": 6, "user_input": "t6", "pool": "wildchat", "toxicity": None, "toxicity_soft": None},  # unscored
   28|         {"row_id": 7, "user_input": "t7", "pool": "P1", "toxicity": 1},                           # never scored
   29|         {"row_id": 8, "user_input": "t8", "pool": "P4", "toxicity": 1, "toxicity_soft": 0.95},   # one member only
   30|     ]
   31|     m = {"A": {1: -6.0, 2: 7.0, 3: -7.0, 4: 6.0, 5: -0.2, 8: -9.0},
   32|          "B": {1: -18.0, 2: 21.0, 3: -21.0, 4: 18.0, 5: -0.5},
   33|          "C": {1: -6.0, 2: 7.0, 3: -7.0, 4: 6.0, 5: 0.1}}
   34|     out, rep = ct.build_targets(rows, m, cal)
   35|     by = {r["row_id"]: r for r in out}
   36|     assert by[1]["target_rule"] == "source" and by[1]["toxicity_soft"] == 0.05
   37|     assert by[2]["target_rule"] == "clash" and by[2]["toxicity_soft"] == 0.5 and by[2]["toxicity"] == 0
   38|     assert by[3]["target_rule"] == "clash" and by[3]["toxicity_soft"] == 0.5 and by[3]["toxicity"] == 1
   39|     assert by[4]["target_rule"] == "committee" and by[4]["toxicity"] == 1 and by[4]["toxicity_soft"] == 0.98
   40|     assert by[5]["target_rule"] == "committee" and 0.02 <= by[5]["toxicity_soft"] <= 0.98
   41|     assert 6 not in by, "an unlabelled row nobody scored is dropped"
   42|     assert by[7]["target_rule"] == "source_unscored" and by[7]["toxicity_soft"] == 1.0
   43|     assert by[8]["target_rule"] == "source_unscored", "one member is below min_members: no clash call"
   44|     assert rep["clash_rows"] == 2 and rep["rows_out"] == 7
   45|     assert rep["per_pool"]["wildchat"]["dropped_unlabelled_unscored"] == 1
   46|     assert all(p["a"] > 0 for p in rep["members"].values())
   47|     assert all(0.0 <= r["toxicity_soft"] <= 1.0 for r in out)
   48| 
   49| 
   50| def test_wrong_sign_margin_is_refused():
   51|     rng = np.random.default_rng(1)
   52|     m, y = calib(rng)
   53|     with pytest.raises(SystemExit):
   54|         ct.build_targets([], {"A": {}}, {"A": ([-v for v in m], y)})
   55| 
   56| 
   57| def test_scale_invariance_of_calibration():
   58|     """A member whose margins are on a different scale must give the same calibrated verdicts."""
   59|     rng = np.random.default_rng(2)
   60|     y = rng.integers(0, 2, 2000)
   61|     m = rng.normal(0, 1, 2000) + 1.0 * (2 * y - 1)  # overlapping classes, as real guard margins are
   62|     a1, b1 = ct.platt(list(m), list(y))
   63|     a2, b2 = ct.platt(list(3 * m + 7), list(y))
   64|     x = np.linspace(-3, 3, 13)
   65|     assert np.allclose(a1 * x + b1, a2 * (3 * x + 7) + b2, atol=1e-3)
   66|     assert 1.6 < a1 < 2.4, "for unit-variance classes at +/-1 the true slope is 2"
   67| 
   68| 
   69| def test_cli_round_trip(tmp_path):
   70|     rng = np.random.default_rng(3)
   71| 
   72|     def w(name, rows):
   73|         path = tmp_path / name
   74|         with open(path, "w") as f:
   75|             for r in rows:
   76|                 f.write(json.dumps(r) + "\n")
   77|         return str(path)
   78| 
   79|     rows = [{"row_id": i, "user_input": f"t{i}", "pool": "P4", "toxicity": i % 2,
   80|              "toxicity_soft": 0.95 if i % 2 else 0.05} for i in range(20)]
   81|     rows += [{"row_id": 100 + i, "user_input": f"u{i}", "pool": "harmaug", "toxicity": None,
   82|               "toxicity_soft": None} for i in range(10)]
   83|     args = ["--rows", w("rows.jsonl", rows), "--out", str(tmp_path / "out.jsonl"),
   84|             "--report", str(tmp_path / "rep.json")]
   85|     for n in ("A", "B"):
   86|         cm, cy = calib(rng)
   87|         args += ["--calib", f"{n}=" + w(f"cal_{n}.jsonl", [{"margin": a, "y": int(b)} for a, b in zip(cm, cy)])]
   88|         args += ["--member", f"{n}=" + w(f"m_{n}.jsonl", [{"row_id": r["row_id"], "margin": float(rng.normal(0, 3))}
   89|                                                           for r in rows])]
   90|     rep = ct.main(args)
   91|     out = [json.loads(l) for l in open(tmp_path / "out.jsonl")]
   92|     assert len(out) == 30 and rep["rows_out"] == 30
   93|     assert all(r["toxicity"] in (0, 1) and r["toxicity_soft"] is not None for r in out)
```

## `training/test_v6_metrics.py`

```python
    1| """Unit tests for v6_metrics.py: dev metrics, the frozen thresholds and the two-model average. CPU, numpy only."""
    2| import math
    3| 
    4| import numpy as np
    5| import pytest
    6| 
    7| import v6_metrics as vm
    8| 
    9| 
   10| def test_probs_from_margin_matches_softmax_and_does_not_saturate():
   11|     m = np.array([-50.0, -3.0, 0.0, 2.5, 40.0])
   12|     l0, l1 = np.zeros_like(m), m
   13|     sm = np.exp(l1) / (np.exp(l0) + np.exp(l1))
   14|     p = vm.probs_from_margin(m)
   15|     assert np.allclose(p, sm, rtol=0, atol=1e-15)
   16|     assert p[0] > 0 and p[-1] <= 1.0
   17|     assert np.all(np.diff(vm.probs_from_margin(np.array([-40.0, -39.0, -38.0]))) > 0)
   18| 
   19| 
   20| def test_two_model_average_is_mean_margin():
   21|     m, p = vm.two_model_average([2.0, -4.0], [0.0, 2.0])
   22|     assert np.allclose(m, [1.0, -1.0])
   23|     assert np.allclose(p, 1 / (1 + np.exp(-m)))
   24| 
   25| 
   26| def test_strict_threshold_budget_and_ties():
   27|     rng = np.random.default_rng(0)
   28|     b = rng.random(1000)
   29|     t = vm.strict_threshold(b)
   30|     assert (b >= t).mean() <= 0.01
   31|     # lowest such threshold: the next-lower candidate (the (k+1)-th largest itself) breaks budget
   32|     k = math.floor(0.01 * b.size)
   33|     assert (b >= np.sort(b)[::-1][k]).mean() > 0.01
   34|     # ties at the boundary are never flagged past the budget
   35|     b2 = np.r_[np.full(50, 0.9), np.zeros(950)]
   36|     t2 = vm.strict_threshold(b2)
   37|     assert (b2 >= t2).mean() <= 0.01 and t2 > 0.9
   38| 
   39| 
   40| def test_prescreen_threshold_recall_and_ties():
   41|     u = np.r_[np.linspace(0.1, 0.9, 98), [0.05, 0.01]]
   42|     t = vm.prescreen_threshold(u)
   43|     assert (u >= t).mean() >= 0.98
   44|     assert t == pytest.approx(0.1)   # the 98th largest
   45|     tied = np.r_[np.full(10, 0.5), np.ones(90)]
   46|     ts = vm.prescreen_threshold(tied)
   47|     assert ts == 0.5 and (tied >= ts).mean() == 1.0
   48|     # k = ceil(0.98 * n): n = 51 -> k = 50, so one item may fall below
   49|     u51 = np.arange(51, dtype=float)
   50|     assert vm.prescreen_threshold(u51) == 1.0
   51| 
   52| 
   53| def test_balanced_threshold_maximises_mean_f1_and_breaks_ties_low():
   54|     y1, s1 = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
   55|     y2, s2 = np.array([0, 1]), np.array([0.3, 0.7])
   56|     t, f = vm.balanced_threshold([(y1, s1), (y2, s2)])
   57|     assert f == pytest.approx(1.0)
   58|     assert t == pytest.approx(0.7)   # lowest threshold reaching mean F1 = 1 is 0.7 (0.3 < t <= 0.7)
   59|     # brute force check on random data
   60|     rng = np.random.default_rng(1)
   61|     sets = [(rng.integers(0, 2, 60), rng.random(60)) for _ in range(3)]
   62|     t, f = vm.balanced_threshold(sets)
   63|     import metrics_core as mc
   64|     cand = np.unique(np.concatenate([s for _, s in sets]))
   65|     brute = [np.mean([mc.f1_at(y, s, c) for y, s in sets]) for c in cand]
   66|     assert f == pytest.approx(max(brute))
   67|     assert t == pytest.approx(cand[int(np.argmax(brute))])
   68| 
   69| 
   70| def test_pair_accuracy_ties_half():
   71|     acc, n = vm.pair_accuracy([0.9, 0.1, 0.5, 0.5, 0.2, 0.3], [1, 0, 1, 0, 1, 0],
   72|                               ["a", "a", "b", "b", "c", "c"])
   73|     assert n == 3 and acc == pytest.approx((1 + 0.5 + 0) / 3)
   74| 
   75| 
   76| def test_pinned_score_share_is_strict():
   77|     assert vm.pinned_score_share([9.0, -9.0, 9.01, -12, 0]) == pytest.approx(2 / 5)
   78| 
   79| 
   80| def test_dev_report_runs():
   81|     rng = np.random.default_rng(2)
   82| 
   83|     def mk(n, pairs=False):
   84|         y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
   85|         m = rng.normal(0, 3, n) + 2 * y
   86|         d = {"label": y.tolist(), "margin": m.tolist(), "score": vm.probs_from_margin(m).tolist()}
   87|         if pairs:
   88|             d["pair_id"] = [f"p{i % (n // 2)}" for i in range(n)]
   89|         return d
   90| 
   91|     sets = {"twins_prompt": mk(40, True), "dev_pair2": mk(50), "tc_val": mk(60),
   92|             "aegis_val": mk(60), "dev_holdout_p3": mk(30), "dev_holdout_p4": mk(30),
   93|             "dev_holdout_p5": {"label": [0] * 10, "margin": [0.0] * 10, "score": [0.5] * 10}}
   94|     r = vm.dev_report(sets)
   95|     assert r["dev_intent"] == pytest.approx((r["twins_auroc"] + r["dev_pair2_auroc"]) / 2)
   96|     for k in ("t_gate", "t_screen", "t_bal", "saturation", "dev_holdout_p5_frr_at_gate"):
   97|         assert k in r
```

## `training/test_v6_score.py`

```python
    1| """Tests for v6_score.py: the train.py command for each recipe variant, and margin-storing scorecards."""
    2| import numpy as np
    3| import pytest
    4| 
    5| import v6_score as vs
    6| 
    7| 
    8| def _cmd(variant, **kw):
    9|     return vs.build_train_cmd("py", variant, 13, "/vol/v6", "/vol/v6/base/deberta-v3-large",
   10|                               "/tmp/m", "/tmp/rm.json", 1234, **kw)
   11| 
   12| 
   13| def test_r4_is_r1_without_p5():
   14|     c, r1 = _cmd("R4"), _cmd("R1")
   15|     files = lambda cmd: [e.rsplit("/", 1)[1] for e in cmd if e.endswith(".jsonl")]  # noqa: E731
   16|     assert files(c) == ["p1_copy.jsonl", "p1_copy.jsonl", "p2.jsonl", "p3.jsonl", "p4.jsonl"]
   17|     assert files(r1) == files(c) + ["p5.jsonl"]
   18|     assert all("/v6/pools/" in e for e in c if e.endswith(".jsonl"))
   19|     strip = lambda cmd: [x for x in cmd if not x.endswith(".jsonl")]  # noqa: E731
   20|     assert strip(c) == strip(r1)          # identical recipe, flags and S
   21|     # R4 row count = 3 ToxicChat copies + Aegis 2.0 + WildGuardMix + WildJailbreak (release file counts)
   22|     assert 3 * 4573 + 21132 + 43080 + 58200 == 136131
   23| 
   24| 
   25| def _val(cmd, flag):
   26|     return cmd[cmd.index(flag) + 1]
   27| 
   28| 
   29| def test_r1_command_matches_the_recipe():
   30|     c = _cmd("R1")
   31|     assert _val(c, "--lr") == "2e-05" and _val(c, "--batch_size") == "32"
   32|     assert _val(c, "--grad_accum") == "1" and _val(c, "--max_len") == "512"
   33|     assert _val(c, "--rdrop_alpha") == "0" and _val(c, "--class_weights") == "1.0,1.0"
   34|     assert _val(c, "--loss") == "weighted_ce" and _val(c, "--soft_label_field") == "toxicity_soft"
   35|     assert _val(c, "--max_steps") == "1234" and _val(c, "--eval_steps") == "1234"
   36|     assert _val(c, "--seed") == "13" and _val(c, "--head_seed") == "13"
   37|     assert _val(c, "--split_seed") == "13" and _val(c, "--val_fraction") == "0.10"
   38|     assert "--allow_tf32" in c and "--group_by_length" in c
   39|     assert "--warm_start_model" not in c and "--push_to_hub" not in c
   40|     assert "--bow_logit_field" not in c
   41|     i = c.index("--extra_train_jsonl")
   42|     extra = c[i + 1:i + 7]
   43|     assert [e.rsplit("/", 1)[1] for e in extra] == ["p1_copy.jsonl", "p1_copy.jsonl", "p2.jsonl",
   44|                                                      "p3.jsonl", "p4.jsonl", "p5.jsonl"]
   45| 
   46| 
   47| def test_r0_drops_p5_and_benign_p4():
   48|     c = _cmd("R0")
   49|     i = c.index("--extra_train_jsonl")
   50|     assert [e.rsplit("/", 1)[1] for e in c[i + 1:i + 6]] == [
   51|         "p1_copy.jsonl", "p1_copy.jsonl", "p2.jsonl", "p3.jsonl", "p4_r0.jsonl"]
   52|     assert "p5.jsonl" not in " ".join(c)
   53| 
   54| 
   55| def test_r2_reads_bow_files_and_sets_flag():
   56|     c = _cmd("R2")
   57|     assert _val(c, "--bow_logit_field") == "bow_logit"
   58|     assert all("/r2/" in e for e in c if e.endswith(".jsonl"))
   59| 
   60| 
   61| def test_oom_fallback_batch():
   62|     c = _cmd("R1", batch_size=16, grad_accum=2)
   63|     assert _val(c, "--batch_size") == "16" and _val(c, "--grad_accum") == "2"
   64| 
   65| 
   66| def test_scorecard_has_margin_and_no_text():
   67|     rows = [{"row_id": "x:1", "user_input": "secret text", "toxicity": 1},
   68|             {"row_id": "x:2", "user_input": "other", "toxicity": 0}]
   69|     m = np.array([3.0, -40.0])
   70|     from v6_metrics import probs_from_margin
   71|     card = vs.scorecard("R1_s13", "set", rows, probs_from_margin(m), m)
   72|     assert set(card["rows"][0]) == {"i", "src", "label", "score", "margin", "decision"}
   73|     assert "secret" not in str(card) and card["rows"][1]["margin"] == -40.0
   74|     assert card["rows"][1]["score"] > 0
   75| 
   76| 
   77| # ---- the test lock ------------------------------------------------------------------------ #
   78| def _seal(tmp_path, text="selection v1"):
   79|     md = tmp_path / vs.SELECTION
   80|     md.write_text(text)
   81|     vs.seal_selection(tmp_path)
   82|     return md
   83| 
   84| 
   85| def _never(*a, **k):
   86|     raise AssertionError("data or model touched while the test sets are locked")
   87| 
   88| 
   89| @pytest.mark.parametrize("name", vs.TEST_SETS)
   90| def test_every_test_set_is_locked_without_a_selection_seal(tmp_path, name):
   91|     with pytest.raises(vs.TestSetLocked):
   92|         vs.assert_test_unlocked(name, tmp_path)
   93|     with pytest.raises(vs.TestSetLocked):
   94|         vs.score_test_sets("R1_s13", [name], tmp_path, _never, _never)
   95| 
   96| 
   97| def test_an_unsealed_record_keeps_the_lock(tmp_path):
   98|     (tmp_path / vs.SELECTION).write_text("selection v1")
   99|     with pytest.raises(vs.TestSetLocked):
  100|         vs.assert_test_unlocked("xstest", tmp_path)
  101| 
  102| 
  103| def test_lock_checks_all_sets_before_loading_any(tmp_path):
  104|     with pytest.raises(vs.TestSetLocked):
  105|         vs.score_test_sets("R1_s13", ["tc_val", "orbench"], tmp_path, _never, _never)
  106| 
  107| 
  108| def test_unknown_set_names_are_refused_even_when_sealed(tmp_path):
  109|     _seal(tmp_path)
  110|     with pytest.raises(vs.TestSetLocked):
  111|         vs.assert_test_unlocked("orbench_renamed", tmp_path)
  112| 
  113| 
  114| def test_drifted_or_partial_seal_keeps_the_lock(tmp_path):
  115|     md = _seal(tmp_path)
  116|     vs.assert_test_unlocked("xstest", tmp_path)
  117|     md.write_text("selection v1, edited after sealing")
  118|     with pytest.raises(vs.TestSetLocked):
  119|         vs.assert_test_unlocked("xstest", tmp_path)
  120|     (tmp_path / vs.SELECTION_SEAL).unlink()
  121|     with pytest.raises(vs.TestSetLocked):
  122|         vs.assert_test_unlocked("xstest", tmp_path)
  123| 
  124| 
  125| def test_a_seal_cannot_be_replaced_for_different_content(tmp_path):
  126|     md = _seal(tmp_path)
  127|     assert vs.seal_selection(tmp_path)["sha256"] == vs.record_digest(md)[0]   # resealing the same is a no-op
  128|     md.write_text("selection v2")
  129|     with pytest.raises(vs.TestSetLocked):
  130|         vs.seal_selection(tmp_path)
  131| 
  132| 
  133| def test_dev_sets_are_never_locked(tmp_path):
  134|     for n in vs.DEV_SETS:
  135|         vs.assert_test_unlocked(n, tmp_path)
  136| 
  137| 
  138| def test_sealed_selection_unlocks_and_scores(tmp_path):
  139|     _seal(tmp_path)
  140|     rows = [{"row_id": "jbb:0", "toxicity": 1}, {"row_id": "jbb:1", "toxicity": 0}]
  141|     m = np.array([2.0, -2.0])
  142|     from v6_metrics import probs_from_margin
  143|     out = vs.score_test_sets("R1_s13", ["jbb"], tmp_path, lambda n: rows,
  144|                              lambda r: (probs_from_margin(m), m), threshold=0.5)
  145|     assert [r["decision"] for r in out["jbb"]["rows"]] == [1, 0]
  146|     assert out["jbb"]["rows"][0]["margin"] == 2.0
  147| 
  148| 
  149| def test_twins_pair_id_parsing():
  150|     card = {"rows": [{"src": "calm:abc:def:1", "score": 0.9, "margin": 2.0, "label": 1},
  151|                      {"src": "calm:abc:def:0", "score": 0.1, "margin": -2.0, "label": 0}]}
  152|     s = vs.dev_sets_for_report({"twins_prompt": card})
  153|     assert s["twins_prompt"]["pair_id"] == ["abc:def", "abc:def"]
```

## `training/test_v6_select.py`

```python
    1| """Tests for v6_select.select (the selection rule and the ToxicChat check) on synthetic candidate summaries."""
    2| import v6_select as s
    3| 
    4| 
    5| def c(intent, rng, auprc):
    6|     return {"dev_intent": {"mean": intent, "range": rng}, "tc_val_auprc": {"mean": auprc, "range": 0.0}}
    7| 
    8| 
    9| def test_r1_wins_when_lead_is_within_range():
   10|     sel, _ = s.select({"R1": c(0.80, 0.03, 0.85), "R0": c(0.82, 0.01, 0.85)})
   11|     assert sel == "R1"
   12| 
   13| 
   14| def test_clear_single_model_winner():
   15|     sel, _ = s.select({"R1": c(0.80, 0.01, 0.85), "R2": c(0.85, 0.02, 0.85)})
   16|     assert sel == "R2"
   17| 
   18| 
   19| def test_two_model_average_needs_to_beat_range():
   20|     base = {"R1": c(0.80, 0.01, 0.85)}
   21|     assert s.select(base | {"E11": c(0.81, 0.02, 0.85)})[0] == "R1"
   22|     assert s.select(base | {"E11": c(0.84, 0.02, 0.85)})[0] == "E11"
   23| 
   24| 
   25| def test_guardrail_falls_back_in_order_r1_last():
   26|     cands = {"R1": c(0.80, 0.01, 0.86), "R2": c(0.86, 0.01, 0.83), "E12": c(0.90, 0.01, 0.80),
   27|              "R0": c(0.82, 0.005, 0.855)}
   28|     sel, trace = s.select(cands)
   29|     # E12 replaces R2, fails the ToxicChat check; R2 fails it too; R0 passes
   30|     assert sel == "R0" and any("E12 fails" in t for t in trace)
   31|     cands["R0"] = c(0.82, 0.005, 0.70)
   32|     assert s.select(cands)[0] == "R1"
   33| 
   34| 
   35| def test_after_a_failed_single_model_the_next_must_still_beat_r1_by_its_range():
   36|     # R2 fails the ToxicChat check; R0 leads R1 by 0.02 < R1's range 0.03, so the choice is R1
   37|     cands = {"R1": c(0.80, 0.03, 0.86), "R2": c(0.90, 0.01, 0.80), "R0": c(0.82, 0.005, 0.86)}
   38|     sel, trace = s.select(cands)
   39|     assert sel == "R1" and any("R2 fails" in t for t in trace)
   40| 
   41| 
   42| def test_after_a_failed_two_model_average_the_other_averages_are_reconsidered():
   43|     cands = {"R1": c(0.80, 0.01, 0.86), "E12": c(0.90, 0.01, 0.80), "E11": c(0.86, 0.01, 0.86)}
   44|     sel, trace = s.select(cands)
   45|     assert sel == "E11" and any("E12 fails" in t for t in trace)
   46|     cands["E11"] = c(0.805, 0.01, 0.86)    # no longer leads R1 by more than the range
   47|     assert s.select(cands)[0] == "R1"
```

## `training/train.py`

```python
    1| #!/usr/bin/env python
    2| """
    3| train.py: fine-tune a DeBERTa-v3 encoder as a 2-class prompt-safety classifier.
    4| 
    5| It takes the ToxicChat (toxicchat0124) training split from tc_common.py, optionally appends
    6| extra JSONL training files, trains in FP32 with the Hugging Face Trainer, and writes the model,
    7| the tokenizer and threshold.json.
    8| 
    9|   * FP32 only. DeBERTa-v3 silently degrades in fp16/bf16, so mixed precision is
   10|     asserted OFF and the flags are never exposed.
   11|   * Class imbalance is handled with a class-weighted loss ("balanced" weights by
   12|     default, or frozen weights from --class_weights).
   13|   * Soft targets (--soft_label_field), a fixed step budget (--max_steps) and the
   14|     bag-of-words correction (--bow_logit_field) are optional.
   15|   * A decision threshold is tuned on the held-out validation split and saved to
   16|     threshold.json. No test set is touched here.
   17| 
   18| The loss is class-weighted cross-entropy over a 2-logit head (num_labels=2). On a
   19| binary problem this equals a class-weighted BCE on the logit difference, and a
   20| 2-class head loads as a plain AutoModelForSequenceClassification with id2label
   21| {0: non_toxic, 1: toxic}. `--loss focal` and `--loss plain` are also available.
   22| 
   23| Usage:
   24|   python train.py --output_dir checkpoints/toxicchat-deberta-v3-large
   25|   python train.py --model microsoft/deberta-v3-base   --loss focal   # fast fallback
   26|   python train.py --max_samples 400 --epochs 1                       # smoke test
   27| """
   28| import argparse
   29| import json
   30| import os
   31| 
   32| import numpy as np
   33| import torch
   34| import torch.nn.functional as F
   35| from datasets import Dataset
   36| from sklearn.utils.class_weight import compute_class_weight
   37| from transformers import (
   38|     AutoModelForSequenceClassification,
   39|     AutoTokenizer,
   40|     DataCollatorWithPadding,
   41|     Trainer,
   42|     TrainingArguments,
   43|     set_seed,
   44| )
   45| 
   46| from tc_common import (
   47|     ID2LABEL,
   48|     INPUT_FIELD,
   49|     LABEL2ID,
   50|     LABEL_FIELD,
   51|     TARGET_F1,
   52|     load_train_val,
   53|     metrics_from_scores,
   54|     softmax_np,
   55|     tune_threshold,
   56| )
   57| 
   58| 
   59| class WeightedLossTrainer(Trainer):
   60|     """Trainer with class-weighted CE or focal loss, optional R-Drop."""
   61| 
   62|     def __init__(self, *args, class_weights=None, loss_type="weighted_ce",
   63|                  focal_gamma=2.0, rdrop_alpha=0.0, **kwargs):
   64|         super().__init__(*args, **kwargs)
   65|         self.class_weights = class_weights
   66|         self.loss_type = loss_type
   67|         self.focal_gamma = focal_gamma
   68|         self.rdrop_alpha = rdrop_alpha
   69| 
   70|     def _base_loss(self, logits, labels, weight):
   71|         """Per-sample-reduced class-weighted CE or focal loss -> scalar."""
   72|         if self.loss_type == "focal":
   73|             ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
   74|             pt = torch.exp(-ce)
   75|             return ((1.0 - pt) ** self.focal_gamma * ce).mean()
   76|         # weighted_ce (default) / plain (weight is None)
   77|         return F.cross_entropy(logits, labels, weight=weight)
   78| 
   79|     def _soft_loss(self, logits, soft, weight):
   80|         """Class-weighted CE against soft targets [1-y, y]:
   81|             l_i = -(w1*y*log p1 + w0*(1-y)*log p0);  L = sum_i l_i / sum_i (w1*y + w0*(1-y)).
   82|         The denominator is F.cross_entropy(weight=...)'s weighted-mean normalisation, so this
   83|         equals weighted_ce exactly when every y is 0 or 1."""
   84|         if self.loss_type != "weighted_ce":
   85|             raise ValueError("soft targets are only defined for --loss weighted_ce")
   86|         y = soft.to(device=logits.device, dtype=logits.dtype)
   87|         logp = F.log_softmax(logits, dim=-1)
   88|         w0, w1 = (weight[0], weight[1]) if weight is not None else (1.0, 1.0)
   89|         num = -(w1 * y * logp[:, 1] + w0 * (1.0 - y) * logp[:, 0])
   90|         den = w1 * y + w0 * (1.0 - y)
   91|         return num.sum() / den.sum()
   92| 
   93|     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
   94|         labels = inputs.pop("labels")
   95|         soft = inputs.pop("soft_labels", None)
   96|         bow = inputs.pop("bow_logit", None)
   97|         outputs = model(**inputs)
   98|         logits = outputs.logits
   99|         weight = (
  100|             self.class_weights.to(device=logits.device, dtype=logits.dtype)
  101|             if self.class_weights is not None
  102|             else None
  103|         )
  104|         # Bag-of-words correction (recipe variant R2). In TRAINING only, the loss sees
  105|         # logits + [-b/2, +b/2], so the encoder gets gradient only for what the frozen
  106|         # bag-of-words expert cannot explain. Evaluation and inference use the raw logits,
  107|         # and `outputs` always carries the raw logits.
  108|         shift = None
  109|         if bow is not None and model.training:
  110|             b = bow.to(device=logits.device, dtype=logits.dtype)
  111|             shift = torch.stack([-0.5 * b, 0.5 * b], dim=1)
  112|         adj = (lambda lg: lg + shift) if shift is not None else (lambda lg: lg)  # noqa: E731
  113|         if soft is not None:
  114|             base = lambda lg: self._soft_loss(adj(lg), soft, weight)  # noqa: E731
  115|         else:
  116|             base = lambda lg: self._base_loss(adj(lg), labels, weight)  # noqa: E731
  117|         loss = base(logits)
  118| 
  119|         # R-Drop: a second stochastic-dropout forward + symmetric KL between the
  120|         # two predictive distributions. Zero inference cost (regularizer only);
  121|         # tightens the decision boundary on hard examples -> typically a small
  122|         # recall/AUPRC lift. Only active in training mode (dropout on) and alpha>0.
  123|         if self.rdrop_alpha > 0 and model.training:
  124|             logits2 = model(**inputs).logits
  125|             loss = 0.5 * (loss + base(logits2))
  126|             lp, lq = F.log_softmax(logits, dim=-1), F.log_softmax(logits2, dim=-1)
  127|             kl = 0.5 * (
  128|                 F.kl_div(lp, lq, reduction="batchmean", log_target=True)
  129|                 + F.kl_div(lq, lp, reduction="batchmean", log_target=True)
  130|             )
  131|             loss = loss + self.rdrop_alpha * kl
  132|         return (loss, outputs) if return_outputs else loss
  133| 
  134| 
  135| def split_hash(df):
  136|     """sha256 over a data split's (text, label) rows in order: identical iff the split is."""
  137|     import hashlib
  138|     h = hashlib.sha256()
  139|     for t, y in zip(df[INPUT_FIELD].astype(str), df[LABEL_FIELD].astype(int)):
  140|         h.update(t.encode("utf-8") + b"\x00" + str(y).encode() + b"\n")
  141|     return h.hexdigest()
  142| 
  143| 
  144| def make_compute_metrics():
  145|     """Eval-time metrics for checkpoint selection. Tunes the threshold on the
  146|     current val scores and reports F1 at that threshold (model is selected on
  147|     best achievable F1)."""
  148| 
  149|     def _fn(eval_pred):
  150|         logits = getattr(eval_pred, "predictions", None)
  151|         labels = getattr(eval_pred, "label_ids", None)
  152|         if logits is None:
  153|             logits, labels = eval_pred[0], eval_pred[1]
  154|         if isinstance(logits, tuple):
  155|             logits = logits[0]
  156|         scores = softmax_np(np.asarray(logits))[:, 1]
  157|         thr, _ = tune_threshold(labels, scores)
  158|         m = metrics_from_scores(labels, scores, thr)
  159|         return {
  160|             "f1": m["f1"],
  161|             "auprc": m["auprc"],
  162|             "precision": m["precision"],
  163|             "recall": m["recall"],
  164|             "threshold": m["threshold"],
  165|         }
  166| 
  167|     return _fn
  168| 
  169| 
  170| def parse_args():
  171|     p = argparse.ArgumentParser(description=__doc__)
  172|     p.add_argument("--model", default="microsoft/deberta-v3-large",
  173|                    help="encoder backbone (fallbacks: microsoft/deberta-v3-base, "
  174|                         "answerdotai/ModernBERT-base)")
  175|     p.add_argument("--output_dir", default="checkpoints/toxicchat-deberta-v3-large")
  176|     p.add_argument("--lr", type=float, default=1.5e-5)      # sweep {1e-5,1.5e-5,2e-5}
  177|     p.add_argument("--epochs", type=float, default=5.0)
  178|     p.add_argument("--batch_size", type=int, default=16)
  179|     p.add_argument("--grad_accum", type=int, default=1,
  180|                    help="gradient accumulation steps. EFFECTIVE batch = batch_size*grad_accum. "
  181|                         "R-Drop runs a 2nd forward pass (≈2x activation memory), so DeBERTa-v3-large "
  182|                         "len512 OOMs an 80GB A100 at bs16; use --batch_size 8 --grad_accum 2 to keep "
  183|                         "the effective batch at 16 with the same math at ~half the peak memory.")
  184|     p.add_argument("--eval_batch_size", type=int, default=64)
  185|     p.add_argument("--max_len", type=int, default=512)
  186|     p.add_argument("--warmup_ratio", type=float, default=0.10)
  187|     p.add_argument("--weight_decay", type=float, default=0.01)
  188|     p.add_argument("--grad_clip", type=float, default=1.0)
  189|     p.add_argument("--val_fraction", type=float, default=0.10)
  190|     p.add_argument("--seed", type=int, default=42)          # SEEDS = [13, 42, 99]
  191|     p.add_argument("--split_seed", type=int, default=None,
  192|                    help="seed for the stratified train/val split. Defaults to --seed. "
  193|                         "Pin it to give several runs the same held-out validation split.")
  194|     p.add_argument("--head_seed", type=int, default=None,
  195|                    help="seed for the reinitialized 2-class head. Defaults to --seed. "
  196|                         "Pin it to give several runs the same head initialisation.")
  197|     p.add_argument("--loss", choices=["weighted_ce", "focal", "plain"],
  198|                    default="weighted_ce")
  199|     p.add_argument("--focal_gamma", type=float, default=2.0)
  200|     p.add_argument("--rdrop_alpha", type=float, default=0.0,
  201|                    help="R-Drop weight on the symmetric-KL consistency term (0=off). "
  202|                         "Two dropout forwards per step regularize toward a tighter boundary; "
  203|                         "zero inference cost. ~0.5-1.0 is a reasonable range for a 2-class head.")
  204|     p.add_argument("--extra_train_jsonl", default=None, nargs="+",
  205|                    help="path to a JSONL of EXTRA decontaminated training rows, each "
  206|                         f"{{'{INPUT_FIELD}': str, '{LABEL_FIELD}': 0|1}}. Appended to the ToxicChat "
  207|                         "training split (never val/test). The file MUST be pre-decontaminated "
  208|                         "against BOTH test splits (decontam.py) — this loader does not re-check.")
  209|     p.add_argument("--soft_label_field", default=None,
  210|                    help="optional float field in the --extra_train_jsonl rows holding a soft target "
  211|                         "y in [0,1] (e.g. toxicity_soft). Rows without it use their hard label. "
  212|                         "Enables the soft-target weighted CE.")
  213|     p.add_argument("--bow_logit_field", default=None,
  214|                    help="optional float field in the --extra_train_jsonl rows holding a bag-of-words "
  215|                         "log-odds b (the bag-of-words correction of recipe variant R2). During TRAINING "
  216|                         "the loss is computed "
  217|                         "on logits + [-b/2, +b/2]; evaluation and inference use the raw logits. Rows "
  218|                         "without the field, and the ToxicChat loader rows, use b = 0.")
  219|     p.add_argument("--group_by_length", action="store_true",
  220|                    help="batch rows of similar token length together. Speed only.")
  221|     p.add_argument("--class_weights", default=None,
  222|                    help="frozen class weights 'w0,w1'. Default: computed "
  223|                         "'balanced' from this run's training labels.")
  224|     p.add_argument("--max_steps", type=int, default=None,
  225|                    help="fixed optimizer-step budget (overrides --epochs). Requires --eval_steps.")
  226|     p.add_argument("--eval_steps", type=int, default=None,
  227|                    help="evaluate + save every N steps (used with --max_steps).")
  228|     p.add_argument("--run_meta", default=None,
  229|                    help="write a JSON of step counts, class weights, the chosen checkpoint step "
  230|                         "and the loss log to this path.")
  231|     p.add_argument("--warm_start_model", default=None,
  232|                    help="init the ENCODER BODY from a toxicity checkpoint (a fresh 2-class head "
  233|                         "is reinitialized), e.g. cooperleong00/deberta-v3-large_toxicity-scorer. "
  234|                         "Release 2.0 does not use it.")
  235|     p.add_argument("--metric_for_best", choices=["auprc", "f1"], default="auprc",
  236|                    help="val metric for checkpoint selection. AUPRC is threshold-free and the more "
  237|                         "reliable bar to clear; F1 depends on the operating point.")
  238|     p.add_argument("--allow_tf32", action="store_true",
  239|                    help="allow TF32 matmuls (Ampere+): ~3-5x faster, usually negligible quality loss. "
  240|                         "This is NOT fp16/bf16 — the DeBERTa-v3 pitfall is reduced STORAGE precision, "
  241|                         "not TF32 accumulation. Off by default to stay strictly IEEE-FP32.")
  242|     p.add_argument("--max_samples", type=int, default=None,
  243|                    help="cap rows for a quick smoke test")
  244|     p.add_argument("--push_to_hub", action="store_true")
  245|     p.add_argument("--hub_model_id", default=None)
  246|     p.add_argument("--private", action="store_true", default=True)
  247|     return p.parse_args()
  248| 
  249| 
  250| def main():
  251|     args = parse_args()
  252|     set_seed(args.seed)
  253|     os.makedirs(args.output_dir, exist_ok=True)
  254|     if args.allow_tf32:
  255|         torch.backends.cuda.matmul.allow_tf32 = True
  256|         torch.backends.cudnn.allow_tf32 = True
  257|         print("[precision] TF32 matmuls ENABLED (still no fp16/bf16)")
  258|     init_from = args.warm_start_model or args.model
  259|     # The validation split and the head initialisation have their own seeds, so
  260|     # runs can share them while --seed still varies shuffling and dropout.
  261|     split_seed = args.seed if args.split_seed is None else args.split_seed
  262|     head_seed = args.seed if args.head_seed is None else args.head_seed
  263| 
  264|     # ---- data ------------------------------------------------------------- #
  265|     train_df, val_df = load_train_val(
  266|         val_fraction=args.val_fraction, seed=split_seed, max_samples=args.max_samples
  267|     )
  268|     val_split_sha256 = split_hash(val_df)
  269|     print(f"[data] val split sha256={val_split_sha256} (split_seed={split_seed})")
  270|     pos = int(train_df[LABEL_FIELD].sum())
  271|     print(f"[data] train={len(train_df)} (pos={pos}, {pos/len(train_df):.2%})  "
  272|           f"val={len(val_df)} (pos={int(val_df[LABEL_FIELD].sum())})")
  273| 
  274|     # ---- optional: append EXTRA pre-decontaminated training rows ----------- #
  275|     # Extra training rows from other datasets. Appended to TRAIN only; the
  276|     # validation split stays pure ToxicChat, so
  277|     # the frozen threshold and val-AUPRC selection remain on in-distribution data.
  278|     if args.extra_train_jsonl:
  279|         import pandas as pd
  280| 
  281|         soft_f = args.soft_label_field
  282|         bow_f = args.bow_logit_field
  283|         if soft_f:
  284|             train_df = train_df.copy()
  285|             train_df["soft_labels"] = train_df[LABEL_FIELD].astype(float)
  286|         if bow_f:
  287|             train_df = train_df.copy()
  288|             train_df["bow_logit"] = 0.0  # loader rows carry no bag-of-words score
  289|         before, epos = len(train_df), int(train_df[LABEL_FIELD].sum())
  290|         parts = [train_df[[INPUT_FIELD, LABEL_FIELD] + (["soft_labels"] if soft_f else [])
  291|                           + (["bow_logit"] if bow_f else [])]]
  292|         for path in args.extra_train_jsonl:
  293|             extra = pd.read_json(path, lines=True)
  294|             miss = {INPUT_FIELD, LABEL_FIELD} - set(extra.columns)
  295|             if miss:
  296|                 raise SystemExit(f"--extra_train_jsonl {path} missing columns {miss}")
  297|             extra = extra.dropna(subset=[INPUT_FIELD])
  298|             extra[LABEL_FIELD] = extra[LABEL_FIELD].astype(int)
  299|             cols = [INPUT_FIELD, LABEL_FIELD]
  300|             if soft_f:
  301|                 if soft_f in extra.columns:
  302|                     if extra[soft_f].isna().any():
  303|                         raise SystemExit(f"--soft_label_field: {path} has rows with no {soft_f} "
  304|                                          "(a missing soft target becomes a NaN loss)")
  305|                     extra["soft_labels"] = extra[soft_f].astype(float)
  306|                     n_soft = int(((extra["soft_labels"] > 0) & (extra["soft_labels"] < 1)).sum())
  307|                 else:
  308|                     extra["soft_labels"] = extra[LABEL_FIELD].astype(float)
  309|                     n_soft = 0
  310|                 cols.append("soft_labels")
  311|                 print(f"[extra] {path}: {n_soft} rows with a fractional soft target")
  312|             if bow_f:
  313|                 if bow_f in extra.columns:
  314|                     if extra[bow_f].isna().any():
  315|                         raise SystemExit(f"--bow_logit_field: {path} has rows with no {bow_f}")
  316|                     extra["bow_logit"] = extra[bow_f].astype(float)
  317|                 else:
  318|                     extra["bow_logit"] = 0.0
  319|                 cols.append("bow_logit")
  320|                 print(f"[extra] {path}: bag-of-words log-odds on "
  321|                       f"{int((extra['bow_logit'] != 0).sum())} rows")
  322|             parts.append(extra[cols])
  323|             print(f"[extra] +{len(extra)} rows (pos={int(extra[LABEL_FIELD].sum())}/{len(extra)}) "
  324|                   f"from {path}")
  325|         train_df = pd.concat(parts, ignore_index=True)
  326|         print(f"[extra] train {before}->{len(train_df)} "
  327|               f"(pos {epos}->{int(train_df[LABEL_FIELD].sum())})")
  328|     elif args.soft_label_field or args.bow_logit_field:
  329|         raise SystemExit("--soft_label_field / --bow_logit_field need --extra_train_jsonl")
  330|     use_soft = bool(args.soft_label_field)
  331|     use_bow = bool(args.bow_logit_field)
  332| 
  333|     tok = AutoTokenizer.from_pretrained(init_from)
  334| 
  335|     def to_hf(df, soft=False, bow=False):
  336|         cols = ([INPUT_FIELD, LABEL_FIELD] + (["soft_labels"] if soft else [])
  337|                 + (["bow_logit"] if bow else []))
  338|         d = df[cols].rename(columns={LABEL_FIELD: "labels"})
  339|         return Dataset.from_pandas(d, preserve_index=False)
  340| 
  341|     def tokenize(batch):
  342|         return tok(batch[INPUT_FIELD], truncation=True, max_length=args.max_len)
  343| 
  344|     train_ds = to_hf(train_df, soft=use_soft, bow=use_bow).map(
  345|         tokenize, batched=True, remove_columns=[INPUT_FIELD])
  346|     val_ds = to_hf(val_df).map(tokenize, batched=True, remove_columns=[INPUT_FIELD])
  347| 
  348|     # ---- model ------------------------------------------------------------ #
  349|     model = AutoModelForSequenceClassification.from_pretrained(
  350|         init_from, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
  351|         ignore_mismatched_sizes=bool(args.warm_start_model),
  352|         torch_dtype=torch.float32,  # force IEEE FP32 at LOAD. Newer transformers
  353|         # defaults to the checkpoint's stored dtype (often fp16), the DeBERTa-v3
  354|         # pitfall. fp16=False in TrainingArguments governs autocast, NOT load dtype.
  355|     )
  356|     if args.warm_start_model:
  357|         # ignore_mismatched_sizes only re-inits the head on a SHAPE mismatch. A
  358|         # toxicity checkpoint often already has a 2-class head, so it would
  359|         # otherwise transfer verbatim and carry the checkpoint's (possibly
  360|         # inverted) class polarity. Re-init the classifier deterministically so
  361|         # warm-start means "encoder body only".
  362|         # Only the classifier is reinitialised, under head_seed. The ContextPooler is
  363|         # kept from the checkpoint: on a warm-start checkpoint that has a head
  364|         # (e.g. cooperleong00) it loads verbatim, so it is the same for every seed.
  365|         set_seed(head_seed)
  366|         model.classifier.apply(model._init_weights)
  367|         print(f"[warm-start] encoder body <- {args.warm_start_model} "
  368|               f"(2-class head reinitialized @ head_seed={head_seed}; "
  369|               f"encoder is the transferred signal)")
  370| 
  371|     # ---- class weights ---------------------------------------------------- #
  372|     class_weights = None
  373|     if args.class_weights:
  374|         cw = np.array([float(x) for x in args.class_weights.split(",")], dtype=np.float64)
  375|         if cw.shape != (2,):
  376|             raise SystemExit("--class_weights must be 'w0,w1'")
  377|         class_weights = torch.tensor(cw, dtype=torch.float)
  378|         print(f"[loss] {args.loss}  class_weights={cw.tolist()} (FROZEN via --class_weights)")
  379|     elif args.loss in ("weighted_ce", "focal"):
  380|         cw = compute_class_weight(
  381|             "balanced", classes=np.array([0, 1]), y=train_df[LABEL_FIELD].values
  382|         )
  383|         class_weights = torch.tensor(cw, dtype=torch.float)
  384|         print(f"[loss] {args.loss}  class_weights={cw.round(3).tolist()}")
  385| 
  386|     # ---- training args (FP32 only, see the module docstring) -------------- #
  387|     if (args.max_steps is None) != (args.eval_steps is None):
  388|         raise SystemExit("--max_steps and --eval_steps go together")
  389|     step_kw = (
  390|         dict(max_steps=args.max_steps, eval_strategy="steps", eval_steps=args.eval_steps,
  391|              save_strategy="steps", save_steps=args.eval_steps)
  392|         if args.max_steps else dict(eval_strategy="epoch", save_strategy="epoch")
  393|     )
  394|     targs = TrainingArguments(
  395|         output_dir=args.output_dir,
  396|         num_train_epochs=args.epochs,
  397|         **step_kw,
  398|         remove_unused_columns=not (use_soft or use_bow),  # keep soft_labels / bow_logit
  399|         # transformers 5.x: the old group_by_length=True is train_sampling_strategy="group_by_length"
  400|         train_sampling_strategy="group_by_length" if args.group_by_length else "random",
  401|         per_device_train_batch_size=args.batch_size,
  402|         gradient_accumulation_steps=args.grad_accum,
  403|         per_device_eval_batch_size=args.eval_batch_size,
  404|         learning_rate=args.lr,
  405|         warmup_ratio=args.warmup_ratio,
  406|         weight_decay=args.weight_decay,
  407|         max_grad_norm=args.grad_clip,
  408|         load_best_model_at_end=True,
  409|         metric_for_best_model=args.metric_for_best,
  410|         greater_is_better=True,
  411|         save_total_limit=2,  # keep best + last; guards load_best_model_at_end
  412|         fp16=False,   # DO NOT enable. DeBERTa-v3 silently degrades in fp16/bf16.
  413|         bf16=False,   # DO NOT enable.
  414|         logging_steps=50,
  415|         report_to="none",
  416|         seed=args.seed,
  417|     )
  418|     assert not targs.fp16 and not targs.bf16, "DeBERTa-v3 must train in FP32"
  419| 
  420|     trainer = WeightedLossTrainer(
  421|         model=model,
  422|         args=targs,
  423|         train_dataset=train_ds,
  424|         eval_dataset=val_ds,
  425|         processing_class=tok,
  426|         data_collator=DataCollatorWithPadding(tokenizer=tok),
  427|         compute_metrics=make_compute_metrics(),
  428|         class_weights=class_weights,
  429|         loss_type=args.loss,
  430|         focal_gamma=args.focal_gamma,
  431|         rdrop_alpha=args.rdrop_alpha,
  432|     )
  433|     if args.rdrop_alpha > 0:
  434|         print(f"[rdrop] symmetric-KL consistency ON (alpha={args.rdrop_alpha})")
  435| 
  436|     trainer.train()
  437|     st = trainer.state
  438|     best_step = None
  439|     if st.best_model_checkpoint:
  440|         best_step = int(str(st.best_model_checkpoint).rstrip("/").rsplit("-", 1)[-1])
  441|     run_meta = {
  442|         "global_step": int(st.global_step),
  443|         "max_steps": int(st.max_steps),
  444|         "num_train_epochs": float(st.num_train_epochs),
  445|         "steps_per_epoch": (int(st.max_steps // st.num_train_epochs)
  446|                             if not args.max_steps else None),
  447|         "effective_epochs": float(st.epoch or 0.0),
  448|         "train_rows": int(len(train_df)),
  449|         "train_pos": int(train_df[LABEL_FIELD].sum()),
  450|         "class_weights": class_weights.tolist() if class_weights is not None else None,
  451|         "soft_targets": use_soft,
  452|         "bow_residual": use_bow,
  453|         "group_by_length": bool(args.group_by_length),
  454|         "best_checkpoint_step": best_step,
  455|         "val_carve_sha256": val_split_sha256,   # key name kept: stored run_meta files use it
  456|         "split_seed": split_seed,
  457|         "best_val_auprc": st.best_metric,
  458|         "log_history": st.log_history,
  459|     }
  460|     print(f"[steps] global_step={run_meta['global_step']} max_steps={run_meta['max_steps']} "
  461|           f"steps_per_epoch={run_meta['steps_per_epoch']} best_step={best_step}")
  462|     if args.run_meta:
  463|         with open(args.run_meta, "w") as f:
  464|             json.dump(run_meta, f, indent=1)
  465| 
  466|     # ---- freeze the decision threshold on val (NEVER on test) ------------- #
  467|     val_logits = trainer.predict(val_ds).predictions
  468|     if isinstance(val_logits, tuple):
  469|         val_logits = val_logits[0]
  470|     val_scores = softmax_np(np.asarray(val_logits))[:, 1]
  471|     val_labels = val_df[LABEL_FIELD].values
  472|     threshold, val_f1 = tune_threshold(val_labels, val_scores)
  473|     val_metrics = metrics_from_scores(val_labels, val_scores, threshold)
  474|     print(f"\n[val] frozen threshold={threshold:.4f}  "
  475|           f"F1={val_metrics['f1']:.4f}  AUPRC={val_metrics['auprc']:.4f}  "
  476|           f"P={val_metrics['precision']:.4f}  R={val_metrics['recall']:.4f}")
  477| 
  478|     # ---- save model + tokenizer + frozen threshold ------------------------ #
  479|     trainer.save_model(args.output_dir)
  480|     tok.save_pretrained(args.output_dir)
  481|     meta = {
  482|         "threshold": threshold,
  483|         "val_f1": val_f1,
  484|         "val_metrics": val_metrics,
  485|         "base_model": args.model,
  486|         "warm_start_model": args.warm_start_model,
  487|         "loss": args.loss,
  488|         "rdrop_alpha": args.rdrop_alpha,
  489|         "extra_train_jsonl": args.extra_train_jsonl,
  490|         "soft_label_field": args.soft_label_field,
  491|         "class_weights": class_weights.tolist() if class_weights is not None else None,
  492|         "max_steps": args.max_steps,
  493|         "eval_steps": args.eval_steps,
  494|         "batch_size": args.batch_size,
  495|         "grad_accum": args.grad_accum,
  496|         "lr": args.lr,
  497|         "epochs": args.epochs,
  498|         "seed": args.seed,
  499|         "split_seed": split_seed,
  500|         "head_seed": head_seed,
  501|         "max_len": args.max_len,
  502|         "val_fraction": args.val_fraction,
  503|         "target_f1": TARGET_F1,
  504|     }
  505|     with open(os.path.join(args.output_dir, "threshold.json"), "w") as f:
  506|         json.dump(meta, f, indent=2)
  507|     print(f"[save] model + tokenizer + threshold.json -> {args.output_dir}")
  508| 
  509|     if args.push_to_hub:
  510|         if not args.hub_model_id:
  511|             raise SystemExit("--push_to_hub requires --hub_model_id")
  512|         from huggingface_hub import HfApi
  513|         model.push_to_hub(args.hub_model_id, private=args.private)
  514|         tok.push_to_hub(args.hub_model_id, private=args.private)
  515|         HfApi().upload_file(
  516|             path_or_fileobj=os.path.join(args.output_dir, "threshold.json"),
  517|             path_in_repo="threshold.json",
  518|             repo_id=args.hub_model_id,
  519|             repo_type="model",
  520|         )
  521|         print(f"[hub] pushed -> {args.hub_model_id} (private={args.private})")
  522| 
  523|     print(f"\nFor reference: the published ToxicChat-T5-large test F1 is {TARGET_F1}")
  524| 
  525| 
  526| if __name__ == "__main__":
  527|     main()
```

## `training/train_image.py`

```python
    1| """train_image.py: the Modal image used for training, dev scoring and the tests.
    2| 
    3| torch and transformers are pinned exactly (2.12.1 and 5.12.1), so every run uses the same model
    4| code; the other packages have lower bounds only. The code in this folder is mounted at CODE_DIR.
    5| """
    6| import pathlib
    7| 
    8| import modal
    9| 
   10| LOCAL_DIR = pathlib.Path(__file__).parent
   11| CODE_DIR = "/root/code"  # where this folder's code lives inside the container
   12| 
   13| PINS = [
   14|     "torch==2.12.1",
   15|     "transformers==5.12.1",
   16|     "accelerate>=1.1.0",
   17|     "datasets>=2.18",
   18|     "scikit-learn>=1.3",
   19|     "scipy>=1.10",
   20|     "sentencepiece>=0.1.99",
   21|     "protobuf>=3.20",
   22|     "huggingface_hub>=0.23",
   23|     "numpy>=1.24",
   24|     "pandas>=2.0",
   25|     "pytest>=8",
   26| ]
   27| 
   28| image = (
   29|     modal.Image.debian_slim(python_version="3.11")
   30|     .pip_install(*PINS)
   31|     .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
   32|           "TOKENIZERS_PARALLELISM": "false"})
   33|     .add_local_python_source("train_image")  # so app files can import this module remotely
   34|     # code + committed scorecards only: never data files
   35|     .add_local_dir(LOCAL_DIR, remote_path=CODE_DIR,
   36|                    ignore=["__pycache__", "checkpoints", ".git", "*.pyc", "*.jsonl", "*.csv",
   37|                            "label_sheets", "chart-app", "demo", "leaderboard",
   38|                            "external_rerun", "harness", "release", "*.png"])
   39| )
```

## `training/v6_bow_crossfit.py`

```python
    1| #!/usr/bin/env python3
    2| """v6_bow_crossfit.py: cross-fitted bag-of-words log-odds for the training files.
    3| 
    4| One computation serves two purposes:
    5|   * a shortcut audit: how far vocabulary alone predicts the label in the training mix
    6|     (AUROC overall and per pool file);
    7|   * recipe variant R2 (base recipe + bag-of-words correction): the per-row `bow_logit` that
    8|     train.py --bow_logit_field adds to the logits during training.
    9| 
   10| For every row of the given JSONL files, a unigram TF-IDF + logistic regression is fitted on the
   11| other folds and its log-odds for the row is written as `bow_logit`, clipped to [-clip, +clip].
   12| Rows with the same normalised text always share a fold, so a duplicate is never scored by a model
   13| that saw its label. CPU only; prints counts and AUROCs, never text.
   14| 
   15|     python v6_bow_crossfit.py --inputs p1_copy.jsonl p2.jsonl p3.jsonl p4.jsonl p5.jsonl \
   16|         --out_dir r2_pools/ [--report bow_crossfit_report.json]
   17| 
   18| Each input is rewritten under --out_dir with the same name, the same rows in the same order, and
   19| one extra field. A path given twice is processed once.
   20| """
   21| import argparse
   22| import hashlib
   23| import json
   24| import os
   25| 
   26| import numpy as np
   27| 
   28| 
   29| def _norm(t):
   30|     return " ".join(str(t).lower().split())
   31| 
   32| 
   33| def parse_args(argv=None):
   34|     p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
   35|     p.add_argument("--inputs", nargs="+", required=True)
   36|     p.add_argument("--out_dir", required=True)
   37|     p.add_argument("--text_field", default="user_input")
   38|     p.add_argument("--label_field", default="toxicity")
   39|     p.add_argument("--out_field", default="bow_logit")
   40|     p.add_argument("--folds", type=int, default=5)
   41|     p.add_argument("--seed", type=int, default=13)
   42|     p.add_argument("--clip", type=float, default=6.0)
   43|     p.add_argument("--C", type=float, default=4.0)
   44|     p.add_argument("--min_df", type=int, default=2)
   45|     p.add_argument("--report", default=None, help="optional JSON path for the counts and AUROCs")
   46|     return p.parse_args(argv)
   47| 
   48| 
   49| def crossfit(texts, labels, folds=5, seed=13, C=4.0, min_df=2):
   50|     """Out-of-fold log-odds of a unigram TF-IDF + logistic regression. Duplicate texts share a fold."""
   51|     from sklearn.feature_extraction.text import TfidfVectorizer
   52|     from sklearn.linear_model import LogisticRegression
   53|     from sklearn.model_selection import StratifiedGroupKFold
   54| 
   55|     y = np.asarray(labels, dtype=int)
   56|     codes, groups = {}, np.empty(len(texts), dtype=np.int64)
   57|     for i, t in enumerate(texts):
   58|         h = hashlib.sha1(_norm(t).encode("utf-8")).hexdigest()
   59|         groups[i] = codes.setdefault(h, len(codes))
   60|     arr = np.asarray(texts, dtype=object)
   61|     oof = np.full(len(texts), np.nan)
   62|     skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
   63|     for tr, te in skf.split(arr, y, groups):
   64|         vec = TfidfVectorizer(ngram_range=(1, 1), min_df=min_df, sublinear_tf=True, lowercase=True)
   65|         clf = LogisticRegression(C=C, max_iter=2000)  # no class weighting: we want calibrated log-odds
   66|         clf.fit(vec.fit_transform(list(arr[tr])), y[tr])
   67|         oof[te] = clf.decision_function(vec.transform(list(arr[te])))
   68|     assert not np.isnan(oof).any(), "a row was left without an out-of-fold score"
   69|     return oof, len(codes)
   70| 
   71| 
   72| def _auroc(y, s):
   73|     from sklearn.metrics import roc_auc_score
   74|     y = np.asarray(y)
   75|     return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
   76| 
   77| 
   78| def main(argv=None):
   79|     a = parse_args(argv)
   80|     os.makedirs(a.out_dir, exist_ok=True)
   81|     paths = list(dict.fromkeys(a.inputs))  # unique, order kept
   82|     names = [os.path.basename(p) for p in paths]
   83|     if len(set(names)) != len(names):
   84|         raise SystemExit("two inputs share a file name; the outputs would collide")
   85|     rows, owner = [], []
   86|     for k, path in enumerate(paths):
   87|         with open(path) as f:
   88|             for line in f:
   89|                 if line.strip():
   90|                     r = json.loads(line)
   91|                     if r.get(a.text_field) is None or r.get(a.label_field) is None:
   92|                         raise SystemExit(f"{path}: a row lacks {a.text_field} or {a.label_field}")
   93|                     rows.append(r)
   94|                     owner.append(k)
   95|     owner = np.asarray(owner)
   96|     texts = [str(r[a.text_field]) for r in rows]
   97|     y = np.asarray([int(r[a.label_field]) for r in rows])
   98|     raw, n_groups = crossfit(texts, y, a.folds, a.seed, a.C, a.min_df)
   99|     b = np.clip(raw, -a.clip, a.clip)
  100| 
  101|     report = {"rows": int(len(rows)), "unique_texts": int(n_groups), "unsafe": int(y.sum()),
  102|               "folds": a.folds, "seed": a.seed, "clip": a.clip, "C": a.C, "min_df": a.min_df,
  103|               "auroc_overall": _auroc(y, raw), "share_at_clip": float((np.abs(raw) >= a.clip).mean()),
  104|               "per_file": {}}
  105|     for k, (path, name) in enumerate(zip(paths, names)):
  106|         m = owner == k
  107|         report["per_file"][name] = {"rows": int(m.sum()), "unsafe": int(y[m].sum()),
  108|                                     "auroc": _auroc(y[m], raw[m]),
  109|                                     "mean_abs_bow_logit": float(np.abs(b[m]).mean())}
  110|         with open(os.path.join(a.out_dir, name), "w") as f:
  111|             for r, v in zip([rows[i] for i in np.flatnonzero(m)], b[m]):
  112|                 r = dict(r)
  113|                 r[a.out_field] = round(float(v), 6)
  114|                 f.write(json.dumps(r, ensure_ascii=False) + "\n")
  115|     print(json.dumps(report, indent=1))
  116|     if a.report:
  117|         with open(a.report, "w") as f:
  118|             json.dump(report, f, indent=1)
  119|     return report
  120| 
  121| 
  122| if __name__ == "__main__":
  123|     main()
```

## `training/v6_build.py`

```python
    1| """v6_build.py: builds the training pools and the dev sets. Runs on Modal only.
    2| 
    3| The pools and dev sets are built from source, de-duplicated (within and across pools),
    4| decontaminated against every reference set (exact normalised match; Jaccard >= 0.8 on 4-word
    5| shingles; MiniLM-L6-v2 cosine >= 0.9), sampled, held-back rows set aside and soft targets
    6| attached. Text is written only to the private Volume; the returned manifest holds counts, ids
    7| and hashes.
    8| 
    9| Pool keys (kept as they are, since they name files and manifest entries):
   10|   P1 ToxicChat (p1_copy.jsonl, written by v6_modal.build_p1), P2 Aegis 2.0, P3 WildGuardMix,
   11|   P4 WildJailbreak, P5 FalseReject. p4_r0.jsonl is the WildJailbreak pool without its benign
   12|   rows, used by recipe variant R0.
   13| Dev sets: twins_prompt = paired prompts generated in-house; dev_pair2 = PHTest (harmless) with
   14| HarmfulQA and CatQA (harmful); dev_holdout_p3/p4/p5 = held-back rows of the WildGuardMix,
   15| WildJailbreak and FalseReject pools; aegis_val and tc_val = the Aegis 2.0 and ToxicChat
   16| validation splits.
   17| """
   18| import hashlib
   19| import json
   20| import math
   21| import os
   22| from collections import Counter, defaultdict
   23| 
   24| import numpy as np
   25| import pandas as pd
   26| 
   27| from decontam import normalize_text, shingles
   28| 
   29| JACCARD, SHINGLE, EMB_T = 0.8, 4, 0.9
   30| EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   31| P3_CAP, P4_PER_TYPE, HOLDOUT = 60_000, 15_000, 0.03   # WildGuardMix cap, WildJailbreak per type, held-back share
   32| # soft targets (benign, unsafe) per pool
   33| SOFT = {"P2": (0.02, 0.98), "P3": (0.05, 0.95), "P4": (0.05, 0.95), "P5": (0.05, 0.95)}
   34| STOP_SHARE = 0.05
   35| TEST_REFS = ("xstest", "jbb", "orbench", "openai_mod", "harmbench_std", "sst", "toxicchat_test",
   36|              "aegis2_test_prompt", "wildguardtest_prompt")
   37| DEV_REFS = ("twins_prompt", "dev_pair2", "aegis_val", "tc_val")
   38| 
   39| 
   40| def sha_file(p):
   41|     h = hashlib.sha256()
   42|     with open(p, "rb") as f:
   43|         for b in iter(lambda: f.read(1 << 20), b""):
   44|             h.update(b)
   45|     return h.hexdigest()
   46| 
   47| 
   48| def write_jsonl(path, rows):
   49|     os.makedirs(os.path.dirname(path), exist_ok=True)
   50|     with open(path, "w", encoding="utf-8") as f:
   51|         for r in rows:
   52|             f.write(json.dumps(r, ensure_ascii=False) + "\n")
   53|     return {"path": path, "rows": len(rows),
   54|             "unsafe": int(sum(int(r["toxicity"]) for r in rows if r.get("toxicity") is not None)),
   55|             "sha256": sha_file(path)}
   56| 
   57| 
   58| def read_jsonl(path):
   59|     with open(path, encoding="utf-8") as f:
   60|         return [json.loads(x) for x in f if x.strip()]
   61| 
   62| 
   63| # ------------------------------------------------------------------------------------------- #
   64| # reference index: exact + shingle-Jaccard (lexical) and MiniLM cosine (embedding), per set     #
   65| # ------------------------------------------------------------------------------------------- #
   66| class RefIndex:
   67|     def __init__(self, refs):
   68|         """refs: {set_name: [text, ...]}"""
   69|         self.names = list(refs)
   70|         self.texts, self.sid = [], []
   71|         for k, n in enumerate(self.names):
   72|             self.texts += [str(t) for t in refs[n]]
   73|             self.sid += [k] * len(refs[n])
   74|         self.sid = np.asarray(self.sid)
   75|         self.exact = defaultdict(set)
   76|         self.inv = defaultdict(list)
   77|         self.size = np.zeros(len(self.texts), dtype=np.int32)
   78|         for i, t in enumerate(self.texts):
   79|             n = normalize_text(t)
   80|             if not n:
   81|                 continue
   82|             self.exact[n].add(int(self.sid[i]))
   83|             sh = shingles(n, SHINGLE)
   84|             self.size[i] = len(sh)
   85|             for g in sh:
   86|                 self.inv[g].append(i)
   87|         self.emb = None
   88| 
   89|     def lexical_one(self, text):
   90|         """{set_id: kind} for one candidate: 'exact' or 'nahash' (Jaccard >= 0.8)."""
   91|         n = normalize_text(text)
   92|         if not n:
   93|             return {}
   94|         hit = {s: "exact" for s in self.exact.get(n, ())}
   95|         sh = shingles(n, SHINGLE)
   96|         c = Counter()
   97|         for g in sh:
   98|             p = self.inv.get(g)
   99|             if p:
  100|                 c.update(p)
  101|         a = len(sh)
  102|         for r, inter in c.items():
  103|             s = int(self.sid[r])
  104|             if s in hit:
  105|                 continue
  106|             if inter / (a + self.size[r] - inter) >= JACCARD:
  107|                 hit[s] = "nahash"
  108|         return hit
  109| 
  110|     def embed_refs(self, model):
  111|         self.emb = _embed(model, self.texts)
  112| 
  113| 
  114| def _embed(model, texts):
  115|     import torch
  116|     e = model.encode([str(t) for t in texts], batch_size=256, convert_to_tensor=True,
  117|                      show_progress_bar=False, normalize_embeddings=True)
  118|     return e.to(torch.float32)
  119| 
  120| 
  121| _IDX = None
  122| 
  123| 
  124| def _lex_chunk(texts):
  125|     return [_IDX.lexical_one(t) for t in texts]
  126| 
  127| 
  128| def flag(index, texts, model, procs=16):
  129|     """Per candidate: {set_name: kind}. Lexical in a process pool, embedding on GPU."""
  130|     global _IDX
  131|     import multiprocessing as mp
  132| 
  133|     import torch
  134|     _IDX = index
  135|     chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
  136|     ctx = mp.get_context("fork")
  137|     with ctx.Pool(procs) as pool:
  138|         lex = [h for part in pool.map(_lex_chunk, chunks) for h in part]
  139|     hits = [{index.names[s]: k for s, k in h.items()} for h in lex]
  140|     if index.emb is None:
  141|         index.embed_refs(model)
  142|     ce = _embed(model, texts)
  143|     sid = torch.as_tensor(index.sid, device=index.emb.device)
  144|     for st in range(0, ce.shape[0], 4096):
  145|         sims = ce[st:st + 4096] @ index.emb.T          # [b, R]
  146|         for k, name in enumerate(index.names):
  147|             m = sid == k
  148|             if not bool(m.any()):
  149|                 continue
  150|             best = sims[:, m].max(dim=1).values.cpu().numpy()
  151|             for j in np.flatnonzero(best >= EMB_T):
  152|                 hits[st + j].setdefault(name, "embedding")
  153|     return hits
  154| 
  155| 
  156| def summarise_hits(hits, n):
  157|     per = defaultdict(Counter)
  158|     for h in hits:
  159|         for s, k in h.items():
  160|             per[s][k] += 1
  161|     flagged = sum(1 for h in hits if h)
  162|     return {"n": n, "flagged_any": flagged,
  163|             "per_ref": {s: {"flagged": sum(c.values()), "share": round(sum(c.values()) / max(n, 1), 5),
  164|                             "by_kind": dict(c)} for s, c in sorted(per.items())}}
  165| 
  166| 
  167| # ------------------------------------------------------------------------------------------- #
  168| # de-duplication within and across pools                                                      #
  169| # ------------------------------------------------------------------------------------------- #
  170| def dedup_within(df):
  171|     """Normalise; drop empty; a text with both labels is dropped entirely; keep first copy."""
  172|     df = df.copy()
  173|     df["_n"] = [normalize_text(t) for t in df["user_input"]]
  174|     n0 = len(df)
  175|     df = df[df["_n"] != ""]
  176|     nl = df.groupby("_n")["toxicity"].nunique()
  177|     conflict = set(nl[nl > 1].index)
  178|     df_c = df[~df["_n"].isin(conflict)]
  179|     out = df_c.drop_duplicates("_n", keep="first")
  180|     return out.reset_index(drop=True), {"in": n0, "empty": int(n0 - len(df)),
  181|                                         "conflict_texts": len(conflict),
  182|                                         "conflict_rows": int(len(df) - len(df_c)),
  183|                                         "dup_rows": int(len(df_c) - len(out)), "out": len(out)}
  184| 
  185| 
  186| def dedup_across(p1_norm_label, pools):
  187|     """pools: ordered {name: df with _n}. The ToxicChat pool (P1) wins any disputed label; among
  188|     the other pools (P2-P5) a disputed label drops every copy, otherwise the first occurrence (pool
  189|     order) is kept."""
  190|     labels = defaultdict(set)
  191|     first = {}
  192|     for name, df in pools.items():
  193|         for n, y in zip(df["_n"], df["toxicity"]):
  194|             labels[n].add(int(y))
  195|             first.setdefault(n, name)
  196|     out, rep = {}, {}
  197|     for name, df in pools.items():
  198|         in_p1 = df["_n"].isin(p1_norm_label.keys())
  199|         conflict = df["_n"].map(lambda n: len(labels[n]) > 1)
  200|         later = df["_n"].map(lambda n: first[n] != name)
  201|         keep = ~in_p1 & ~conflict & ~later
  202|         rep[name] = {"in": len(df), "dup_or_conflict_with_P1": int(in_p1.sum()),
  203|                      "conflict_P2_P5": int((~in_p1 & conflict).sum()),
  204|                      "dup_of_earlier_pool": int((~in_p1 & ~conflict & later).sum()),
  205|                      "out": int(keep.sum())}
  206|         out[name] = df[keep].reset_index(drop=True)
  207|     return out, rep
  208| 
  209| 
  210| # ------------------------------------------------------------------------------------------- #
  211| # sampling caps, held-back rows and soft targets                                              #
  212| # ------------------------------------------------------------------------------------------- #
  213| def stratified_cap(df, strata_cols, cap, rng):
  214|     """Proportional stratified sample of `cap` rows (largest remainder); all rows if fewer."""
  215|     if len(df) <= cap:
  216|         return df.reset_index(drop=True), {"cap_applied": False}
  217|     groups = sorted(df.groupby(strata_cols).groups.items(), key=lambda kv: str(kv[0]))
  218|     sizes = np.array([len(ix) for _, ix in groups], float)
  219|     raw = cap * sizes / sizes.sum()
  220|     take = np.floor(raw).astype(int)
  221|     for i in np.argsort(-(raw - take), kind="stable")[: cap - take.sum()]:
  222|         take[i] += 1
  223|     parts = []
  224|     for (key, ix), k in zip(groups, take):
  225|         ix = np.asarray(sorted(ix))
  226|         parts.append(df.loc[np.sort(rng.choice(ix, size=k, replace=False))])
  227|     return pd.concat(parts).reset_index(drop=True), {"cap_applied": True, "strata": {
  228|         str(key): int(k) for (key, _), k in zip(groups, take)}}
  229| 
  230| 
  231| def holdout(df, strata_cols, rng, share=HOLDOUT):
  232|     is_h = np.zeros(len(df), bool)
  233|     for key, ix in sorted(df.groupby(strata_cols).groups.items(), key=lambda kv: str(kv[0])):
  234|         ix = np.asarray(sorted(ix))
  235|         k = int(round(share * len(ix)))
  236|         is_h[rng.permutation(ix)[:k]] = True
  237|     return df[~is_h].reset_index(drop=True), df[is_h].reset_index(drop=True)
  238| 
  239| 
  240| def to_rows(df, pool, soft=None, hard_soft=False):
  241|     rows = []
  242|     for r in df.itertuples(index=False):
  243|         y = int(r.toxicity)
  244|         d = {"row_id": r.row_id, "user_input": r.user_input, "toxicity": y, "pool": pool,
  245|              "type": getattr(r, "type", pool)}
  246|         if soft is not None:
  247|             d["toxicity_soft"] = soft[1] if y else soft[0]
  248|         elif hard_soft:
  249|             d["toxicity_soft"] = float(y)
  250|         rows.append(d)
  251|     return rows
  252| 
  253| 
  254| # ------------------------------------------------------------------------------------------- #
  255| # orchestration                                                                                #
  256| # ------------------------------------------------------------------------------------------- #
  257| def load_references_test(vol):
  258|     import v6_data as vd
  259|     from tc_common import INPUT_FIELD, load_test
  260|     j = lambda n: json.load(open(f"{vol}/sets/{n}.json"))  # noqa: E731
  261|     refs, meta = {}, {}
  262|     refs["xstest"] = j("xstest_safe") + j("xstest_unsafe")
  263|     refs["orbench"] = j("orbench_hard") + j("orbench_toxic")
  264|     jb, rev = vd.load_hf_any(vd.JBB, "behaviors")
  265|     refs["jbb"] = list(jb["harmful"]["Goal"]) + list(jb["benign"]["Goal"])
  266|     meta["jbb_revision"] = rev
  267|     om, sha = vd.load_openai_mod()
  268|     refs["openai_mod"] = [r["prompt"] for r in om]
  269|     meta["openai_mod_commit"] = sha
  270|     refs["harmbench_std"] = vd.load_harmbench_std()
  271|     refs["sst"] = vd.load_sst()
  272|     refs["toxicchat_test"] = [str(t) for t in load_test()[INPUT_FIELD]]
  273|     at = vd.load_aegis("test.json")
  274|     refs["aegis2_test_prompt"] = [p for p in at["prompt"] if p != "REDACTED"]
  275|     wt, rev, _ = vd.load_wgm("wildguardtest")
  276|     refs["wildguardtest_prompt"] = list(wt.loc[wt["prompt_harm_label"].notna(), "prompt"])
  277|     meta["wildguardmix_revision"] = rev
  278|     meta["n"] = {k: len(v) for k, v in refs.items()}
  279|     return refs, meta
  280| 
  281| 
  282| def aegis_prompts(df):
  283|     df = df[(df["prompt"] != "REDACTED") & df["prompt_label"].isin(["safe", "unsafe"])]
  284|     df = df[df["prompt"].astype(str).str.strip().str.len() >= 3]
  285|     return pd.DataFrame({"row_id": "aegis:" + df["id"].astype(str), "user_input": df["prompt"].astype(str),
  286|                          "toxicity": (df["prompt_label"] == "unsafe").astype(int), "type": "aegis"})
  287| 
  288| 
  289| def build_all(vol, out_root, procs=16):
  290|     import v6_data as vd
  291|     from sentence_transformers import SentenceTransformer
  292|     model = SentenceTransformer(EMB_MODEL, device="cuda")
  293|     man = {"params": {"jaccard": JACCARD, "shingle": SHINGLE, "emb_threshold": EMB_T,
  294|                       "emb_model": EMB_MODEL, "p3_cap": P3_CAP, "p4_per_type": P4_PER_TYPE,
  295|                       "holdout": HOLDOUT, "rng": "numpy.random.default_rng(13), one stream: "
  296|                       "P3 cap, P4 cap (types in sorted order), holdouts P3, P4, P5"}}
  297| 
  298|     # ---- test references ---------------------------------------------------------------- #
  299|     test_refs, man["references"] = load_references_test(vol)
  300|     idx_test = RefIndex(test_refs)
  301| 
  302|     # ---- dev sets ------------------------------------------------------------------------- #
  303|     calm = read_jsonl(f"{vol}/data/calm_train.jsonl") + read_jsonl(f"{vol}/data/calm_dev.jsonl")
  304|     tw = [r for r in calm if r["axis"] == "prompt"]
  305|     h = flag(idx_test, [r["user_input"] for r in tw], model, procs)
  306|     bad_pairs = {r["pair_id"] for r, x in zip(tw, h) if x}
  307|     man["twins_decontam"] = summarise_hits(h, len(tw)) | {"pairs_dropped": len(bad_pairs)}
  308|     twins = [{"row_id": f"calm:{r['pair_id']}:{r['toxicity']}", "user_input": r["user_input"],
  309|               "toxicity": int(r["toxicity"]), "pair_id": r["pair_id"], "category": r["category"]}
  310|              for r in tw if r["pair_id"] not in bad_pairs]
  311| 
  312|     ph, ph_rev = vd.load_hf_any(vd.PHTEST)
  313|     ph = ph["train"]
  314|     man["phtest_harmfulness_counts"] = {str(k): int(v) for k, v in ph["Harmfulness"].value_counts().items()}
  315|     ph = ph[ph["Harmfulness"].astype(str).str.lower() == "harmless"]
  316|     assert len(ph) > 0, "PHTest: no rows labelled harmless"
  317|     hq, hq_rev = vd.load_hf_any(vd.HQA)
  318|     cq, cq_rev = vd.load_hf_any(vd.CATQA)
  319|     pair2 = pd.concat([
  320|         pd.DataFrame({"row_id": "phtest:" + ph["ID"].astype(str), "user_input": ph["Request"].astype(str),
  321|                       "toxicity": 0, "type": "phtest_harmless"}),
  322|         pd.DataFrame({"row_id": "harmfulqa:" + hq["train"]["id"].astype(str),
  323|                       "user_input": hq["train"]["question"].astype(str), "toxicity": 1, "type": "harmfulqa"}),
  324|         pd.DataFrame({"row_id": [f"catqa_en:{i}" for i in range(len(cq["en"]))],
  325|                       "user_input": cq["en"]["Question"].astype(str), "toxicity": 1, "type": "catqa_en"}),
  326|     ], ignore_index=True)
  327|     pair2, man["dev_pair2_dedup"] = dedup_within(pair2)
  328|     h = flag(idx_test, list(pair2["user_input"]), model, procs)
  329|     man["dev_pair2_decontam"] = summarise_hits(h, len(pair2))
  330|     pair2 = pair2[[not x for x in h]].reset_index(drop=True)
  331|     man["dev_pair2_sources"] = {"phtest": ph_rev, "harmfulqa": hq_rev, "catqa": cq_rev,
  332|                                 "columns": {"phtest": "Request (Harmfulness == harmless)",
  333|                                             "harmfulqa": "question", "catqa": "Question (split en)"},
  334|                                 "counts": {k: int(v) for k, v in pair2["type"].value_counts().items()}}
  335| 
  336|     av = aegis_prompts(vd.load_aegis("validation.json"))
  337|     h = flag(idx_test, list(av["user_input"]), model, procs)
  338|     man["aegis_val_vs_test_refs_report_only"] = summarise_hits(h, len(av))
  339| 
  340|     tc_val = read_jsonl(f"{out_root}/dev/tc_val.jsonl")
  341|     p1 = read_jsonl(f"{out_root}/pools/p1_copy.jsonl")
  342| 
  343|     dev_refs = {"twins_prompt": [r["user_input"] for r in twins], "dev_pair2": list(pair2["user_input"]),
  344|                 "aegis_val": list(av["user_input"]), "tc_val": [r["user_input"] for r in tc_val]}
  345|     all_refs = dict(test_refs) | dev_refs
  346|     idx_all = RefIndex(all_refs)
  347|     os.makedirs(f"{out_root}/refs", exist_ok=True)
  348|     with open(f"{out_root}/refs/refs.json", "w", encoding="utf-8") as f:
  349|         json.dump(all_refs, f, ensure_ascii=False)
  350| 
  351|     # ---- pools P2-P5 (Aegis 2.0, WildGuardMix, WildJailbreak, FalseReject): load, dedup ------ #
  352|     p2 = aegis_prompts(vd.load_aegis("train.json"))
  353|     g, wgm_rev, _ = vd.load_wgm("wildguardtrain")
  354|     g = g[g["prompt_harm_label"].isin(["harmful", "unharmful"])]
  355|     p3 = pd.DataFrame({"row_id": [f"wgm:{i}" for i in g.index], "user_input": g["prompt"].astype(str),
  356|                        "toxicity": (g["prompt_harm_label"] == "harmful").astype(int).values,
  357|                        "type": np.where(g["adversarial"].astype(bool), "adversarial", "vanilla")})
  358|     w, wjb_rev = vd.load_wjb()
  359|     txt = np.where(w["data_type"].str.startswith("adversarial"), w["adversarial"], w["vanilla"])
  360|     p4 = pd.DataFrame({"row_id": [f"wjb:{i}" for i in w.index], "user_input": txt,
  361|                        "toxicity": w["data_type"].str.endswith("_harmful").astype(int).values,
  362|                        "type": w["data_type"].values})
  363|     p4 = p4[p4["user_input"].str.len() > 0]
  364|     fr = read_jsonl(f"{vol}/v6/src/falsereject_pool.clean.jsonl")
  365|     p5 = pd.DataFrame({"row_id": [f"falsereject:{i}" for i in range(len(fr))],
  366|                        "user_input": [r["user_input"] for r in fr],
  367|                        "toxicity": [int(r["toxicity"]) for r in fr], "type": "falsereject"})
  368|     man["sources"] = {"P2": {"hub": vd.AEGIS[0], "revision": vd.AEGIS[1], "file": "train.json",
  369|                              "rows_after_rules": int(len(p2))},
  370|                       "P3": {"hub": vd.WGM, "revision": wgm_rev, "config": "wildguardtrain",
  371|                              "rows_labelled": int(len(p3))},
  372|                       "P4": {"hub": vd.WJB, "revision": wjb_rev, "file": "train/train.tsv",
  373|                              "rows_raw": int(len(w)), "rows_nonempty_text": int(len(p4))},
  374|                       "P5": {"file": "gen_data/falsereject_pool.clean.jsonl", "rows": int(len(p5)),
  375|                              "sha256": sha_file(f"{vol}/v6/src/falsereject_pool.clean.jsonl")}}
  376|     pools, man["dedup_within"] = {}, {}
  377|     for name, df in (("P2", p2), ("P3", p3), ("P4", p4), ("P5", p5)):
  378|         pools[name], man["dedup_within"][name] = dedup_within(df)
  379|     p1_norm = {normalize_text(r["user_input"]): r["toxicity"] for r in p1}
  380|     pools, man["dedup_across"] = dedup_across(p1_norm, pools)
  381| 
  382|     # ---- decontaminate -------------------------------------------------------------------- #
  383|     man["decontam"] = {}
  384|     h = flag(idx_all, [r["user_input"] for r in p1], model, procs)
  385|     h = [{k: v for k, v in x.items() if k not in ("toxicchat_test", "tc_val")} for x in h]
  386|     man["decontam"]["P1_report_only"] = summarise_hits(h, len(p1))
  387|     stops = []
  388|     for name in ("P2", "P3", "P4", "P5"):
  389|         df = pools[name]
  390|         h = flag(idx_all, list(df["user_input"]), model, procs)
  391|         s = summarise_hits(h, len(df))
  392|         man["decontam"][name] = s
  393|         stops += [(name, r, v["share"]) for r, v in s["per_ref"].items() if v["share"] > STOP_SHARE]
  394|         pools[name] = df[[not x for x in h]].reset_index(drop=True)
  395|     if stops:
  396|         # Stop when more than 5% of a pool matches one reference set.
  397|         man["STOP"] = {"rule": "more than 5% of a pool flagged against one reference set",
  398|                        "hits": [{"pool": p, "ref": r, "share": s} for p, r, s in stops]}
  399|         return man
  400| 
  401|     # ---- sample, hold out, soft targets ----------------------------------------------------- #
  402|     rng = np.random.default_rng(13)
  403|     pools["P3"], man["p3_cap"] = stratified_cap(pools["P3"], ["type", "toxicity"], P3_CAP, rng)
  404|     parts, unused = [], []
  405|     for t in sorted(pools["P4"]["type"].unique()):
  406|         d = pools["P4"][pools["P4"]["type"] == t].reset_index(drop=True)
  407|         pick = np.sort(rng.choice(len(d), size=min(P4_PER_TYPE, len(d)), replace=False))
  408|         mask = np.zeros(len(d), bool)
  409|         mask[pick] = True
  410|         parts.append(d[mask])
  411|         unused.append(d[~mask])
  412|     pools["P4"] = pd.concat(parts).reset_index(drop=True)
  413|     p4_unused = pd.concat(unused).reset_index(drop=True)
  414|     hold = {}
  415|     for name in ("P3", "P4", "P5"):
  416|         pools[name], hold[name] = holdout(pools[name], ["type", "toxicity"], rng)
  417| 
  418|     files = {}
  419|     pdir, ddir = f"{out_root}/pools", f"{out_root}/dev"
  420|     for name in ("P2", "P3", "P4", "P5"):
  421|         files[name] = write_jsonl(f"{pdir}/{name.lower()}.jsonl", to_rows(pools[name], name, SOFT[name]))
  422|     r0p4 = pools["P4"][pools["P4"]["type"].isin(["vanilla_harmful", "adversarial_harmful"])]
  423|     files["P4_R0"] = write_jsonl(f"{pdir}/p4_r0.jsonl", to_rows(r0p4, "P4", SOFT["P4"]))
  424|     for name in ("P3", "P4", "P5"):
  425|         files[f"dev_holdout_{name.lower()}"] = write_jsonl(f"{ddir}/dev_holdout_{name.lower()}.jsonl",
  426|                                                            to_rows(hold[name], name))
  427|     files["twins_prompt"] = write_jsonl(f"{ddir}/twins_prompt.jsonl", twins)
  428|     files["dev_pair2"] = write_jsonl(f"{ddir}/dev_pair2.jsonl", to_rows(pair2, "dev_pair2"))
  429|     files["aegis_val"] = write_jsonl(f"{ddir}/aegis_val.jsonl", to_rows(av, "aegis_val"))
  430|     files["p4_vanilla_unused_for_r3"] = write_jsonl(
  431|         f"{out_root}/r3/p4_vanilla_unused.jsonl",
  432|         to_rows(p4_unused[p4_unused["type"].str.startswith("vanilla")], "WJB_EXTRA", SOFT["P4"]))
  433|     man["files"] = files
  434|     man["type_counts"] = {n: {f"{t}|{y}": int(c) for (t, y), c in
  435|                               pools[n].groupby(["type", "toxicity"]).size().items()} for n in pools}
  436|     n_p1 = len(p1)
  437|     man["N_R1"] = 3 * n_p1 + sum(files[k]["rows"] for k in ("P2", "P3", "P4", "P5"))
  438|     man["N_R0"] = 3 * n_p1 + files["P2"]["rows"] + files["P3"]["rows"] + files["P4_R0"]["rows"]
  439|     man["unsafe_R1"] = 3 * sum(r["toxicity"] for r in p1) + sum(
  440|         files[k]["unsafe"] for k in ("P2", "P3", "P4", "P5"))
  441|     return man
  442| 
  443| 
  444| def reverse_audit(out_root, procs=16):
  445|     """Reverse audit: every reference row against the base recipe's (R1) final training files (the
  446|     four pool files plus the ToxicChat copy). The ToxicChat copy is not checked against the
  447|     ToxicChat references, which come from the same dataset by design."""
  448|     from sentence_transformers import SentenceTransformer
  449|     model = SentenceTransformer(EMB_MODEL, device="cuda")
  450|     refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
  451|     pools = {n: [r["user_input"] for r in read_jsonl(f"{out_root}/pools/{n}.jsonl")]
  452|              for n in ("p2", "p3", "p4", "p5")}
  453|     p1 = [r["user_input"] for r in read_jsonl(f"{out_root}/pools/p1_copy.jsonl")]
  454|     idx = RefIndex(pools | {"p1_copy": p1})
  455|     out = {}
  456|     for name, texts in refs.items():
  457|         h = flag(idx, texts, model, procs)
  458|         if name in ("toxicchat_test", "tc_val"):
  459|             h = [{k: v for k, v in x.items() if k != "p1_copy"} for x in h]
  460|         out[name] = summarise_hits(h, len(texts))
  461|     out["total_flagged"] = int(sum(v["flagged_any"] for v in out.values() if isinstance(v, dict)))
  462|     return out
  463| 
  464| 
  465| # ------------------------------------------------------------------------------------------- #
  466| # extra rows for R3 (base recipe + teacher labels), collected best effort                     #
  467| # ------------------------------------------------------------------------------------------- #
  468| R3_WJB_PER_TYPE, R3_HARMAUG, R3_WILDCHAT = 15_000, 30_000, 40_000
  469| 
  470| 
  471| def _wildchat_first_turns(rng, want, max_shards=None):
  472|     """English first user turns, 20-2000 chars, at most 3 per hashed_ip. Only the first turn's
  473|     content, `language` and `hashed_ip` are accessed; no assistant text or moderation field."""
  474|     import pyarrow.parquet as pq
  475|     from huggingface_hub import HfApi, hf_hub_download
  476|     import v6_data as vd
  477|     rev = vd.revision(vd.WILDCHAT)
  478|     files = sorted(f for f in HfApi().list_repo_files(vd.WILDCHAT, repo_type="dataset", revision=rev)
  479|                    if f.endswith(".parquet"))
  480|     rows, per_ip, used = [], Counter(), []
  481|     for f in files[:max_shards]:
  482|         t = pq.read_table(hf_hub_download(vd.WILDCHAT, f, repo_type="dataset", revision=rev),
  483|                           columns=["conversation", "language", "hashed_ip", "conversation_hash"])
  484|         used.append(f)
  485|         lang, ip, conv, ch = (t.column(c).to_pylist() for c in
  486|                               ("language", "hashed_ip", "conversation", "conversation_hash"))
  487|         order = rng.permutation(len(lang))
  488|         for i in order:
  489|             if lang[i] != "English" or not conv[i]:
  490|                 continue
  491|             first = conv[i][0]
  492|             if first.get("role") != "user":
  493|                 continue
  494|             txt = str(first.get("content") or "")
  495|             if not (20 <= len(txt) <= 2000) or per_ip[ip[i]] >= 3:
  496|                 continue
  497|             per_ip[ip[i]] += 1
  498|             rows.append({"row_id": f"wildchat:{ch[i]}", "user_input": txt})
  499|         del t, conv
  500|         if len(rows) >= want:
  501|             break
  502|     return rows, {"revision": rev, "shards_read": used, "shards_total": len(files),
  503|                   "eligible_rows_collected": len(rows)}
  504| 
  505| 
  506| def p1_matched_heldout(out_root, procs=8):
  507|     """Sensitivity check: the held-out reference items (orbench, jbb, sst) that match the ToxicChat
  508|     pool (p1_copy), as (card set, row index, sha256 of the item text). No text is returned."""
  509|     from sentence_transformers import SentenceTransformer
  510|     model = SentenceTransformer(EMB_MODEL, device="cuda")
  511|     refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
  512|     p1 = [r["user_input"] for r in read_jsonl(f"{out_root}/pools/p1_copy.jsonl")]
  513|     idx = RefIndex({"p1_copy": p1})
  514|     # refs.json order (load_references_test): orbench = hard (1,319) + toxic (655);
  515|     # jbb = harmful (100) + benign (100); sst = file order
  516|     split = {"orbench": [("orbench_hard", 1319), ("orbench_toxic", 655)],
  517|              "jbb": [("jbb_harmful", 100), ("jbb_benign", 100)], "sst": [("sst", len(refs["sst"]))]}
  518|     out = []
  519|     for name, parts in split.items():
  520|         texts = refs[name]
  521|         assert len(texts) == sum(n for _, n in parts), name
  522|         h = flag(idx, texts, model, procs)
  523|         for k in [i for i, x in enumerate(h) if x]:
  524|             off = 0
  525|             for card_set, n in parts:
  526|                 if k < off + n:
  527|                     out.append({"set": card_set, "i": k - off, "kind": h[k]["p1_copy"],
  528|                                 "text_sha256": hashlib.sha256(texts[k].encode("utf-8")).hexdigest()})
  529|                     break
  530|                 off += n
  531|     return {"items": out, "n": len(out),
  532|             "note": "row index i is the item's position in the pinned set file order used by both "
  533|                     "harnesses; verify by text_sha256 against the card's source before excluding"}
  534| 
  535| 
  536| def filter_aegis_val(out_root, procs=16):
  537|     """Drop aegis_val rows that match any test set, because aegis_val is used to set the frozen
  538|     thresholds. The pre-filter file is kept as aegis_val_prefilter.jsonl."""
  539|     from sentence_transformers import SentenceTransformer
  540|     model = SentenceTransformer(EMB_MODEL, device="cuda")
  541|     refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
  542|     idx = RefIndex({k: refs[k] for k in TEST_REFS})
  543|     src = f"{out_root}/dev/aegis_val.jsonl"
  544|     pre = f"{out_root}/dev/aegis_val_prefilter.jsonl"
  545|     if not os.path.exists(pre):
  546|         os.rename(src, pre)
  547|     rows = read_jsonl(pre)
  548|     h = flag(idx, [r["user_input"] for r in rows], model, procs)
  549|     kept = [r for r, x in zip(rows, h) if not x]
  550|     out = {"prefilter": {"rows": len(rows), "sha256": sha_file(pre)},
  551|            "decontam": summarise_hits(h, len(rows)),
  552|            "aegis_val": write_jsonl(src, kept)}
  553|     out["aegis_val"]["safe"] = out["aegis_val"]["rows"] - out["aegis_val"]["unsafe"]
  554|     return out
  555| 
  556| 
  557| def build_r3_training(out_root, targets_path):
  558|     """R3 training files from the committee's targets (v6_committee_targets.py --out): the
  559|     ToxicChat and Aegis 2.0 files are unchanged; WildGuardMix, WildJailbreak and FalseReject rows
  560|     take the committee target; the extra rows go to r3_extra.jsonl."""
  561|     tg = {r["row_id"]: r for r in read_jsonl(targets_path)}
  562|     out, rep = {}, {"targets_rows": len(tg), "targets_sha256": sha_file(targets_path)}
  563|     d = f"{out_root}/r3/pools"
  564|     for n in ("p1_copy", "p2"):
  565|         rows = read_jsonl(f"{out_root}/pools/{n}.jsonl")
  566|         out[n] = write_jsonl(f"{d}/{n}.jsonl", rows)
  567|     for n in ("p3", "p4", "p5"):
  568|         rows, changed, missing = [], 0, 0
  569|         for r in read_jsonl(f"{out_root}/pools/{n}.jsonl"):
  570|             t = tg.get(r["row_id"])
  571|             if t is None:
  572|                 missing += 1
  573|             else:
  574|                 changed += int(float(t["toxicity_soft"]) != float(r["toxicity_soft"]))
  575|                 r = dict(r, toxicity=int(t["toxicity"]), toxicity_soft=float(t["toxicity_soft"]))
  576|             rows.append(r)
  577|         out[n] = write_jsonl(f"{d}/{n}.jsonl", rows) | {"target_changed": changed, "not_in_targets": missing}
  578|     extra = [dict(row_id=t["row_id"], user_input=t["user_input"], toxicity=int(t["toxicity"]),
  579|                   toxicity_soft=float(t["toxicity_soft"]), pool=t["pool"], type=t.get("type", t["pool"]))
  580|              for t in tg.values() if t["pool"] in ("WJB_EXTRA", "HARMAUG", "WILDCHAT")]
  581|     out["r3_extra"] = write_jsonl(f"{d}/r3_extra.jsonl", extra)
  582|     out["r3_extra"]["per_pool"] = dict(Counter(r["pool"] for r in extra))
  583|     assert all(r["toxicity_soft"] is not None for r in extra)
  584|     rep["files"] = out
  585|     rep["N_R3"] = 2 * out["p1_copy"]["rows"] + sum(out[k]["rows"] for k in out)
  586|     return rep
  587| 
  588| 
  589| def build_r3(vol, out_root, procs=16):
  590|     import v6_data as vd
  591|     from sentence_transformers import SentenceTransformer
  592|     model = SentenceTransformer(EMB_MODEL, device="cuda")
  593|     rng = np.random.default_rng(13)
  594|     man = {"rng": "default_rng(13): WJB_EXTRA per vanilla type (sorted), HarmAug, WildChat"}
  595|     all_refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
  596|     idx = RefIndex(all_refs)
  597|     base = {n: read_jsonl(f"{out_root}/pools/{n}.jsonl") for n in ("p1_copy", "p2", "p3", "p4", "p5")}
  598|     held = [r for n in ("p3", "p4", "p5") for r in read_jsonl(f"{out_root}/dev/dev_holdout_{n}.jsonl")]
  599|     seen = {normalize_text(r["user_input"]) for rs in base.values() for r in rs}
  600|     seen |= {normalize_text(r["user_input"]) for r in held}
  601| 
  602|     # extra vanilla WildJailbreak rows: already de-duplicated and decontaminated with the WildJailbreak pool
  603|     un = pd.DataFrame(read_jsonl(f"{out_root}/r3/p4_vanilla_unused.jsonl"))
  604|     parts = []
  605|     for t in sorted(un["type"].unique()):
  606|         d = un[un["type"] == t].reset_index(drop=True)
  607|         parts.append(d.iloc[np.sort(rng.choice(len(d), size=min(R3_WJB_PER_TYPE, len(d)), replace=False))])
  608|     wjb = pd.concat(parts).reset_index(drop=True)
  609|     man["WJB_EXTRA"] = {"available": int(len(un)), "taken": {k: int(v) for k, v in wjb["type"].value_counts().items()}}
  610| 
  611|     def clean_unlabelled(rows, name, cap):
  612|         df = pd.DataFrame(rows)
  613|         df["toxicity"] = 0   # placeholder for dedup_within only; never written
  614|         df, dd = dedup_within(df)
  615|         n0 = len(df)
  616|         df = df[~df["_n"].isin(seen)].reset_index(drop=True)
  617|         cross = n0 - len(df)
  618|         h = flag(idx, list(df["user_input"]), model, procs)
  619|         s = summarise_hits(h, len(df))
  620|         df = df[[not x for x in h]].reset_index(drop=True)
  621|         if len(df) > cap:
  622|             df = df.iloc[np.sort(rng.choice(len(df), size=cap, replace=False))].reset_index(drop=True)
  623|         seen.update(df["_n"])
  624|         man[name] = {"dedup_within": dd, "dup_of_r1_or_holdout": int(cross), "decontam": s,
  625|                      "kept": int(len(df))}
  626|         return [{"row_id": r.row_id, "user_input": r.user_input, "toxicity": None,
  627|                  "toxicity_soft": None, "pool": name, "type": name} for r in df.itertuples(index=False)]
  628| 
  629|     from huggingface_hub import HfApi, hf_hub_download
  630|     ha_rev = vd.revision(vd.HARMAUG)
  631|     ha_files = [f for f in HfApi().list_repo_files(vd.HARMAUG, repo_type="dataset", revision=ha_rev)
  632|                 if f.endswith(".parquet")]
  633|     ha = pd.concat([pd.read_parquet(hf_hub_download(vd.HARMAUG, f, repo_type="dataset", revision=ha_rev))
  634|                     for f in ha_files], ignore_index=True)
  635|     man["HARMAUG_source"] = {"revision": ha_rev, "columns": list(map(str, ha.columns)), "rows": int(len(ha))}
  636|     col = next((c for c in ("instruction", "prompt", "question", "query") if c in ha.columns), None)
  637|     if col is None:
  638|         man["STOP_HARMAUG"] = "no instruction-like column; HarmAug skipped"
  639|         harmaug = []
  640|     else:
  641|         man["HARMAUG_source"]["text_column"] = col
  642|         txt = ha[col].astype(str)
  643|         ha_rows = [{"row_id": f"harmaug:{i}", "user_input": t} for i, t in zip(ha.index, txt)
  644|                    if 3 <= len(t.strip())]
  645|         sub = rng.permutation(len(ha_rows))[: 2 * R3_HARMAUG]
  646|         harmaug = clean_unlabelled([ha_rows[i] for i in sorted(sub)], "HARMAUG", R3_HARMAUG)
  647| 
  648|     wc, man["WILDCHAT_source"] = _wildchat_first_turns(rng, int(1.5 * R3_WILDCHAT))
  649|     wildchat = clean_unlabelled(wc, "WILDCHAT", R3_WILDCHAT)
  650| 
  651|     rows = []
  652|     for n, pool in (("p1_copy", "P1"), ("p2", "P2"), ("p3", "P3"), ("p4", "P4"), ("p5", "P5")):
  653|         for r in base[n]:
  654|             rows.append({k: r.get(k) for k in ("row_id", "user_input", "toxicity", "toxicity_soft", "type")}
  655|                         | {"pool": pool})
  656|     for r in wjb.to_dict("records"):
  657|         rows.append({"row_id": r["row_id"], "user_input": r["user_input"], "toxicity": int(r["toxicity"]),
  658|                      "toxicity_soft": float(r["toxicity_soft"]), "type": r["type"], "pool": "WJB_EXTRA"})
  659|     rows += harmaug + wildchat
  660|     ids = [r["row_id"] for r in rows]
  661|     assert len(ids) == len(set(ids)), "duplicate row_id in R3 rows"
  662|     man["r3_rows"] = write_jsonl(f"{out_root}/r3/r3_rows.jsonl", rows)
  663|     man["r3_rows"]["per_pool"] = dict(Counter(r["pool"] for r in rows))
  664|     return man
```
