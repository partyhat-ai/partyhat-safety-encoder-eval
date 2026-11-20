# partyhat-safety-encoder-eval: source and docs, part 7 of 7

Generated from commit `ca598a68c892` (2026-11-20) of github.com/partyhat-ai/partyhat-safety-encoder-eval by `scripts/llms_digest.py`, which `.github/workflows/llms-digest.yml` reruns on every push to `main`, so this text matches the code at that commit.
Files `training/v6_committee_targets.py` to `training/v6_select.py`. Lines are numbered as in the repository.

## `training/v6_committee_targets.py`

```python
    1| #!/usr/bin/env python3
    2| """v6_committee_targets.py: turns the teacher committee's verdicts into training labels for recipe
    3| variant R3 (base recipe + teacher labels). This is the CPU half; the Modal half scores the rows.
    4| 
    5| Turns committee margins into training targets. It does no model inference: the committee scorer
    6| (r3_committee/r3_committee_modal.py) scores rows on Modal and hands this script one margin file per member.
    7| 
    8| Inputs (JSONL):
    9|   --rows   R3's rows: row_id, user_input, pool, toxicity (0/1, or null when the row has no source
   10|            label), toxicity_soft (the pool's soft target, or null when there is no source label)
   11|   --member NAME=path   repeated; rows of {row_id, margin}. The margin is the verdict-token margin,
   12|            unsafe minus safe (for Qwen3Guard: Unsafe against Safe-or-Controversial). Rows the member
   13|            did not score are simply absent.
   14|   --calib  NAME=path   repeated; rows of {margin, y} for that member on tc_val + aegis_val.
   15| 
   16| Rules:
   17|   * each member is Platt-scaled on its calibration rows; the committee log-odds is the mean of the
   18|     calibrated log-odds over the members that scored the row; p_c is its sigmoid;
   19|   * a row needs at least --min_members (default 2) members to count as scored;
   20|   * source label kept, unless p_c >= 0.9 on a row labelled safe or p_c <= 0.1 on a row labelled
   21|     unsafe: then the label is disputed and the target is 0.5;
   22|   * no source label: target = p_c clipped to [0.02, 0.98], toxicity = 1 when p_c >= 0.5;
   23|   * a source-labelled row that was not scored keeps its target; an unlabelled row that was not
   24|     scored is dropped (there is nothing to train it on).
   25| 
   26| Output: --out (JSONL: every kept row with toxicity, toxicity_soft, p_c, target_rule) and
   27| --report (JSON: counts per pool, Platt parameters). Prints the report; never prints text.
   28| A disputed label is recorded as target_rule "clash" and counted in clash_rows; those names are
   29| kept because the stored files use them.
   30| A member whose Platt slope is not positive stops the run: its margin has the wrong sign.
   31| """
   32| import argparse
   33| import json
   34| from collections import Counter, defaultdict
   35| 
   36| import numpy as np
   37| 
   38| 
   39| def read_jsonl(path):
   40|     with open(path) as f:
   41|         return [json.loads(line) for line in f if line.strip()]
   42| 
   43| 
   44| def platt(margins, y):
   45|     """Return (a, b) with calibrated log-odds = a * margin + b."""
   46|     from sklearn.linear_model import LogisticRegression
   47|     m, y = np.asarray(margins, float).reshape(-1, 1), np.asarray(y, int)
   48|     if not (np.isfinite(m).all() and 0 < y.sum() < len(y)):
   49|         raise SystemExit("calibration rows need finite margins and both classes")
   50|     # Fit on the standardised margin, then convert back: the fit is then the same whatever scale a
   51|     # member's logits live on.
   52|     mu, sd = float(m.mean()), float(m.std()) or 1.0
   53|     clf = LogisticRegression(C=1e6, max_iter=5000, tol=1e-8).fit((m - mu) / sd, y)
   54|     a = float(clf.coef_[0, 0]) / sd
   55|     return a, float(clf.intercept_[0]) - a * mu
   56| 
   57| 
   58| def build_targets(rows, member_margins, calib, min_members=2, hi=0.9, lo=0.1, clip=(0.02, 0.98)):
   59|     """rows: list of dicts. member_margins: {name: {row_id: margin}}. calib: {name: (margins, y)}."""
   60|     params = {}
   61|     for name, (m, y) in calib.items():
   62|         a, b = platt(m, y)
   63|         if a <= 0:
   64|             raise SystemExit(f"member {name}: Platt slope {a:.4f} is not positive; the margin's sign is wrong")
   65|         params[name] = {"a": a, "b": b, "n_calib": int(len(y))}
   66|     missing = set(member_margins) - set(params)
   67|     if missing:
   68|         raise SystemExit(f"members without calibration rows: {sorted(missing)}")
   69|     out, counts = [], defaultdict(Counter)
   70|     for r in rows:
   71|         rid, pool = r["row_id"], r.get("pool", "?")
   72|         lods = [params[n]["a"] * mm[rid] + params[n]["b"] for n, mm in member_margins.items() if rid in mm]
   73|         scored = len(lods) >= min_members
   74|         p_c = float(1.0 / (1.0 + np.exp(-np.mean(lods)))) if scored else None
   75|         y = r.get("toxicity")
   76|         new = dict(r)
   77|         if y is None:
   78|             if not scored:
   79|                 counts[pool]["dropped_unlabelled_unscored"] += 1
   80|                 continue
   81|             new["toxicity"] = int(p_c >= 0.5)
   82|             new["toxicity_soft"] = float(min(max(p_c, clip[0]), clip[1]))
   83|             rule = "committee"
   84|             counts[pool]["committee_unsafe" if p_c >= 0.5 else "committee_safe"] += 1
   85|         else:
   86|             y = int(y)
   87|             base = r.get("toxicity_soft")
   88|             new["toxicity"] = y
   89|             new["toxicity_soft"] = float(y) if base is None else float(base)
   90|             rule = "source" if scored else "source_unscored"
   91|             if scored and ((y == 0 and p_c >= hi) or (y == 1 and p_c <= lo)):
   92|                 new["toxicity_soft"] = 0.5
   93|                 rule = "clash"
   94|             counts[pool][rule] += 1
   95|         new["p_c"], new["target_rule"] = p_c, rule
   96|         if not 0.0 <= new["toxicity_soft"] <= 1.0:
   97|             raise SystemExit(f"row {rid}: target {new['toxicity_soft']} is outside [0, 1]")
   98|         out.append(new)
   99|     report = {"members": params, "min_members": min_members, "rows_in": len(rows), "rows_out": len(out),
  100|               "per_pool": {p: dict(c) for p, c in sorted(counts.items())}}
  101|     unl = sum(c["committee_unsafe"] + c["committee_safe"] for c in counts.values())
  102|     report["unlabelled_share_unsafe"] = (sum(c["committee_unsafe"] for c in counts.values()) / unl) if unl else None
  103|     report["clash_rows"] = int(sum(c["clash"] for c in counts.values()))
  104|     return out, report
  105| 
  106| 
  107| def _pairs(items):
  108|     out = {}
  109|     for it in items or []:
  110|         name, _, path = it.partition("=")
  111|         if not path:
  112|             raise SystemExit(f"expected NAME=path, got {it}")
  113|         out[name] = path
  114|     return out
  115| 
  116| 
  117| def main(argv=None):
  118|     p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  119|     p.add_argument("--rows", required=True)
  120|     p.add_argument("--member", action="append", required=True)
  121|     p.add_argument("--calib", action="append", required=True)
  122|     p.add_argument("--out", required=True)
  123|     p.add_argument("--report", required=True)
  124|     p.add_argument("--min_members", type=int, default=2)
  125|     a = p.parse_args(argv)
  126|     rows = read_jsonl(a.rows)
  127|     ids = [r["row_id"] for r in rows]
  128|     if len(set(ids)) != len(ids):
  129|         raise SystemExit("row_id values are not unique")
  130|     members = {n: {r["row_id"]: float(r["margin"]) for r in read_jsonl(path)}
  131|                for n, path in _pairs(a.member).items()}
  132|     calib = {}
  133|     for n, path in _pairs(a.calib).items():
  134|         c = read_jsonl(path)
  135|         calib[n] = ([float(r["margin"]) for r in c], [int(r["y"]) for r in c])
  136|     out, report = build_targets(rows, members, calib, a.min_members)
  137|     with open(a.out, "w") as f:
  138|         for r in out:
  139|             f.write(json.dumps(r, ensure_ascii=False) + "\n")
  140|     with open(a.report, "w") as f:
  141|         json.dump(report, f, indent=1)
  142|     print(json.dumps(report, indent=1))
  143|     return report
  144| 
  145| 
  146| if __name__ == "__main__":
  147|     main()
```

## `training/v6_data.py`

```python
    1| """v6_data.py: source loaders for the v6 training pools, dev sets and decontamination references.
    2| 
    3| Runs on Modal only (downloads happen there). Every loader returns plain Python lists/DataFrames;
    4| nothing here prints or persists text. Revisions are resolved once and recorded in the manifest.
    5| """
    6| import csv
    7| import gzip
    8| import io
    9| import json
   10| 
   11| import numpy as np
   12| import pandas as pd
   13| 
   14| AEGIS = ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c")
   15| WGM = "allenai/wildguardmix"
   16| WJB = "allenai/wildjailbreak"
   17| PHTEST = "furonghuang-lab/PHTest"
   18| HQA = "declare-lab/HarmfulQA"
   19| CATQA = "declare-lab/CategoricalHarmfulQA"
   20| JBB = "JailbreakBench/JBB-Behaviors"
   21| HARMAUG = "hbseong/HarmAug_generated_dataset"
   22| WILDCHAT = "allenai/WildChat-1M"
   23| SST = ("Bertievidgen/SimpleSafetyTests", "sst_test_cases.csv",
   24|        "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5")
   25| HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
   26|                  "8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/"
   27|                  "harmbench_behaviors_text_all.csv")
   28| OPENAI_MOD_REPO = "openai/moderation-api-release"
   29| OPENAI_MOD_FILE = "data/samples-1680.jsonl.gz"
   30| OPENAI_MOD_CATS = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")
   31| DEBERTA = "microsoft/deberta-v3-large"
   32| 
   33| 
   34| def _api():
   35|     from huggingface_hub import HfApi
   36|     return HfApi()
   37| 
   38| 
   39| def revision(repo, repo_type="dataset"):
   40|     info = _api().repo_info(repo, repo_type=repo_type)
   41|     return info.sha
   42| 
   43| 
   44| def hub_file(repo, fname, rev=None):
   45|     from huggingface_hub import hf_hub_download
   46|     return hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
   47| 
   48| 
   49| def http_bytes(url):
   50|     import requests
   51|     r = requests.get(url, timeout=180, headers={"User-Agent": "v6-build"})
   52|     r.raise_for_status()
   53|     return r.content
   54| 
   55| 
   56| def github_head(repo):
   57|     import requests
   58|     r = requests.get(f"https://api.github.com/repos/{repo}/commits/HEAD", timeout=60,
   59|                      headers={"User-Agent": "v6-build", "Accept": "application/vnd.github+json"})
   60|     r.raise_for_status()
   61|     return r.json()["sha"]
   62| 
   63| 
   64| def _summ(df, label_cols=()):
   65|     out = {"rows": int(len(df)), "columns": list(map(str, df.columns))}
   66|     for c in label_cols:
   67|         if c in df.columns:
   68|             out[f"counts_{c}"] = {str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()}
   69|     return out
   70| 
   71| 
   72| # ------------------------------------------------------------------------------------------- #
   73| # loaders                                                                                       #
   74| # ------------------------------------------------------------------------------------------- #
   75| def load_aegis(split_file):
   76|     data = json.load(open(hub_file(AEGIS[0], split_file, AEGIS[1]), encoding="utf-8"))
   77|     return pd.DataFrame(data)
   78| 
   79| 
   80| def load_wgm(config):
   81|     from datasets import load_dataset
   82|     rev = revision(WGM)
   83|     ds = load_dataset(WGM, config, revision=rev)
   84|     split = list(ds.keys())[0]
   85|     return ds[split].to_pandas(), rev, split
   86| 
   87| 
   88| def load_wjb():
   89|     rev = revision(WJB)
   90|     p = hub_file(WJB, "train/train.tsv", rev)
   91|     df = pd.read_csv(p, sep="\t", keep_default_na=False, dtype=str, quoting=csv.QUOTE_MINIMAL)
   92|     return df, rev
   93| 
   94| 
   95| def load_hf_any(repo, config=None, rev=None):
   96|     from datasets import load_dataset
   97|     rev = rev or revision(repo)
   98|     ds = load_dataset(repo, config, revision=rev) if config else load_dataset(repo, revision=rev)
   99|     return {k: v.to_pandas() for k, v in ds.items()}, rev
  100| 
  101| 
  102| def load_openai_mod():
  103|     sha = github_head(OPENAI_MOD_REPO)
  104|     raw = http_bytes(f"https://raw.githubusercontent.com/{OPENAI_MOD_REPO}/{sha}/{OPENAI_MOD_FILE}")
  105|     rows = [json.loads(x) for x in gzip.decompress(raw).decode("utf-8").splitlines() if x.strip()]
  106|     return rows, sha
  107| 
  108| 
  109| def load_harmbench_std():
  110|     rows = list(csv.DictReader(io.StringIO(http_bytes(HARMBENCH_URL).decode("utf-8"))))
  111|     return [r["Behavior"] for r in rows if r["FunctionalCategory"] == "standard"]
  112| 
  113| 
  114| def load_sst():
  115|     with open(hub_file(*SST), newline="", encoding="utf-8") as f:
  116|         return [r["prompt"] for r in csv.DictReader(f)]
  117| 
  118| 
  119| # ------------------------------------------------------------------------------------------- #
  120| # inspection                                                                                    #
  121| # ------------------------------------------------------------------------------------------- #
  122| def inspect_all():
  123|     out = {}
  124| 
  125|     def guard(name, fn):
  126|         try:
  127|             out[name] = fn()
  128|         except Exception as e:  # report and keep going
  129|             out[name] = {"error": repr(e)[:500]}
  130| 
  131|     def aegis():
  132|         from huggingface_hub import list_repo_files
  133|         files = list_repo_files(AEGIS[0], repo_type="dataset", revision=AEGIS[1])
  134|         r = {"files": files}
  135|         for f in ("train.json", "validation.json", "test.json"):
  136|             if f in files:
  137|                 r[f] = _summ(load_aegis(f), ["prompt_label"])
  138|         return r
  139| 
  140|     def wgm():
  141|         r = {}
  142|         for c in ("wildguardtrain", "wildguardtest"):
  143|             df, rev, split = load_wgm(c)
  144|             r[c] = _summ(df, ["prompt_harm_label", "adversarial"]) | {"revision": rev, "split": split}
  145|         return r
  146| 
  147|     def wjb():
  148|         from huggingface_hub import list_repo_files
  149|         rev = revision(WJB)
  150|         files = list_repo_files(WJB, repo_type="dataset", revision=rev)
  151|         df, _ = load_wjb()
  152|         r = _summ(df, ["data_type"]) | {"revision": rev, "files": files}
  153|         r["empty_vanilla"] = int((df["vanilla"].str.len() == 0).sum()) if "vanilla" in df else None
  154|         r["empty_adversarial"] = (int((df["adversarial"].str.len() == 0).sum())
  155|                                   if "adversarial" in df else None)
  156|         return r
  157| 
  158|     def generic(repo, config=None, label_cols=()):
  159|         def f():
  160|             splits, rev = load_hf_any(repo, config)
  161|             return {"revision": rev, "splits": {k: _summ(v, label_cols) for k, v in splits.items()}}
  162|         return f
  163| 
  164|     def openai_mod():
  165|         rows, sha = load_openai_mod()
  166|         keys = sorted({k for r in rows for k in r})
  167|         unsafe = sum(1 for r in rows if any(r.get(c) == 1 for c in OPENAI_MOD_CATS))
  168|         return {"commit": sha, "rows": len(rows), "keys": keys, "unsafe_any_flag": unsafe}
  169| 
  170|     guard("aegis", aegis)
  171|     guard("wildguardmix", wgm)
  172|     guard("wildjailbreak", wjb)
  173|     guard("phtest", generic(PHTEST, label_cols=("label", "harmfulness", "category")))
  174|     guard("harmfulqa", generic(HQA, label_cols=("topic",)))
  175|     guard("catqa", generic(CATQA, label_cols=("Category",)))
  176|     guard("jbb", generic(JBB, "behaviors", label_cols=("Category",)))
  177|     guard("openai_mod", openai_mod)
  178|     guard("harmbench_std", lambda: {"rows": len(load_harmbench_std())})
  179|     guard("sst", lambda: {"rows": len(load_sst())})
  180|     guard("deberta_rev", lambda: revision(DEBERTA, "model"))
  181|     guard("harmaug_cols", lambda: {"revision": revision(HARMAUG), "files": _api().list_repo_files(
  182|         HARMAUG, repo_type="dataset")[:20]})
  183|     return out
```

## `training/v6_metrics.py`

```python
    1| """v6_metrics.py: dev-set metrics, the three frozen thresholds, and the two-model average.
    2| 
    3| Pure numpy/sklearn on per-item scores; no text. Higher score = more unsafe; a prompt is flagged
    4| when score >= threshold. The FPR-budget threshold rule and AUPRC come from metrics_core.py.
    5| 
    6| The dictionary keys that dev_report() returns (t_gate, t_screen, t_bal, saturation and so on) are
    7| stored in the run results, so they keep their original names; README.md lists what each means.
    8| """
    9| import math
   10| 
   11| import numpy as np
   12| from sklearn.metrics import roc_auc_score
   13| 
   14| import metrics_core as mc
   15| 
   16| EPS = 1e-12
   17| PINNED_MARGIN = 9.0   # |logit margin| above this means P(unsafe) is within about 1e-4 of 0 or 1
   18| 
   19| 
   20| def probs_from_margin(margin):
   21|     """P(unsafe) = softmax([l0, l1])[1] = sigmoid(l1 - l0), in float64 (so it never rounds to 0 or 1)."""
   22|     m = np.asarray(margin, dtype=np.float64)
   23|     out = np.empty_like(m)
   24|     pos = m >= 0
   25|     out[pos] = 1.0 / (1.0 + np.exp(-m[pos]))
   26|     e = np.exp(m[~pos])
   27|     out[~pos] = e / (1.0 + e)
   28|     return out
   29| 
   30| 
   31| def two_model_average(margin_a, margin_b):
   32|     """Average two models: the mean of their logit margins; P(unsafe) is its sigmoid."""
   33|     m = (np.asarray(margin_a, np.float64) + np.asarray(margin_b, np.float64)) / 2.0
   34|     return m, probs_from_margin(m)
   35| 
   36| 
   37| def auroc(y, s):
   38|     y = np.asarray(y, int)
   39|     return float(roc_auc_score(y, np.asarray(s, float))) if 0 < y.sum() < y.size else None
   40| 
   41| 
   42| # ------------------------------------------------------------------------------------------- #
   43| # the three frozen thresholds (set on dev data only)                                          #
   44| # ------------------------------------------------------------------------------------------- #
   45| def strict_threshold(benign, fpr=0.01):
   46|     """Strict threshold (stored as t_gate): the lowest threshold with FPR <= 1%. k = floor(0.01 n);
   47|     the smallest float strictly above the (k+1)-th largest benign score."""
   48|     return mc.threshold_lowest_at_frr(benign, fpr)
   49| 
   50| 
   51| def prescreen_threshold(unsafe, sens=0.98):
   52|     """Pre-screen threshold (stored as t_screen): the k-th largest unsafe score, k = ceil(0.98 n).
   53|     Escalate when score >= this threshold, so realised recall >= 98% even with ties (ties at the
   54|     threshold are all escalated)."""
   55|     u = np.sort(np.asarray(unsafe, float))[::-1]
   56|     k = min(max(math.ceil(sens * u.size - EPS), 1), u.size)
   57|     return float(u[k - 1])
   58| 
   59| 
   60| def _f1_curve(y, s, thresholds):
   61|     """F1 on the unsafe class for every threshold (flag when s >= t), vectorised."""
   62|     y, s = np.asarray(y, int), np.asarray(s, float)
   63|     pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
   64|     tp = pos.size - np.searchsorted(pos, thresholds, side="left")
   65|     fp = neg.size - np.searchsorted(neg, thresholds, side="left")
   66|     fn = pos.size - tp
   67|     denom = 2 * tp + fp + fn
   68|     return np.where(tp > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
   69| 
   70| 
   71| def balanced_threshold(sets):
   72|     """Balanced threshold (stored as t_bal): the threshold maximising the mean F1 over the given
   73|     (labels, scores) sets, searched over the distinct scores of their items; a tie goes to the
   74|     lower threshold. Returns (t, mean_f1)."""
   75|     cand = np.unique(np.concatenate([np.asarray(s, float) for _, s in sets]))
   76|     mean = np.mean([_f1_curve(y, s, cand) for y, s in sets], axis=0)
   77|     i = int(np.flatnonzero(mean == mean.max())[0])   # cand ascending -> first max = lowest t
   78|     return float(cand[i]), float(mean[i])
   79| 
   80| 
   81| # ------------------------------------------------------------------------------------------- #
   82| # dev metrics                                                                                  #
   83| # ------------------------------------------------------------------------------------------- #
   84| def pair_accuracy(scores, labels, pair_ids):
   85|     """Share of pairs whose unsafe side scores above its benign side; ties count half."""
   86|     by = {}
   87|     for s, y, p in zip(scores, labels, pair_ids):
   88|         by.setdefault(p, {})[int(y)] = float(s)
   89|     vals = []
   90|     for d in by.values():
   91|         if 0 in d and 1 in d:
   92|             vals.append(1.0 if d[1] > d[0] else 0.5 if d[1] == d[0] else 0.0)
   93|     return float(np.mean(vals)) if vals else None, len(vals)
   94| 
   95| 
   96| def pinned_score_share(margins):
   97|     """Share of scores pinned near 0 or 1 (stored as `saturation`): |margin| > PINNED_MARGIN."""
   98|     m = np.abs(np.asarray(margins, float))
   99|     return float((m > PINNED_MARGIN).mean())
  100| 
  101| 
  102| def dev_report(sets):
  103|     """sets: {name: {"score": [...], "margin": [...], "label": [...], "pair_id": [...]?}}.
  104|     Needs twins_prompt, dev_pair2, tc_val, aegis_val; the held-back pool rows are used when present.
  105|     `dev_intent` is the intent score: the mean AUROC on twins_prompt and dev_pair2."""
  106|     g = lambda n, k: np.asarray(sets[n][k])  # noqa: E731
  107|     out = {}
  108|     tw = auroc(g("twins_prompt", "label"), g("twins_prompt", "score"))
  109|     pair2 = auroc(g("dev_pair2", "label"), g("dev_pair2", "score"))
  110|     out["twins_auroc"], out["dev_pair2_auroc"] = tw, pair2
  111|     out["dev_intent"] = float((tw + pair2) / 2.0)
  112|     out["twins_pair_acc"], out["twins_pairs"] = pair_accuracy(
  113|         g("twins_prompt", "score"), g("twins_prompt", "label"), sets["twins_prompt"]["pair_id"])
  114|     out["tc_val_auprc"] = mc.auprc(g("tc_val", "label"), g("tc_val", "score"))
  115|     out["aegis_val_auroc"] = auroc(g("aegis_val", "label"), g("aegis_val", "score"))
  116|     for h in ("dev_holdout_p3", "dev_holdout_p4"):
  117|         if h in sets:
  118|             out[f"{h}_auroc"] = auroc(g(h, "label"), g(h, "score"))
  119|     margins = np.concatenate([g(n, "margin") for n in ("twins_prompt", "tc_val", "aegis_val")])
  120|     out["saturation"] = pinned_score_share(margins)
  121| 
  122|     benign = np.concatenate([g(n, "score")[g(n, "label") == 0] for n in ("tc_val", "aegis_val")])
  123|     unsafe = np.concatenate([g(n, "score")[g(n, "label") == 1] for n in ("tc_val", "aegis_val")])
  124|     strict, prescreen = strict_threshold(benign), prescreen_threshold(unsafe)
  125|     out["t_gate"], out["t_screen"] = strict, prescreen
  126|     out["t_gate_benign_n"], out["t_gate_k"] = int(benign.size), int(math.floor(0.01 * benign.size + EPS))
  127|     bal_sets = [(g(n, "label"), g(n, "score")) for n in ("tc_val", "aegis_val", "dev_holdout_p3")
  128|                 if n in sets]
  129|     if len(bal_sets) == 3:
  130|         out["t_bal"], out["t_bal_mean_f1"] = balanced_threshold(bal_sets)
  131|     if "dev_holdout_p5" in sets:
  132|         out["dev_holdout_p5_frr_at_gate"] = float((g("dev_holdout_p5", "score") >= strict).mean())
  133|     return out
```

## `training/v6_modal.py`

```python
    1| #!/usr/bin/env python
    2| """v6_modal.py: the Modal launcher for building, training and dev-scoring the release-2.0 encoder.
    3| 
    4| Writes under /vol/v6/ on the private Volume VOL_NAME (env V6_VOLUME). Only JSON (counts, hashes,
    5| ids, scores, margins, labels) comes back; text and weights stay on the Volume.
    6| 
    7|   modal run v6_modal.py --action tests        # the v6 unit tests, on the training image
    8|   modal run v6_modal.py --action inspect      # source columns / counts / revisions (no text)
    9|   modal run v6_modal.py --action build        # the five data pools, the R0 file, dev sets, decontamination
   10|   modal run v6_modal.py --action crossfit     # shortcut audit + bag-of-words files for R2
   11|   modal run v6_modal.py --action smoke        # 300-step base-recipe trial run (weights discarded)
   12|   modal run v6_modal.py --action baseline     # release 1.0 trial checkpoints on the dev sets
   13|   modal run v6_modal.py --action wave --arms R1,R0 --seeds 13,42,99 --steps S
   14|   modal run v6_modal.py --action fetch        # pull JSON results to v6_results/
   15| 
   16| `--arms` takes recipe-variant keys (R0-R4); RECIPE_NAMES in v6_score.py says what each one is.
   17| """
   18| import json
   19| import os
   20| import pathlib
   21| import time
   22| 
   23| import modal
   24| 
   25| from train_image import CODE_DIR, image as train_image
   26| 
   27| VOL_NAME = os.environ.get("V6_VOLUME", "v6-data")
   28| APP_NAME = os.environ.get("V6_APP", "v6-train")
   29| MNT = "/vol"
   30| V6 = f"{MNT}/v6"
   31| GPU = "A100-80GB"
   32| MAX_CONTAINERS = 20
   33| LOCAL_DIR = pathlib.Path(__file__).parent
   34| RESULTS_LOCAL = LOCAL_DIR / "v6_results"
   35| 
   36| _IGNORE = ["__pycache__", "checkpoints", ".git", "*.pyc", "*.jsonl", "*.csv", "label_sheets",
   37|            "chart-app", "demo", "leaderboard", "external_rerun", "harness",
   38|            "release", "*.png", "v6_results"]
   39| 
   40| # data-build image: sentence-transformers for the MiniLM pass (not the training stack)
   41| prep_image = (
   42|     modal.Image.debian_slim(python_version="3.11")
   43|     .pip_install("torch>=2.1", "sentence-transformers>=3.0", "datasets>=2.18",
   44|                  "huggingface_hub>=0.23", "numpy>=1.24", "pandas>=2.0", "scikit-learn>=1.3",
   45|                  "requests>=2.31")
   46|     .env({"TOKENIZERS_PARALLELISM": "false"})
   47|     .add_local_python_source("train_image")
   48|     .add_local_dir(LOCAL_DIR, remote_path=CODE_DIR, ignore=_IGNORE)
   49| )
   50| 
   51| app = modal.App(APP_NAME, image=train_image)
   52| vol = modal.Volume.from_name(VOL_NAME)
   53| hf_secret = modal.Secret.from_name("huggingface-secret")
   54| 
   55| 
   56| def _enter_code_dir():
   57|     """Run from the mounted copy of this folder, so its modules import and relative paths work."""
   58|     import sys
   59|     sys.path.insert(0, CODE_DIR)
   60|     os.chdir(CODE_DIR)
   61| 
   62| 
   63| # --------------------------------------------------------------------------------------------- #
   64| # tests                                                                                           #
   65| # --------------------------------------------------------------------------------------------- #
   66| @app.function(cpu=4, timeout=30 * 60)
   67| def run_tests():
   68|     import subprocess
   69|     import sys
   70| 
   71|     import torch
   72|     import transformers
   73|     r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
   74|                         "test_v6_bow.py", "test_v6_committee.py", "test_v6_metrics.py", "test_v6_score.py", "test_v6_build.py"],
   75|                        cwd=CODE_DIR, capture_output=True, text=True)
   76|     return {"returncode": r.returncode, "tail": r.stdout[-6000:] + r.stderr[-2000:],
   77|             "transformers": transformers.__version__, "torch": torch.__version__}
   78| 
   79| 
   80| # --------------------------------------------------------------------------------------------- #
   81| # source inspection (no text returned)                                                            #
   82| # --------------------------------------------------------------------------------------------- #
   83| @app.function(image=prep_image, secrets=[hf_secret], cpu=4, memory=32768, timeout=40 * 60)
   84| def inspect_sources():
   85|     _enter_code_dir()
   86|     import v6_data as vd
   87|     return vd.inspect_all()
   88| 
   89| 
   90| # --------------------------------------------------------------------------------------------- #
   91| # data build                                                                                      #
   92| # --------------------------------------------------------------------------------------------- #
   93| VAL_SPLIT_PREFIX = "e890c3fe"   # first characters of the release's ToxicChat validation-split hash
   94| 
   95| 
   96| @app.function(volumes={MNT: vol}, secrets=[hf_secret], cpu=4, memory=16384, timeout=30 * 60)
   97| def build_p1():
   98|     """The ToxicChat pool (P1, p1_copy) = exactly train.py's training split (training image, same
   99|     datasets/sklearn as train.py), and tc_val = its validation split. Asserts the split's hash prefix."""
  100|     _enter_code_dir()
  101|     from tc_common import INPUT_FIELD, LABEL_FIELD, load_train_val
  102|     from train import split_hash
  103|     from v6_build import write_jsonl
  104|     tr, va = load_train_val(val_fraction=0.10, seed=13)
  105|     h = split_hash(va)
  106|     assert h.startswith(VAL_SPLIT_PREFIX), f"val split hash {h} does not start {VAL_SPLIT_PREFIX}"
  107|     p1 = [{"row_id": f"tc_train:{i}", "user_input": str(t), "toxicity": int(y), "pool": "P1",
  108|            "type": "toxicchat", "toxicity_soft": float(int(y))}
  109|           for i, (t, y) in enumerate(zip(tr[INPUT_FIELD], tr[LABEL_FIELD]))]
  110|     tv = [{"row_id": f"tc_val:{i}", "user_input": str(t), "toxicity": int(y), "pool": "tc_val",
  111|            "type": "toxicchat"} for i, (t, y) in enumerate(zip(va[INPUT_FIELD], va[LABEL_FIELD]))]
  112|     out = {"val_carve_sha256": h, "train_carve_sha256": split_hash(tr),
  113|            "p1_copy": write_jsonl(f"{V6}/pools/p1_copy.jsonl", p1),
  114|            "tc_val": write_jsonl(f"{V6}/dev/tc_val.jsonl", tv)}
  115|     vol.commit()
  116|     return out
  117| 
  118| 
  119| @app.function(image=prep_image, volumes={MNT: vol}, secrets=[hf_secret], gpu="A10G", cpu=16,
  120|               memory=65536, timeout=3 * 60 * 60)
  121| def build_data():
  122|     _enter_code_dir()
  123|     import v6_build
  124|     vol.reload()
  125|     t0 = time.time()
  126|     man = v6_build.build_all(MNT, V6, procs=16)
  127|     man["build_seconds"] = round(time.time() - t0, 1)
  128|     os.makedirs(f"{V6}/manifest", exist_ok=True)
  129|     with open(f"{V6}/manifest/build_manifest.json", "w") as f:
  130|         json.dump(man, f, indent=1)
  131|     vol.commit()
  132|     return man
  133| 
  134| 
  135| @app.function(image=prep_image, volumes={MNT: vol}, gpu="A10G", cpu=16, memory=65536,
  136|               timeout=2 * 60 * 60)
  137| def reverse_audit_job():
  138|     _enter_code_dir()
  139|     import v6_build
  140|     vol.reload()
  141|     out = v6_build.reverse_audit(V6, procs=16)
  142|     with open(f"{V6}/manifest/reverse_audit.json", "w") as f:
  143|         json.dump(out, f, indent=1)
  144|     vol.commit()
  145|     return out
  146| 
  147| 
  148| @app.function(image=prep_image, volumes={MNT: vol}, secrets=[hf_secret], gpu="A10G", cpu=16,
  149|               memory=98304, timeout=3 * 60 * 60)
  150| def build_r3_data():
  151|     _enter_code_dir()
  152|     import v6_build
  153|     vol.reload()
  154|     t0 = time.time()
  155|     man = v6_build.build_r3(MNT, V6, procs=16)
  156|     man["build_seconds"] = round(time.time() - t0, 1)
  157|     with open(f"{V6}/manifest/r3_manifest.json", "w") as f:
  158|         json.dump(man, f, indent=1)
  159|     vol.commit()
  160|     return man
  161| 
  162| 
  163| @app.function(image=prep_image, volumes={MNT: vol}, gpu="A10G", cpu=8, memory=16384,
  164|               timeout=20 * 60)
  165| def p1_matched_heldout_job():
  166|     _enter_code_dir()
  167|     import v6_build
  168|     vol.reload()
  169|     return v6_build.p1_matched_heldout(V6, procs=8)
  170| 
  171| 
  172| @app.function(image=prep_image, volumes={MNT: vol}, gpu="A10G", cpu=16, memory=32768,
  173|               timeout=30 * 60)
  174| def filter_aegis_val_job():
  175|     _enter_code_dir()
  176|     import v6_build
  177|     vol.reload()
  178|     out = v6_build.filter_aegis_val(V6, procs=16)
  179|     with open(f"{V6}/manifest/aegis_val_filter.json", "w") as f:
  180|         json.dump(out, f, indent=1)
  181|     vol.commit()
  182|     return out
  183| 
  184| 
  185| @app.function(volumes={MNT: vol}, cpu=4, memory=32768, timeout=30 * 60)
  186| def build_r3_training_job(targets_path):
  187|     _enter_code_dir()
  188|     import v6_build
  189|     vol.reload()
  190|     rep = v6_build.build_r3_training(V6, targets_path)
  191|     with open(f"{V6}/manifest/r3_training.json", "w") as f:
  192|         json.dump(rep, f, indent=1)
  193|     vol.commit()
  194|     return rep
  195| 
  196| 
  197| # --------------------------------------------------------------------------------------------- #
  198| # base weights pinned by revision (train.py has no --revision flag: pass a local snapshot)        #
  199| # --------------------------------------------------------------------------------------------- #
  200| BASE_DIR = f"{V6}/base/deberta-v3-large"
  201| 
  202| 
  203| def _ensure_base():
  204|     from huggingface_hub import snapshot_download
  205|     from v6_score import BASE_MODEL, BASE_REVISION
  206|     marker = f"{BASE_DIR}/REVISION"
  207|     if os.path.exists(marker) and open(marker).read().strip() == BASE_REVISION:
  208|         return BASE_DIR
  209|     snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_dir=BASE_DIR,
  210|                       allow_patterns=["*.json", "*.model", "*.txt", "*.safetensors", "*.bin"])
  211|     pathlib.Path(marker).write_text(BASE_REVISION)
  212|     vol.commit()
  213|     return BASE_DIR
  214| 
  215| 
  216| # --------------------------------------------------------------------------------------------- #
  217| # shortcut audit + the bag-of-words log-odds for R2 (base recipe + bag-of-words correction)      #
  218| # --------------------------------------------------------------------------------------------- #
  219| @app.function(volumes={MNT: vol}, cpu=16, memory=65536, timeout=2 * 60 * 60)
  220| def crossfit():
  221|     _enter_code_dir()
  222|     import v6_bow_crossfit
  223|     vol.reload()
  224|     names = ["p1_copy", "p2", "p3", "p4", "p5"]
  225|     rep = v6_bow_crossfit.main(["--inputs", *[f"{V6}/pools/{n}.jsonl" for n in names],
  226|                                 "--out_dir", f"{V6}/r2", "--report", f"{V6}/r2/bow_crossfit_report.json"])
  227|     from v6_build import sha_file
  228|     rep["r2_files"] = {n: sha_file(f"{V6}/r2/{n}.jsonl") for n in names}
  229|     vol.commit()
  230|     return rep
  231| 
  232| 
  233| # --------------------------------------------------------------------------------------------- #
  234| # training + dev scoring                                                                          #
  235| # --------------------------------------------------------------------------------------------- #
  236| def _load_model(path):
  237|     import torch
  238|     from transformers import AutoModelForSequenceClassification, AutoTokenizer
  239|     tok = AutoTokenizer.from_pretrained(path)
  240|     model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to("cuda")
  241|     return model, tok
  242| 
  243| 
  244| def _score_dev(model_path, run, out_dir, names=None):
  245|     from v6_build import sha_file
  246|     from v6_metrics import dev_report
  247|     from v6_score import DEV_SETS, dev_sets_for_report, dump, load_dev, score_texts, scorecard
  248|     model, tok = _load_model(model_path)
  249|     dev = load_dev(V6, names or DEV_SETS)
  250|     cards = {}
  251|     for n, rows in dev.items():
  252|         p, m = score_texts(model, tok, [r["user_input"] for r in rows], "cuda")
  253|         cards[n] = scorecard(run, n, rows, p, m,
  254|                              meta={"file_sha256": sha_file(f"{V6}/dev/{n}.jsonl")})
  255|         dump(f"{out_dir}/scorecards/{n}.json", cards[n])
  256|     rep = dev_report(dev_sets_for_report(cards))
  257|     del model
  258|     return rep
  259| 
  260| 
  261| def _train(variant, seed, steps, batch, accum, work, smoke=False):
  262|     """Runs train.py, streaming its output; returns (returncode, log_tail, step_times, peak_mem)."""
  263|     import re
  264|     import subprocess
  265|     import sys
  266|     import threading
  267| 
  268|     from v6_score import build_train_cmd
  269|     base = _ensure_base()
  270|     cmd = build_train_cmd(sys.executable, variant, seed, V6, base, f"{work}/model",
  271|                           f"{work}/run_meta.json", steps, batch, accum)
  272|     print("[cmd]", " ".join(cmd[2:]), flush=True)
  273|     peak = [0]
  274|     stop = threading.Event()
  275| 
  276|     def poll():
  277|         while not stop.is_set():
  278|             try:
  279|                 o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
  280|                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
  281|                 peak[0] = max(peak[0], int(o.stdout.strip().splitlines()[0]))
  282|             except Exception:
  283|                 pass
  284|             stop.wait(5)
  285| 
  286|     th = threading.Thread(target=poll, daemon=True)
  287|     th.start()
  288|     pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  289|     step_t, buf, tail = {}, b"", []
  290|     pat = re.compile(rb"\s(\d+)/(\d+) \[")
  291|     while True:
  292|         ch = pr.stdout.read1(65536) if hasattr(pr.stdout, "read1") else pr.stdout.read(4096)
  293|         if not ch:
  294|             break
  295|         buf += ch
  296|         parts = re.split(rb"[\r\n]", buf)
  297|         buf = parts.pop()
  298|         now = time.time()
  299|         for ln in parts:
  300|             mm = pat.search(ln)
  301|             if mm and int(mm.group(2)) == steps:
  302|                 step_t.setdefault(int(mm.group(1)), now)
  303|             elif ln.strip():
  304|                 tail.append(ln.decode("utf-8", "replace"))
  305|                 tail = tail[-200:]
  306|                 if not smoke:
  307|                     print(ln.decode("utf-8", "replace"), flush=True)
  308|     rc = pr.wait()
  309|     stop.set()
  310|     return rc, "\n".join(tail), step_t, peak[0]
  311| 
  312| 
  313| @app.function(gpu=GPU, volumes={MNT: vol}, secrets=[hf_secret], timeout=2 * 60 * 60)
  314| def smoke(steps=300, batch=32, accum=1):
  315|     """A 300-step base-recipe (R1) trial run, seed 13, to check speed and memory. Weights are
  316|     discarded (never leave /tmp, never scored)."""
  317|     import shutil
  318|     _enter_code_dir()
  319|     vol.reload()
  320|     work = "/tmp/v6_smoke"
  321|     shutil.rmtree(work, ignore_errors=True)
  322|     os.makedirs(work)
  323|     rc, tail, st, peak = _train("R1", 13, steps, batch, accum, work, smoke=True)
  324|     out = {"returncode": rc, "batch": batch, "grad_accum": accum, "peak_gpu_mem_mib": peak,
  325|            "oom": "out of memory" in tail.lower(), "log_tail": tail[-3000:]}
  326|     k0 = min([k for k in st if k >= 100], default=None)
  327|     if k0 is not None and steps in st and k0 < steps:
  328|         out["r_steps_per_s_100_to_end"] = (steps - k0) / (st[steps] - st[k0])
  329|         out["r_from_step"] = k0
  330|     out["step_times_rel"] = {k: round(v - min(st.values()), 2) for k, v in sorted(st.items())
  331|                              if k in (1, 50, 100, 150, 200, 250, 300)} if st else {}
  332|     if rc == 0 and os.path.exists(f"{work}/run_meta.json"):
  333|         rm = json.load(open(f"{work}/run_meta.json"))
  334|         out["run_meta"] = {k: v for k, v in rm.items() if k != "log_history"}
  335|         out["log_history_tail"] = rm.get("log_history", [])[-4:]
  336|     shutil.rmtree(work, ignore_errors=True)   # weights discarded
  337|     return out
  338| 
  339| 
  340| @app.function(gpu=GPU, volumes={MNT: vol}, timeout=60 * 60, max_containers=4)
  341| def score_trial_checkpoint(run):
  342|     """Score one release 1.0 trial checkpoint on the v6 dev sets (margins stored). The mean intent
  343|     score of these runs is the release 1.0 trial intent score that v6_select.py compares against."""
  344|     _enter_code_dir()
  345|     vol.reload()
  346|     rep = _score_dev(f"{MNT}/runs/{run}/model", run, f"{V6}/baseline/{run}")
  347|     from v6_score import dump
  348|     dump(f"{V6}/baseline/{run}/dev_report.json", rep)
  349|     vol.commit()
  350|     return {"run": run, **rep}
  351| 
  352| 
  353| @app.function(gpu=GPU, volumes={MNT: vol}, secrets=[hf_secret], timeout=4 * 60 * 60,
  354|               max_containers=MAX_CONTAINERS)
  355| def train_and_score(variant, seed, steps, batch=32, accum=1):
  356|     import shutil
  357|     _enter_code_dir()
  358|     from v6_score import dump, run_name
  359|     run = run_name(variant, seed)
  360|     rdir = f"{V6}/runs/{run}"
  361|     vol.reload()
  362|     if os.path.exists(f"{rdir}/DONE"):
  363|         return json.load(open(f"{rdir}/summary.json")) | {"skipped": True}
  364|     work = f"/tmp/{run}"
  365|     shutil.rmtree(work, ignore_errors=True)
  366|     os.makedirs(work)
  367|     t0 = time.time()
  368|     rc, tail, st, peak = _train(variant, seed, steps, batch, accum, work)
  369|     if rc:
  370|         raise RuntimeError(f"{run}: train.py exited {rc}\n{tail[-3000:]}")
  371|     train_s = time.time() - t0
  372|     for d in pathlib.Path(f"{work}/model").glob("checkpoint-*"):
  373|         shutil.rmtree(d)
  374|     os.makedirs(rdir, exist_ok=True)
  375|     shutil.rmtree(f"{rdir}/model", ignore_errors=True)
  376|     shutil.copytree(f"{work}/model", f"{rdir}/model")
  377|     rm = json.load(open(f"{work}/run_meta.json"))
  378|     t1 = time.time()
  379|     rep = _score_dev(f"{work}/model", run, rdir)
  380|     import torch
  381|     import transformers
  382|     summary = {"run": run, "arm": variant, "seed": seed, "steps": steps, "batch": batch,
  383|                "grad_accum": accum, "dev": rep,
  384|                "run_meta": {k: v for k, v in rm.items() if k != "log_history"},
  385|                "timing_s": {"train": round(train_s, 1), "score": round(time.time() - t1, 1)},
  386|                "peak_gpu_mem_mib": peak, "gpu_name": torch.cuda.get_device_name(0),
  387|                "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
  388|     dump(f"{rdir}/run_meta.json", rm)
  389|     dump(f"{rdir}/summary.json", summary)
  390|     pathlib.Path(f"{rdir}/DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
  391|     vol.commit()
  392|     return summary
  393| 
  394| 
  395| RELEASE_LEAK = ("threshold.json", "training_args.bin", "results.md", "optimizer.pt", "scheduler.pt",
  396|                 "trainer_state.json", "all_results.json", "eval_results.json")
  397| 
  398| 
  399| @app.function(gpu=GPU, volumes={MNT: vol}, timeout=60 * 60)
  400| def stage_release(run, target, training_config):
  401|     """Stage the release files for `run` under /v6/release/<run>/stage on the private Volume.
  402|     No Hugging Face call is made. Then reload the staged directory through the v6 scoring path and
  403|     compare its tc_val and aegis_val margins with the run's stored dev scorecards."""
  404|     import hashlib
  405|     import shutil
  406| 
  407|     import numpy as np
  408|     import torch
  409|     import transformers
  410|     from transformers import AutoModelForSequenceClassification, AutoTokenizer
  411|     _enter_code_dir()
  412|     from v6_score import load_dev, score_texts
  413|     vol.reload()
  414|     src, out = f"{V6}/runs/{run}/model", f"{V6}/release/{run}/stage"
  415|     shutil.rmtree(out, ignore_errors=True)
  416|     os.makedirs(out)
  417|     model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32)
  418|     tok = AutoTokenizer.from_pretrained(src)
  419|     src_id2label = {int(k): v for k, v in model.config.id2label.items()}
  420|     assert src_id2label == {0: "non_toxic", 1: "toxic"}, f"unexpected id2label {src_id2label}"
  421|     model.config.id2label = {0: "safe", 1: "unsafe"}
  422|     model.config.label2id = {"safe": 0, "unsafe": 1}
  423|     model.config._name_or_path = target
  424|     if hasattr(model.config, "name_or_path"):
  425|         model.config.name_or_path = target
  426|     model.save_pretrained(out, safe_serialization=True)
  427|     tok.save_pretrained(out)
  428|     tcp = f"{out}/tokenizer_config.json"
  429|     if os.path.exists(tcp):
  430|         tc = json.load(open(tcp))
  431|         if tc.pop("is_local", None) is not None:
  432|             with open(tcp, "w") as f:
  433|                 json.dump(tc, f, indent=2, ensure_ascii=False)
  434|                 f.write("\n")
  435|     with open(f"{out}/training_config.json", "w") as f:
  436|         json.dump(training_config, f, indent=2, ensure_ascii=False)
  437|         f.write("\n")
  438|     for leaky in RELEASE_LEAK:
  439|         if os.path.exists(f"{out}/{leaky}"):
  440|             os.remove(f"{out}/{leaky}")
  441|     cfg_text = open(f"{out}/config.json").read()
  442|     assert "/vol/" not in cfg_text and "/tmp/" not in cfg_text, "a local path leaked into config.json"
  443|     n_params = int(sum(p.numel() for p in model.parameters()))
  444|     del model
  445| 
  446|     def sha(p):
  447|         h = hashlib.sha256()
  448|         with open(p, "rb") as fh:
  449|             for b in iter(lambda: fh.read(1 << 20), b""):
  450|                 h.update(b)
  451|         return h.hexdigest()
  452| 
  453|     files = {f: {"bytes": os.path.getsize(f"{out}/{f}"), "sha256": sha(f"{out}/{f}")}
  454|              for f in sorted(os.listdir(out))}
  455|     # parity: the staged directory through the same scoring path as the stored dev scorecards
  456|     m2 = AutoModelForSequenceClassification.from_pretrained(out, dtype=torch.float32).to("cuda")
  457|     t2 = AutoTokenizer.from_pretrained(out)
  458|     dev = load_dev(V6, ("tc_val", "aegis_val"))
  459|     parity = {}
  460|     for n, rows in dev.items():
  461|         _, m = score_texts(m2, t2, [r["user_input"] for r in rows], "cuda")
  462|         ref = json.load(open(f"{V6}/runs/{run}/scorecards/{n}.json"))["rows"]
  463|         assert [r["src"] for r in ref] == [r["row_id"] for r in rows], n
  464|         d = np.abs(m - np.array([r["margin"] for r in ref]))
  465|         parity[n] = {"n": len(rows), "max_abs_margin_diff": float(d.max()),
  466|                      "bit_identical": bool((d == 0).all())}
  467|     src_w = sha(f"{src}/model.safetensors")
  468|     vol.commit()
  469|     return {"run": run, "stage": out, "target": target, "files": files, "n_params": n_params,
  470|             "source_model_safetensors_sha256": src_w,
  471|             "weights_byte_identical_to_scored": src_w == files["model.safetensors"]["sha256"],
  472|             "config": json.load(open(f"{out}/config.json")), "parity": parity,
  473|             "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
  474|             "gpu": torch.cuda.get_device_name(0)}
  475| 
  476| 
  477| @app.function(volumes={MNT: vol}, timeout=30 * 60)
  478| def collect():
  479|     """JSON results only (summaries, dev reports, scorecards of ids/scores/margins/labels)."""
  480|     vol.reload()
  481|     out = {}
  482|     for base in ("runs", "baseline"):
  483|         for rdir in sorted(pathlib.Path(f"{V6}/{base}").glob("*")):
  484|             files = [p for p in rdir.rglob("*.json") if "model" not in p.relative_to(rdir).parts]
  485|             out[f"{base}/{rdir.name}"] = {str(p.relative_to(rdir)): json.load(open(p)) for p in files}
  486|     return out
  487| 
  488| 
  489| @app.local_entrypoint()
  490| def main(action: str = "tests", arms: str = "", seeds: str = "13,42,99", steps: int = 0,
  491|          batch: int = 32, accum: int = 1):
  492|     if action == "tests":
  493|         out = run_tests.remote()
  494|         print(out["tail"])
  495|         print(f"[versions] transformers={out['transformers']} torch={out['torch']}")
  496|         if out["returncode"]:
  497|             raise SystemExit(out["returncode"])
  498|     elif action == "inspect":
  499|         print(json.dumps(inspect_sources.remote(), indent=1))
  500|     elif action == "build":
  501|         p1 = build_p1.remote()
  502|         print(json.dumps(p1, indent=1), flush=True)
  503|         man = build_data.remote()
  504|         man["p1"] = p1
  505|         RESULTS_LOCAL.mkdir(exist_ok=True)
  506|         (RESULTS_LOCAL / "build_manifest.json").write_text(json.dumps(man, indent=1))
  507|         print(json.dumps({k: v for k, v in man.items() if k not in ("type_counts",)}, indent=1))
  508|     elif action == "reverse":
  509|         out = reverse_audit_job.remote()
  510|         RESULTS_LOCAL.mkdir(exist_ok=True)
  511|         (RESULTS_LOCAL / "reverse_audit.json").write_text(json.dumps(out, indent=1))
  512|         print(json.dumps(out, indent=1))
  513|     elif action == "stage":
  514|         run, target = "R3_s13", "partyhat-hf/partyhat-safety-encoder"
  515|         sel = json.loads((RESULTS_LOCAL / "selection.json").read_text())
  516|         thr = sel["thresholds"][run]
  517|         # training_config.json ships with the released model; re-staging writes the published file unchanged.
  518|         tc = json.loads((LOCAL_DIR / "training_config.json").read_text())
  519|         assert tc["operating_threshold"] == thr["t_bal"]
  520|         out = stage_release.remote(run, target, tc)
  521|         RESULTS_LOCAL.mkdir(exist_ok=True)
  522|         (RESULTS_LOCAL / "release_stage_R3_s13.json").write_text(json.dumps(out, indent=1))
  523|         print(json.dumps({k: v for k, v in out.items() if k != "config"}, indent=1))
  524|     elif action == "d1items":   # ToxicChat-pool items that match held-out evaluation items
  525|         out = p1_matched_heldout_job.remote()
  526|         RESULTS_LOCAL.mkdir(exist_ok=True)
  527|         (RESULTS_LOCAL / "d1_p1_matched_heldout.json").write_text(json.dumps(out, indent=1))
  528|         print(json.dumps(out, indent=1))
  529|     elif action == "aegisval":
  530|         out = filter_aegis_val_job.remote()
  531|         RESULTS_LOCAL.mkdir(exist_ok=True)
  532|         (RESULTS_LOCAL / "aegis_val_filter.json").write_text(json.dumps(out, indent=1))
  533|         print(json.dumps(out, indent=1))
  534|     elif action == "r3train":
  535|         rep = build_r3_training_job.remote(f"{V6}/r3/r3_committee/r3_targets.jsonl")
  536|         RESULTS_LOCAL.mkdir(exist_ok=True)
  537|         (RESULTS_LOCAL / "r3_training.json").write_text(json.dumps(rep, indent=1))
  538|         print(json.dumps(rep, indent=1))
  539|     elif action == "r3build":
  540|         man = build_r3_data.remote()
  541|         RESULTS_LOCAL.mkdir(exist_ok=True)
  542|         (RESULTS_LOCAL / "r3_manifest.json").write_text(json.dumps(man, indent=1))
  543|         print(json.dumps(man, indent=1))
  544|     elif action == "crossfit":
  545|         rep = crossfit.remote()
  546|         RESULTS_LOCAL.mkdir(exist_ok=True)
  547|         (RESULTS_LOCAL / "bow_crossfit_report.json").write_text(json.dumps(rep, indent=1))
  548|         print(json.dumps(rep, indent=1))
  549|     elif action == "smoke":
  550|         out = smoke.remote(steps or 300, batch, accum)
  551|         RESULTS_LOCAL.mkdir(exist_ok=True)
  552|         (RESULTS_LOCAL / f"smoke_b{batch}x{accum}.json").write_text(json.dumps(out, indent=1))
  553|         print(json.dumps({k: v for k, v in out.items() if k != "log_tail"}, indent=1))
  554|         print(out["log_tail"][-2500:])
  555|     elif action == "baseline":
  556|         # the release 1.0 trial checkpoints, by their run folder names on the volume
  557|         res = list(score_trial_checkpoint.map(["A0_s13", "A0_s13_repeat"]))
  558|         keys = ("dev_intent", "twins_auroc", "dev_pair2_auroc", "twins_pair_acc", "tc_val_auprc",
  559|                 "aegis_val_auroc", "saturation")
  560|         b = sum(r["dev_intent"] for r in res) / len(res)
  561|         out = {"runs": res, "B": b}
  562|         RESULTS_LOCAL.mkdir(exist_ok=True)
  563|         (RESULTS_LOCAL / "baseline_B.json").write_text(json.dumps(out, indent=1))
  564|         for r in res:
  565|             print(r["run"], {k: round(r[k], 4) for k in keys if r.get(k) is not None})
  566|         print(f"[baseline] release 1.0 trial intent score (B) = {b:.4f}")
  567|     elif action == "wave":
  568|         assert steps > 0 and arms, "--arms and --steps are required"
  569|         jobs = [(a, int(s), steps, batch, accum) for a in arms.split(",") for s in seeds.split(",")]
  570|         print(f"[wave] {len(jobs)} runs: {jobs}", flush=True)
  571|         res = list(train_and_score.starmap(jobs, return_exceptions=True))
  572|         from v6_score import RECIPE_NAMES
  573|         for j, r in zip(jobs, res):
  574|             print(j[:2], RECIPE_NAMES.get(j[0], ""),
  575|                   "FAILED: " + repr(r)[:500] if isinstance(r, Exception)
  576|                   else {k: r["dev"].get(k) for k in ("dev_intent", "tc_val_auprc", "saturation")})
  577|     elif action == "fetch":
  578|         allr = collect.remote()
  579|         for run, files in allr.items():
  580|             for f, obj in files.items():
  581|                 q = RESULTS_LOCAL / run / f
  582|                 q.parent.mkdir(parents=True, exist_ok=True)
  583|                 q.write_text(json.dumps(obj))
  584|         print(f"[fetch] {len(allr)} dirs -> {RESULTS_LOCAL}")
  585|     else:
  586|         raise SystemExit(f"unknown action {action}")
```

## `training/v6_score.py`

```python
    1| """v6_score.py: the recipe variants, the train.py command for each run, and the dev scorer.
    2| 
    3| Plain Python (no Modal), unit-tested in test_v6_score.py.
    4| """
    5| import datetime
    6| import hashlib
    7| import json
    8| import os
    9| import pathlib
   10| 
   11| import numpy as np
   12| 
   13| import v6_metrics as vm
   14| 
   15| BASE_MODEL = "microsoft/deberta-v3-large"
   16| BASE_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
   17| SEEDS = (13, 42, 99)
   18| # Training files per recipe variant: (folder, files, uses the bag-of-words correction).
   19| # The ToxicChat file (p1_copy) is listed twice; train.py's own ToxicChat loader adds the third copy.
   20| # The keys R0-R4 are part of run folder names (for example R3_s13), so they stay as they are.
   21| RECIPE_VARIANTS = {
   22|     "R1": ("pools", ["p1_copy", "p1_copy", "p2", "p3", "p4", "p5"], False),
   23|     "R0": ("pools", ["p1_copy", "p1_copy", "p2", "p3", "p4_r0"], False),
   24|     "R2": ("r2", ["p1_copy", "p1_copy", "p2", "p3", "p4", "p5"], True),
   25|     "R3": ("r3/pools", ["p1_copy", "p1_copy", "p2", "p3", "p4", "p5", "r3_extra"], False),
   26|     # R4 was added after the first training runs: the base recipe without FalseReject (P5).
   27|     "R4": ("pools", ["p1_copy", "p1_copy", "p2", "p3", "p4"], False),
   28| }
   29| # Readable names for the recipe-variant keys, used where results are printed.
   30| RECIPE_NAMES = {
   31|     "R1": "base recipe",
   32|     "R0": "base recipe without the alarming-but-benign data",
   33|     "R2": "base recipe + bag-of-words correction",
   34|     "R3": "base recipe + teacher labels (the released recipe)",
   35|     "R4": "base recipe without FalseReject",
   36| }
   37| RECIPE = dict(lr="2e-05", max_len="512", loss="weighted_ce", class_weights="1.0,1.0",
   38|               rdrop_alpha="0", val_fraction="0.10", split_seed="13",
   39|               soft_label_field="toxicity_soft")
   40| DEV_SETS = ("twins_prompt", "dev_pair2", "tc_val", "aegis_val",
   41|             "dev_holdout_p3", "dev_holdout_p4", "dev_holdout_p5")
   42| 
   43| 
   44| def run_name(variant, seed):
   45|     return f"{variant}_s{seed}"
   46| 
   47| 
   48| def build_train_cmd(py, variant, seed, v6_root, base_path, out_dir, run_meta, steps,
   49|                     batch_size=32, grad_accum=1):
   50|     """The exact train.py command line for one training run of a recipe variant."""
   51|     sub, files, bow = RECIPE_VARIANTS[variant]
   52|     r = RECIPE
   53|     extra = [os.path.join(v6_root, sub, f"{f}.jsonl") for f in files]
   54|     cmd = [py, "train.py", "--output_dir", out_dir, "--model", base_path,
   55|            "--extra_train_jsonl", *extra,
   56|            "--soft_label_field", r["soft_label_field"], "--loss", r["loss"],
   57|            "--class_weights", r["class_weights"], "--lr", r["lr"],
   58|            "--batch_size", str(batch_size), "--grad_accum", str(grad_accum),
   59|            "--max_len", r["max_len"], "--rdrop_alpha", r["rdrop_alpha"], "--allow_tf32",
   60|            "--group_by_length", "--max_steps", str(int(steps)), "--eval_steps", str(int(steps)),
   61|            "--seed", str(seed), "--head_seed", str(seed), "--split_seed", r["split_seed"],
   62|            "--val_fraction", r["val_fraction"], "--run_meta", run_meta]
   63|     if bow:
   64|         cmd += ["--bow_logit_field", "bow_logit"]
   65|     assert "--push_to_hub" not in cmd and "--warm_start_model" not in cmd
   66|     return cmd
   67| 
   68| 
   69| def score_texts(model, tok, texts, device, max_len=512, batch_size=64):
   70|     """(P(unsafe), margin) per item in the original order. margin = logit_unsafe - logit_safe
   71|     from the FP32 logits; P = softmax(logits)[:, 1] computed as sigmoid(margin) in float64."""
   72|     import torch
   73|     order = np.argsort([len(t) for t in texts], kind="stable")
   74|     margin = np.empty(len(texts), dtype=np.float64)
   75|     model.eval()
   76|     with torch.no_grad():
   77|         for s in range(0, len(order), batch_size):
   78|             idx = order[s:s + batch_size]
   79|             enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len,
   80|                       padding=True, return_tensors="pt").to(device)
   81|             lg = model(**enc).logits.float()
   82|             margin[idx] = (lg[:, 1] - lg[:, 0]).double().cpu().numpy()
   83|     return vm.probs_from_margin(margin), margin
   84| 
   85| 
   86| def scorecard(run, set_name, rows, scores, margins, threshold=0.5, meta=None):
   87|     """One scorecard: per item {i, src, label, score, margin, decision}; no text."""
   88|     return {"run": run, "set": set_name, "n": len(rows), "decision_threshold": threshold,
   89|             "rows": [{"i": i, "src": r["row_id"], "label": int(r["toxicity"]), "score": float(s),
   90|                       "margin": float(m), "decision": int(s >= threshold)}
   91|                      for i, (r, s, m) in enumerate(zip(rows, scores, margins))],
   92|             "meta": meta or {}}
   93| 
   94| 
   95| def load_dev(v6_root, names=DEV_SETS):
   96|     out = {}
   97|     for n in names:
   98|         p = pathlib.Path(v6_root) / "dev" / f"{n}.jsonl"
   99|         with open(p, encoding="utf-8") as f:
  100|             out[n] = [json.loads(x) for x in f if x.strip()]
  101|     return out
  102| 
  103| 
  104| def dev_sets_for_report(cards):
  105|     """scorecards {name: card} -> the dict v6_metrics.dev_report expects."""
  106|     sets = {}
  107|     for n, c in cards.items():
  108|         d = {"score": [r["score"] for r in c["rows"]], "margin": [r["margin"] for r in c["rows"]],
  109|              "label": [r["label"] for r in c["rows"]]}
  110|         if n == "twins_prompt":
  111|             # src = "calm:<pair_id>:<label>"
  112|             d["pair_id"] = [r["src"][len("calm:"):].rsplit(":", 1)[0] for r in c["rows"]]
  113|         sets[n] = d
  114|     return sets
  115| 
  116| 
  117| # ------------------------------------------------------------------------------------------- #
  118| # the test lock: test sets stay closed until the selection record has been sealed            #
  119| # ------------------------------------------------------------------------------------------- #
  120| TEST_SETS = ("xstest", "jbb", "orbench", "openai_mod", "harmbench_std", "sst", "toxicchat_test",
  121|              "toxicchat_dedup", "toxicchat_human", "aegis2_test_prompt", "wildguardtest_prompt")
  122| SELECTION = "V6_SELECTION.md"
  123| SELECTION_SEAL = "V6_SELECTION.sha256"   # {"sha256", "utc_timestamp", "size"} of the sealed record
  124| 
  125| 
  126| class TestSetLocked(PermissionError):
  127|     pass
  128| 
  129| 
  130| def record_digest(path):
  131|     """sha256 and size of a file with line endings normalised to LF and exactly one trailing newline."""
  132|     raw = pathlib.Path(path).read_bytes()
  133|     canon = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n") + b"\n"
  134|     return hashlib.sha256(canon).hexdigest(), len(canon)
  135| 
  136| 
  137| def seal_selection(root):
  138|     """Seal V6_SELECTION.md by writing its sha256 and size to V6_SELECTION.sha256. Refuses to replace a
  139|     seal of different content, so a sealed selection cannot be rewritten without it showing."""
  140|     md, seal = pathlib.Path(root) / SELECTION, pathlib.Path(root) / SELECTION_SEAL
  141|     digest, size = record_digest(md)
  142|     if seal.exists():
  143|         old = json.loads(seal.read_text())
  144|         if old.get("sha256") != digest:
  145|             raise TestSetLocked(f"{SELECTION_SEAL} seals different content; refusing to reseal")
  146|         return old
  147|     rec = {"sha256": digest,
  148|            "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  149|            "size": size}
  150|     seal.write_text(json.dumps(rec, indent=1) + "\n")
  151|     return rec
  152| 
  153| 
  154| def selection_sealed(root):
  155|     """True once V6_SELECTION.md has been sealed and V6_SELECTION.sha256 still matches it."""
  156|     md, seal = pathlib.Path(root) / SELECTION, pathlib.Path(root) / SELECTION_SEAL
  157|     if not (md.is_file() and seal.is_file()):
  158|         return False
  159|     try:
  160|         rec = json.loads(seal.read_text())
  161|     except (OSError, json.JSONDecodeError):
  162|         return False
  163|     return (rec.get("sha256"), rec.get("size")) == record_digest(md)
  164| 
  165| 
  166| def assert_test_unlocked(set_name, root):
  167|     """Refuse to score a v6 checkpoint on any test set until the selection record is sealed, so the
  168|     recipe and thresholds are chosen on dev sets before any test result is seen. Unknown set names
  169|     are refused too, so a renamed test set cannot slip past the lock."""
  170|     if set_name in DEV_SETS:
  171|         return
  172|     if set_name not in TEST_SETS:
  173|         raise TestSetLocked(f"{set_name}: not a known v6 dev or test set; refusing")
  174|     if not selection_sealed(root):
  175|         raise TestSetLocked(f"{set_name} is a test set: locked until {SELECTION} is sealed and "
  176|                             f"{SELECTION_SEAL} matches it")
  177| 
  178| 
  179| def score_test_sets(run, set_names, root, load_set, score, threshold=0.5):
  180|     """The only v6 test-scoring path. Every set is checked against the lock before ANY set is
  181|     loaded or any model is touched. load_set(name) -> rows; score(rows) -> (p, margin)."""
  182|     for n in set_names:
  183|         assert_test_unlocked(n, root)
  184|     out = {}
  185|     for n in set_names:
  186|         rows = load_set(n)
  187|         p, m = score(rows)
  188|         out[n] = scorecard(run, n, rows, p, m, threshold)
  189|     return out
  190| 
  191| 
  192| def dump(path, obj):
  193|     os.makedirs(os.path.dirname(path), exist_ok=True)
  194|     with open(path, "w") as f:
  195|         json.dump(obj, f)
```

## `training/v6_select.py`

```python
    1| #!/usr/bin/env python3
    2| """v6_select.py: choose the release candidate from per-item dev scorecards (no text).
    3| 
    4| Reads v6_results/runs/<run>/scorecards/<set>.json (fetched with `v6_modal.py --action fetch`) and
    5| builds every eligible candidate: each recipe variant on its own (R1, R0, R2, R3, R4) and the
    6| two-model averages E11, E12, E13, E23, made from the single models' margins. It then applies the
    7| selection rule and the release bars, computes the three frozen thresholds for each training run
    8| of the selected candidate, and writes V6_SELECTION.md and v6_results/selection.json.
    9| 
   10|     python3 v6_select.py [--results v6_results] [--out V6_SELECTION.md]
   11| 
   12| To re-run the selection from the evidence bundle's development scorecards (labelled copy, made by
   13| attach_labels.py), point --dev-scorecards at its scorecards/dev folder:
   14| 
   15|     python3 v6_select.py --dev-scorecards ../evidence_with_labels/scorecards/dev --results rerun --out rerun/V6_SELECTION.md
   16| 
   17| The keys written to selection.json (B, G1-G3, replicates, t_gate and so on) keep their original
   18| names, because v6_modal.py and the stored results read them; README.md lists what each means.
   19| """
   20| import argparse
   21| import json
   22| import pathlib
   23| 
   24| import numpy as np
   25| 
   26| import v6_metrics as vm
   27| from v6_score import DEV_SETS, RECIPE_NAMES, dev_sets_for_report
   28| 
   29| SEEDS = (13, 42, 99)
   30| SINGLE_MODELS = ("R1", "R0", "R2", "R3", "R4")   # R4 was added later and is only used on its own
   31| # Two-model averages: each candidate pairs two runs, (variant, seed, variant, seed), per seed.
   32| TWO_MODEL_AVERAGES = {
   33|     "E11": [("R1", 13, "R1", 42), ("R1", 42, "R1", 99), ("R1", 99, "R1", 13)],
   34|     "E12": [("R1", s, "R2", s) for s in SEEDS],
   35|     "E13": [("R1", s, "R3", s) for s in SEEDS],
   36|     "E23": [("R2", s, "R3", s) for s in SEEDS],
   37| }
   38| CANDIDATE_NAMES = RECIPE_NAMES | {
   39|     "E11": "two-model average: base recipe + base recipe (different seeds)",
   40|     "E12": "two-model average: base recipe + bag-of-words correction",
   41|     "E13": "two-model average: base recipe + teacher labels",
   42|     "E23": "two-model average: bag-of-words correction + teacher labels",
   43| }
   44| 
   45| # The intent score of the release 1.0 trial checkpoints scored on the v6 dev sets
   46| # (v6_modal.py --action baseline, saved as v6_results/baseline_B.json).
   47| RELEASE_1_TRIAL_INTENT_SCORE = 0.7147105385513868
   48| # The three release bars, checked on the selected candidate's mean over its training runs.
   49| INTENT_SCORE_BAR = RELEASE_1_TRIAL_INTENT_SCORE + 0.05    # intent score must reach this
   50| TOXICCHAT_AUPRC_BAR = 0.80                               # tc_val AUPRC must reach this
   51| PINNED_SCORE_LIMIT = 0.50                                # share of pinned scores must stay below
   52| # A candidate may trail the base recipe's tc_val AUPRC by at most this much.
   53| TOXICCHAT_AUPRC_MARGIN = 0.02
   54| GATE_NAMES = {"G1": "intent score bar", "G2": "ToxicChat AUPRC bar",
   55|               "G3": "share of scores pinned near 0 or 1"}
   56| 
   57| 
   58| def name_of(candidate):
   59|     return f"{candidate} ({CANDIDATE_NAMES[candidate]})" if candidate in CANDIDATE_NAMES else candidate
   60| 
   61| 
   62| def load_run(results, run, dev_scorecards=None):
   63|     """One run's dev scorecards, from v6_results/runs/<run>/scorecards/ or, if dev_scorecards is
   64|     given, from <dev_scorecards>/<run>/ (the evidence bundle's layout)."""
   65|     d = (pathlib.Path(dev_scorecards) / run if dev_scorecards
   66|          else pathlib.Path(results) / "runs" / run / "scorecards")
   67|     if not all((d / f"{n}.json").exists() for n in DEV_SETS):
   68|         return None
   69|     return {n: json.load(open(d / f"{n}.json")) for n in DEV_SETS}
   70| 
   71| 
   72| def combine(ca, cb):
   73|     """Two-model average scorecards: mean margin, sigmoid score; items must align by src."""
   74|     out = {}
   75|     for n in DEV_SETS:
   76|         ra, rb = ca[n]["rows"], cb[n]["rows"]
   77|         assert [r["src"] for r in ra] == [r["src"] for r in rb], f"{n}: item order differs"
   78|         m, p = vm.two_model_average([r["margin"] for r in ra], [r["margin"] for r in rb])
   79|         out[n] = {"rows": [dict(r, margin=float(mm), score=float(pp)) for r, mm, pp in zip(ra, m, p)]}
   80|     return out
   81| 
   82| 
   83| def summarise(reps):
   84|     """reps: list of dev reports, one per training run. Mean and range of every numeric key."""
   85|     keys = [k for k, v in reps[0].items() if isinstance(v, (int, float)) and v is not None]
   86|     out = {}
   87|     for k in keys:
   88|         vals = [r[k] for r in reps if r.get(k) is not None]
   89|         if len(vals) == len(reps):
   90|             out[k] = {"mean": float(np.mean(vals)), "range": float(max(vals) - min(vals)),
   91|                       "values": vals}
   92|     return out
   93| 
   94| 
   95| def choose_candidate(cands, trace):
   96|     """Pick the leading candidate by intent score. A single model must beat the base recipe (R1)
   97|     by more than the larger of their run-to-run ranges, and a two-model average must beat that
   98|     single model the same way. Returns the chosen candidate."""
   99|     di = lambda c: cands[c]["dev_intent"]["mean"]           # noqa: E731
  100|     rg = lambda c: cands[c]["dev_intent"]["range"]          # noqa: E731
  101|     singles = sorted([c for c in cands if c in SINGLE_MODELS], key=di, reverse=True)
  102|     best = singles[0]
  103|     if best != "R1" and di(best) - di("R1") < max(rg(best), rg("R1")):
  104|         trace.append(f"best single model: {best} leads R1 by {di(best) - di('R1'):.4f}, less than "
  105|                      f"the larger run-to-run range {max(rg(best), rg('R1')):.4f}, so R1 stays")
  106|         best = "R1"
  107|     else:
  108|         trace.append(f"best single model: {best} (mean intent score {di(best):.4f})")
  109|     averages = sorted([c for c in cands if c in TWO_MODEL_AVERAGES], key=di, reverse=True)
  110|     if averages:
  111|         t = averages[0]
  112|         lead, need = di(t) - di(best), max(rg(t), rg(best))
  113|         if lead > need:
  114|             trace.append(f"best two-model average: {t} leads {best} by {lead:.4f} > {need:.4f}, "
  115|                          f"so {t} replaces it")
  116|             return t
  117|         trace.append(f"best two-model average: {t} leads {best} by {lead:.4f}, not > {need:.4f}; "
  118|                      f"{best} stands")
  119|     return best
  120| 
  121| 
  122| def select(cands):
  123|     """Choose a candidate, then check it against the base recipe's tc_val AUPRC. A candidate that
  124|     trails R1 by more than TOXICCHAT_AUPRC_MARGIN is removed and the choice is made again from
  125|     the rest, until one passes. R1 always passes. Returns (selected, trace)."""
  126|     if "R1" not in cands:
  127|         raise SystemExit("R1 does not have three dev-scored training runs; nothing can be selected")
  128|     trace, remaining = [], dict(cands)
  129|     r1 = cands["R1"]["tc_val_auprc"]["mean"]
  130|     while True:
  131|         c = choose_candidate(remaining, trace)
  132|         gap = r1 - cands[c]["tc_val_auprc"]["mean"]
  133|         if c == "R1" or gap <= TOXICCHAT_AUPRC_MARGIN:
  134|             trace.append(f"ToxicChat check: {c} tc_val AUPRC {cands[c]['tc_val_auprc']['mean']:.4f} "
  135|                          f"(R1 {r1:.4f}; gap {gap:.4f} <= {TOXICCHAT_AUPRC_MARGIN}): selected")
  136|             return c, trace
  137|         trace.append(f"ToxicChat check: {c} fails (tc_val AUPRC gap {gap:.4f} > "
  138|                      f"{TOXICCHAT_AUPRC_MARGIN}); removed, and the choice is made again")
  139|         del remaining[c]
  140| 
  141| 
  142| def main(argv=None):
  143|     a = argparse.ArgumentParser()
  144|     a.add_argument("--results", default="v6_results")
  145|     a.add_argument("--out", default="V6_SELECTION.md")
  146|     a.add_argument("--dev-scorecards", default=None,
  147|                    help="read <dir>/<run>/<set>.json (the evidence bundle's scorecards/dev, labelled)")
  148|     a = a.parse_args(argv)
  149|     rerun = a.dev_scorecards is not None
  150| 
  151|     runs = {}
  152|     for variant in SINGLE_MODELS:
  153|         for s in SEEDS:
  154|             c = load_run(a.results, f"{variant}_s{s}", a.dev_scorecards)
  155|             if c is not None:
  156|                 runs[(variant, s)] = c
  157|     reports, cands, runs_of = {}, {}, {}
  158|     for (variant, s), c in runs.items():
  159|         reports[f"{variant}_s{s}"] = vm.dev_report(dev_sets_for_report(c))
  160|     for variant in SINGLE_MODELS:
  161|         if all((variant, s) in runs for s in SEEDS):
  162|             runs_of[variant] = [f"{variant}_s{s}" for s in SEEDS]
  163|             cands[variant] = summarise([reports[r] for r in runs_of[variant]])
  164|     for name, pairs in TWO_MODEL_AVERAGES.items():
  165|         if all((x, s) in runs and (y, t) in runs for x, s, y, t in pairs):
  166|             reps = []
  167|             for x, s, y, t in pairs:
  168|                 key = f"{name}[{x}_s{s}+{y}_s{t}]"
  169|                 reports[key] = vm.dev_report(dev_sets_for_report(combine(runs[(x, s)], runs[(y, t)])))
  170|                 reps.append(key)
  171|             runs_of[name] = reps
  172|             cands[name] = summarise([reports[r] for r in reps])
  173| 
  174|     sel, trace = select(cands)
  175|     s = cands[sel]
  176|     gates = {"G1": {"value": s["dev_intent"]["mean"], "bar": INTENT_SCORE_BAR,
  177|                     "pass": s["dev_intent"]["mean"] >= INTENT_SCORE_BAR},
  178|              "G2": {"value": s["tc_val_auprc"]["mean"], "bar": TOXICCHAT_AUPRC_BAR,
  179|                     "pass": s["tc_val_auprc"]["mean"] >= TOXICCHAT_AUPRC_BAR},
  180|              "G3": {"value": s["saturation"]["mean"], "bar": PINNED_SCORE_LIMIT,
  181|                     "pass": s["saturation"]["mean"] < PINNED_SCORE_LIMIT}}
  182|     shipped = runs_of[sel][0]
  183|     thr = {r: {k: reports[r][k] for k in ("t_gate", "t_screen", "t_bal")} for r in runs_of[sel]}
  184|     out = {"B": RELEASE_1_TRIAL_INTENT_SCORE, "candidates": cands, "replicates": runs_of,
  185|            "run_reports": reports, "selected": sel, "trace": trace, "gates": gates,
  186|            "all_gates_pass": all(g["pass"] for g in gates.values()),
  187|            "shipped_replicate": shipped, "thresholds": thr}
  188|     pathlib.Path(a.results).mkdir(parents=True, exist_ok=True)
  189|     pathlib.Path(a.results, "selection.json").write_text(json.dumps(out, indent=1))
  190| 
  191|     source = ("Recomputed by `v6_select.py` from the evidence bundle's development scorecards."
  192|               if rerun else
  193|               "Computed by `v6_select.py` from per-item dev scorecards only. No test set has been scored "
  194|               "for any v6 checkpoint.")
  195|     L = ["# V6_SELECTION: dev-set selection", "",
  196|          f"{source} Full numbers: `{pathlib.Path(a.results, 'selection.json')}`.", "",
  197|          f"Release 1.0 trial intent score = {RELEASE_1_TRIAL_INTENT_SCORE:.4f}; intent score bar = "
  198|          f"{INTENT_SCORE_BAR:.4f}; ToxicChat AUPRC bar = {TOXICCHAT_AUPRC_BAR}; share of scores "
  199|          f"pinned near 0 or 1 must stay below {PINNED_SCORE_LIMIT}.", "",
  200|          "## Candidates (mean over three training runs [range])", "",
  201|          "| Candidate | What it is | Training runs | Intent score | Twins AUROC | dev_pair2 AUROC | Pair acc | tc_val AUPRC | aegis_val AUROC | Pinned scores |",
  202|          "|---|---|---|---|---|---|---|---|---|---|"]
  203|     f = lambda c, k: f"{cands[c][k]['mean']:.4f} [{cands[c][k]['range']:.4f}]"  # noqa: E731
  204|     for c in cands:
  205|         L.append(f"| {c} | {CANDIDATE_NAMES.get(c, '')} | {', '.join(runs_of[c])} | {f(c, 'dev_intent')} | "
  206|                  f"{f(c, 'twins_auroc')} | {f(c, 'dev_pair2_auroc')} | {f(c, 'twins_pair_acc')} | "
  207|                  f"{f(c, 'tc_val_auprc')} | {f(c, 'aegis_val_auroc')} | {f(c, 'saturation')} |")
  208|     L += ["", "## Every run", "",
  209|           "| Run | Intent score | Twins | dev_pair2 | tc_val AUPRC | aegis_val AUROC | Held-back WildGuardMix AUROC | Held-back WildJailbreak AUROC | Held-back FalseReject FRR at strict | Pinned scores | Strict | Pre-screen | Balanced |",
  210|           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
  211|     for r, d in reports.items():
  212|         L.append(f"| {r} | {d['dev_intent']:.4f} | {d['twins_auroc']:.4f} | {d['dev_pair2_auroc']:.4f} | "
  213|                  f"{d['tc_val_auprc']:.4f} | {d['aegis_val_auroc']:.4f} | {d['dev_holdout_p3_auroc']:.4f} | "
  214|                  f"{d['dev_holdout_p4_auroc']:.4f} | {d['dev_holdout_p5_frr_at_gate']:.4f} | {d['saturation']:.4f} | "
  215|                  f"{d['t_gate']!r} | {d['t_screen']!r} | {d['t_bal']!r} |")
  216|     L += ["", "## Selection", ""] + [f"- {t}" for t in trace]
  217|     L += ["", f"**Selected: {name_of(sel)}.** Shipped model (fixed before training): the first "
  218|           f"training run, `{shipped}`.", "",
  219|           "## Release bars (on the selected candidate's mean over its training runs)", ""]
  220|     for k, g in gates.items():
  221|         L.append(f"- {GATE_NAMES[k]}: {g['value']:.4f} vs {g['bar']:.4f}: **{'PASS' if g['pass'] else 'FAIL'}**")
  222|     L += ["", f"**All bars: {'PASS: sealing this file unlocks the test sets' if out['all_gates_pass'] else 'FAIL: the test sets stay locked'}.**",
  223|           "", "## Frozen thresholds (per training run, on P(unsafe) = sigmoid(margin))", "",
  224|           "| Training run | Strict | Pre-screen | Balanced |", "|---|---|---|---|"]
  225|     for r, t in thr.items():
  226|         L.append(f"| {r}{' (shipped)' if r == shipped else ''} | {t['t_gate']!r} | {t['t_screen']!r} | {t['t_bal']!r} |")
  227|     if rerun:
  228|         L += [""]
  229|     else:
  230|         # The Record states how the selection was made. Those statements are written, not computed;
  231|         # R4's figures are computed like every other number in this file.
  232|         r4 = cands.get("R4")
  233|         r4_line = (f"- R4 was added after the first round's dev results had been seen and before any v6 "
  234|                    f"test score. Its outcome here: intent score {r4['dev_intent']['mean']:.4f} (range "
  235|                    f"{r4['dev_intent']['range']:.4f}), tc_val AUPRC {r4['tc_val_auprc']['mean']:.4f}. "
  236|                    f"**R4 was {'' if sel == 'R4' else 'not '}selected.**" if r4 else
  237|                    "- R4 has no three dev-scored training runs.")
  238|         L += ["", "## Record", "",
  239|               "- One selection, made after every launched run was dev-scored: the first round of "
  240|               "training runs (R1, R0, R2), then R3 and R4. The decision to train R3 used no dev result.",
  241|               r4_line,
  242|               "- Dev results could be seen as soon as each run finished. None of them could change this "
  243|               "selection, since the selection rule is mechanical.",
  244|               "- No v6 checkpoint has been scored on any test set.", "",
  245|               "The selection is not revisited after any test set is seen.", ""]
  246|     pathlib.Path(a.out).write_text("\n".join(L))
  247|     print("\n".join(trace))
  248|     print(json.dumps({"selected": sel, "selected_name": CANDIDATE_NAMES.get(sel, sel),
  249|                       "gates": gates, "shipped": shipped}, indent=1))
  250| 
  251| 
  252| if __name__ == "__main__":
  253|     main()
```
