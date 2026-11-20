# partyhat-safety-encoder-eval: source and docs, part 2 of 7

Generated from commit `d251619f4822` (2026-11-20) of github.com/partyhat-ai/partyhat-safety-encoder-eval by `scripts/llms_digest.py`, which `.github/workflows/llms-digest.yml` reruns on every push to `main`, so this text matches the code at that commit.
Files `evidence/attach_labels.py` to `evidence/reports/dev_ablation.md`. Lines are numbered as in the repository.

## `evidence/attach_labels.py`

```python
    1| #!/usr/bin/env python3
    2| """attach_labels.py: rebuild the item labels stripped from the evidence bundle, from the public datasets.
    3| 
    4| The label-stripped bundle omits item labels for the ToxicChat, Aegis 2.0 and WildGuardTest test sets and for five
    5| development sets (tc_val from ToxicChat, aegis_val from Aegis 2.0, dev_holdout_p3 from WildGuardMix,
    6| dev_holdout_p4 from WildJailbreak and dev_holdout_p5 from FalseReject), so that no third-party labels are
    7| redistributed. This script rebuilds them from the public datasets at the pinned revisions, with YOUR Hugging Face
    8| access (accept the gated terms of allenai/wildguardmix and allenai/wildjailbreak yourself first), and verifies
    9| every rebuilt label vector against the sha256 the bundle carries (reports/label_hashes.json). It refuses to write a
   10| vector whose hash differs.
   11| 
   12|     pip install pandas scikit-learn huggingface_hub pyarrow
   13|     python3 attach_labels.py <stripped bundle folder> <output folder>
   14| 
   15| The output folder is a copy of the bundle with labels attached; then run
   16|     python3 reproduce_card.py <output folder>
   17|     python3 dev_ablation.py <output folder>
   18| """
   19| import csv
   20| import hashlib
   21| import json
   22| import pathlib
   23| import shutil
   24| import sys
   25| 
   26| PINS = {
   27|     "toxicchat": ("lmsys/toxic-chat", "29df8e4dba60e1f4af4b4075c0705c5b313548a8"),
   28|     "aegis": ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"),
   29|     "wildguardmix": ("allenai/wildguardmix", "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"),
   30|     "wildjailbreak": ("allenai/wildjailbreak", "5ddc12a7894f842b0619b8e1c7ee496b198af009"),
   31|     "falsereject": ("AmazonScience/FalseReject", "493ba967714ea54c6f01067e1f61e389cc2c9b3e"),
   32| }
   33| _CACHE = {}
   34| 
   35| 
   36| def hub(key, fname):
   37|     from huggingface_hub import hf_hub_download
   38|     repo, rev = PINS[key]
   39|     return hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
   40| 
   41| 
   42| def toxicchat(split):
   43|     if split not in _CACHE:
   44|         with open(hub("toxicchat", f"data/0124/toxic-chat_annotation_{split}.csv"), newline="", encoding="utf-8") as f:
   45|             _CACHE[split] = list(csv.DictReader(f))
   46|     return _CACHE[split]
   47| 
   48| 
   49| def aegis(fname):
   50|     if fname not in _CACHE:
   51|         _CACHE[fname] = json.load(open(hub("aegis", fname), encoding="utf-8"))
   52|     return _CACHE[fname]
   53| 
   54| 
   55| def labels_toxicchat_test(rows):
   56|     src = toxicchat("test")
   57|     return [{"label": int(src[r["src"]]["toxicity"]), "jailbreaking": int(src[r["src"]]["jailbreaking"])} for r in rows]
   58| 
   59| 
   60| def labels_aegis2_test_prompt(rows):
   61|     data = aegis("test.json")
   62|     out = []
   63|     for r in rows:
   64|         d = data[r["src"]]
   65|         cats = [c.strip() for c in str(d.get("violated_categories") or "").split(",") if c.strip()]
   66|         nc_only = cats == ["Needs Caution"]
   67|         shipped = int(d["prompt_label"] == "unsafe")
   68|         out.append({"label": 1 if nc_only else shipped, "label_shipped": shipped, "gt_nc_only": nc_only})
   69|     return out
   70| 
   71| 
   72| def labels_wildguardtest_prompt(rows):
   73|     import pandas as pd
   74|     if "wgt" not in _CACHE:
   75|         _CACHE["wgt"] = pd.read_parquet(hub("wildguardmix", "test/wildguard_test.parquet"))
   76|     df = _CACHE["wgt"]
   77|     return [{"label": int(df.iloc[r["src"]]["prompt_harm_label"] == "harmful"),
   78|              "adversarial": bool(df.iloc[r["src"]]["adversarial"])} for r in rows]
   79| 
   80| 
   81| def labels_tc_val(rows):
   82|     import pandas as pd
   83|     from sklearn.model_selection import train_test_split
   84|     if "tcval" not in _CACHE:
   85|         df = pd.DataFrame(toxicchat("train"))
   86|         df["toxicity"] = df["toxicity"].astype(int)
   87|         _, val = train_test_split(df, test_size=0.1, random_state=13, stratify=df["toxicity"])
   88|         _CACHE["tcval"] = val.reset_index(drop=True)
   89|     val = _CACHE["tcval"]
   90|     return [{"label": int(val.iloc[int(str(r["src"]).split(":")[1])]["toxicity"])} for r in rows]
   91| 
   92| 
   93| def labels_aegis_val(rows):
   94|     data = {d["id"]: d for d in aegis("validation.json")}
   95|     return [{"label": int(data[str(r["src"]).split(":", 1)[1]]["prompt_label"] == "unsafe")} for r in rows]
   96| 
   97| 
   98| def labels_dev_holdout_p3(rows):
   99|     import pandas as pd
  100|     if "wgm" not in _CACHE:
  101|         _CACHE["wgm"] = pd.read_parquet(hub("wildguardmix", "train/wildguard_train.parquet"))
  102|     df = _CACHE["wgm"]
  103|     return [{"label": int(df.iloc[int(str(r["src"]).split(":")[1])]["prompt_harm_label"] == "harmful")} for r in rows]
  104| 
  105| 
  106| def labels_dev_holdout_p4(rows):
  107|     import pandas as pd
  108|     if "wjb" not in _CACHE:
  109|         _CACHE["wjb"] = pd.read_csv(hub("wildjailbreak", "train/train.tsv"), sep="\t", keep_default_na=False,
  110|                                     usecols=["data_type"])
  111|     df = _CACHE["wjb"]
  112|     return [{"label": int(str(df.iloc[int(str(r["src"]).split(":")[1])]["data_type"]).endswith("_harmful"))} for r in rows]
  113| 
  114| 
  115| def labels_dev_holdout_p5(rows):
  116|     """Every dev_holdout_p5 row is a prompt from FalseReject's train split, which the dataset defines as benign
  117|     prompts that look harmful (its card at the pinned revision: "A benign but high-risk-looking prompt"). The label
  118|     is therefore 0 for every row; the card is fetched to confirm the pinned revision, and the hash check below
  119|     confirms the vector."""
  120|     card = pathlib.Path(hub("falsereject", "README.md")).read_text(encoding="utf-8")
  121|     if "benign but high-risk-looking prompt" not in card:
  122|         raise SystemExit("FalseReject card at the pinned revision no longer defines its train prompts as benign")
  123|     return [{"label": 0} for _ in rows]
  124| 
  125| 
  126| BUILDERS = {"toxicchat_test": labels_toxicchat_test, "aegis2_test_prompt": labels_aegis2_test_prompt,
  127|             "wildguardtest_prompt": labels_wildguardtest_prompt, "tc_val": labels_tc_val,
  128|             "aegis_val": labels_aegis_val, "dev_holdout_p3": labels_dev_holdout_p3,
  129|             "dev_holdout_p4": labels_dev_holdout_p4, "dev_holdout_p5": labels_dev_holdout_p5}
  130| 
  131| 
  132| def label_sha(rows):
  133|     return hashlib.sha256(json.dumps([int(r["label"]) for r in rows], separators=(",", ":")).encode()).hexdigest()
  134| 
  135| 
  136| def main():
  137|     src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
  138|     if dst.exists():
  139|         sys.exit(f"{dst} exists; choose a new output folder")
  140|     shutil.copytree(src, dst)
  141|     hashes = json.loads((dst / "reports" / "label_hashes.json").read_text())
  142|     done = bad = 0
  143|     for rel, want in sorted(hashes.items()):
  144|         p = dst / rel
  145|         card = json.loads(p.read_text())
  146|         rows = card["rows"]
  147|         set_key = p.stem.split("__")[-1] if "__" in p.stem else p.stem
  148|         built = BUILDERS[set_key](rows)
  149|         merged = [dict(r, **b) for r, b in zip(rows, built)]
  150|         got = label_sha(merged)
  151|         if got != want:
  152|             print(f"MISMATCH {rel}: rebuilt labels sha256 {got[:12]} != bundle {want[:12]}; not written")
  153|             bad += 1
  154|             continue
  155|         card["rows"] = merged
  156|         card.pop("labels_removed", None)
  157|         card["labels_rebuilt"] = "attach_labels.py from the public dataset at the pinned revision; sha256 verified"
  158|         p.write_text(json.dumps(card))
  159|         done += 1
  160|     print(f"labels rebuilt and verified for {done} files; {bad} mismatches")
  161|     sys.exit(1 if bad else 0)
  162| 
  163| 
  164| if __name__ == "__main__":
  165|     main()
```

## `evidence/dev_ablation.py`

```python
    1| #!/usr/bin/env python3
    2| """dev_ablation.py: recompute the "Where the gain came from" section of the partyhat-safety-encoder 2.0 card and
    3| the full tables behind it (reports/dev_ablation.md), from the per-item development scorecards in this bundle.
    4| 
    5| This is a descriptive analysis made after the fact. It was not planned before the training runs, it uses no test
    6| set, and it trains or scores nothing: it only re-reads the development scorecards produced when the release 2.0
    7| recipe was chosen. Read each difference as "on these development sets, these training runs differed by this much",
    8| not as a measured causal effect of one change.
    9| 
   10| Reads a LABELLED copy of the bundle (run attach_labels.py first):
   11| 
   12|     pip install numpy scikit-learn
   13|     python3 dev_ablation.py <labelled bundle folder> [--quick] [--write-report]
   14| 
   15| For every figure in the card section it prints the card's wording, the card's value and the recomputed value; it
   16| then regenerates reports/dev_ablation.md and compares it with the copy in the bundle, line by line. It exits
   17| non-zero on any mismatch. --write-report rewrites reports/dev_ablation.md instead of comparing it. --quick uses
   18| 1,000 bootstrap draws (intervals will then not match). With the full draws it runs in about a minute.
   19| 
   20| Recipe variants (run folders scorecards/dev/<folder>_s<seed>/; the folder names are kept from the training code):
   21| - base recipe (folder R1): ToxicChat train (three copies), Aegis 2.0, WildGuardMix, WildJailbreak (all four
   22|   types) and FalseReject, from microsoft/deberta-v3-large with no warm start.
   23| - base recipe without the alarming-but-benign data (R0): no FalseReject and no WildJailbreak benign rows.
   24| - base recipe without FalseReject (R4).
   25| - base recipe + bag-of-words correction (R2): the same rows; during training the logit of a bag-of-words model is
   26|   added to the encoder's logit.
   27| - base recipe + teacher labels (R3, the released recipe; R3_s13 is the published weights): labels that three open
   28|   guard models (the teachers) confidently disputed are trained toward 0.5, and 100,000 extra prompts are added, most
   29|   of them with teacher targets.
   30| - Each of these has three training runs (seeds 13, 42, 99), all 9,396 optimizer steps at batch 32, so the number of
   31|   passes over the data differs by variant.
   32| - Two-model averages (no training): the mean of two runs' logit margins.
   33| - Release 1.0 trial checkpoints (folders A0_s13, A0_s13_repeat): release 1.0's recipe retrained twice with the same
   34|   seed (the runs differ only by GPU nondeterminism), warm-started from
   35|   cooperleong00/deberta-v3-large_toxicity-scorer, scored on the same development sets.
   36| 
   37| Metrics (higher score = more unsafe; `score` is P(unsafe)):
   38| - AUROC on each two-class development set; the intent score is the mean AUROC on the paired prompts and on the
   39|   PHTest, HarmfulQA and CatQA set; AUPRC on the ToxicChat validation split (sklearn average_precision_score).
   40| - The FalseReject held-back rows are all benign, so they have no AUROC; their figure is the share flagged at the
   41|   strict threshold: with n benign rows of the ToxicChat validation split and the Aegis 2.0 validation set and
   42|   k = floor(0.01 n), a prompt is flagged when its score is above the (k+1)-th largest of those benign scores. The
   43|   threshold is recomputed inside every bootstrap draw.
   44| - The trial checkpoints were scored on an earlier, 1,199-row Aegis 2.0 validation set; 40 rows that match test items
   45|   were removed before the other runs were scored. They are evaluated here on the 1,159 shared rows (joined by `src`).
   46| 
   47| Intervals: 95% paired bootstrap, 10,000 draws, built as in reproduce_card.py. Benign and unsafe items are resampled
   48| separately within each set, with the same item draws for every training run; each set has its own random stream,
   49| numpy default_rng([13, zlib.crc32(set file name)]). Training runs are resampled with replacement inside each draw,
   50| independently for each recipe variant (stream "training_runs:<variant key>"; reproduce_card.py resamples one recipe
   51| and uses "training_runs"). A difference's interval is the 2.5th to 97.5th percentile of its 10,000 draws.
   52| """
   53| import argparse
   54| import json
   55| import math
   56| import pathlib
   57| import sys
   58| import zlib
   59| 
   60| import numpy as np
   61| from sklearn.metrics import average_precision_score, roc_auc_score
   62| 
   63| BASE_SEED = 13
   64| SEEDS = (13, 42, 99)
   65| EPS = 1e-12
   66| 
   67| # variant key -> (run-folder prefix, plain name)
   68| TRAINED = {
   69|     "base": ("R1", "base recipe"),
   70|     "no_alarming_benign": ("R0", "base recipe without the alarming-but-benign data"),
   71|     "no_falsereject": ("R4", "base recipe without FalseReject"),
   72|     "bag_of_words": ("R2", "base recipe + bag-of-words correction"),
   73|     "teacher_labels": ("R3", "base recipe + teacher labels (released)"),
   74| }
   75| # two-model averages: key -> (plain name, [(variant, seed), (variant, seed)] per training run)
   76| AVERAGES = {
   77|     "avg_base_base": ("two-model average: base recipe, two seeds",
   78|                       [(("base", 13), ("base", 42)), (("base", 42), ("base", 99)), (("base", 99), ("base", 13))]),
   79|     "avg_base_bow": ("two-model average: base recipe and + bag-of-words correction",
   80|                      [(("base", s), ("bag_of_words", s)) for s in SEEDS]),
   81|     "avg_base_teacher": ("two-model average: base recipe and + teacher labels",
   82|                          [(("base", s), ("teacher_labels", s)) for s in SEEDS]),
   83|     "avg_bow_teacher": ("two-model average: + bag-of-words correction and + teacher labels",
   84|                         [(("bag_of_words", s), ("teacher_labels", s)) for s in SEEDS]),
   85| }
   86| TRIAL_RUNS = ["A0_s13", "A0_s13_repeat"]
   87| TRIAL = ("release_1_trial", "release 1.0 trial checkpoints")
   88| DISPLAY_ORDER = [TRIAL[0], "base", "no_alarming_benign", "no_falsereject", "bag_of_words", "teacher_labels",
   89|                  "avg_base_base", "avg_base_bow", "avg_base_teacher", "avg_bow_teacher"]
   90| 
   91| # set file name -> plain name
   92| SETS = {
   93|     "twins_prompt": "paired prompts (generated in-house)",
   94|     "dev_pair2": "PHTest, HarmfulQA and CatQA set",
   95|     "tc_val": "ToxicChat validation split",
   96|     "aegis_val": "Aegis 2.0 validation set",
   97|     "dev_holdout_p3": "WildGuardMix held-back rows",
   98|     "dev_holdout_p4": "WildJailbreak held-back rows",
   99|     "dev_holdout_p5": "FalseReject held-back rows",
  100| }
  101| TWO_CLASS_SETS = ["twins_prompt", "dev_pair2", "tc_val", "aegis_val", "dev_holdout_p3", "dev_holdout_p4"]
  102| BENIGN_ONLY_SET = "dev_holdout_p5"
  103| REFERENCE_RUN = "R1_s13"
  104| 
  105| # metric key, column label, kind ("auc": 3 decimals; "flagged": percent)
  106| METRICS = [("intent_score", "intent score", "auc")]
  107| METRICS += [(f"{s}.auroc", SETS[s], "auc") for s in TWO_CLASS_SETS]
  108| METRICS += [(f"{BENIGN_ONLY_SET}.flagged", "FalseReject held-back rows flagged at the strict threshold", "flagged"),
  109|             ("tc_val.auprc", "ToxicChat validation AUPRC", "auc")]
  110| 
  111| # contrast: (plain name, variant X, variant Y, what differs). The difference is X minus Y.
  112| CONTRASTS = [
  113|     ("Adding teacher labels (with 100,000 extra prompts)", "teacher_labels", "base",
  114|      "two changes together. (1) 2,182 of the base recipe's WildGuardMix, WildJailbreak and FalseReject rows are "
  115|      "trained toward 0.5 where the teachers confidently disputed the label. (2) 100,000 prompts are added: 30,000 more "
  116|      "WildJailbreak vanilla rows (their own labels, unless disputed: 425 were, which with the 2,182 above makes the "
  117|      "card's 2,607 disputed rows), 30,000 instructions from the HarmAug generated "
  118|      "dataset and 40,000 WildChat-1M first turns (teacher targets only); 35,761 of the 100,000 are unsafe. "
  119|      "250,316 rows seen 1.20 times against 150,316 rows seen 2.00 times"),
  120|     ("Adding FalseReject", "base", "no_falsereject",
  121|      "the FalseReject pool alone: 14,185 benign prompts. 150,316 rows seen 2.00 times against 136,131 seen 2.21 times"),
  122|     ("Adding the alarming-but-benign data", "base", "no_alarming_benign",
  123|      "a combined change: the FalseReject pool (14,185 rows) and WildJailbreak's benign rows (29,100: 14,550 "
  124|      "vanilla_benign and 14,550 adversarial_benign) together. 150,316 rows seen 2.00 times against 107,031 seen 2.81 "
  125|      "times"),
  126|     ("Adding WildJailbreak's benign rows", "no_falsereject", "no_alarming_benign",
  127|      "WildJailbreak's benign rows alone (29,100), FalseReject absent from both. 136,131 rows seen 2.21 times against "
  128|      "107,031 seen 2.81 times. The variant without FalseReject was added after the first development results of the "
  129|      "base recipe, the variant without the alarming-but-benign data and the bag-of-words variant had been seen"),
  130|     ("Adding the bag-of-words correction", "bag_of_words", "base",
  131|      "the bag-of-words correction alone: the same rows, seen the same number of times (2.00)"),
  132|     ("Base recipe against the release 1.0 trial checkpoints", "base", TRIAL[0],
  133|      "everything between release 1.0's recipe and the base recipe at once: the training data, the starting weights "
  134|      "(a toxicity-scorer checkpoint against the plain base model) and the training settings"),
  135|     ("Released recipe against the release 1.0 trial checkpoints", "teacher_labels", TRIAL[0],
  136|      "the two previous steps together: base recipe against the trial checkpoints, plus adding teacher labels"),
  137| ]
  138| 
  139| 
  140| # ----------------------------------------------------------------------------------------------- data
  141| def probability_from_margin(margin):
  142|     m = np.asarray(margin, dtype=np.float64)
  143|     out = np.empty_like(m)
  144|     pos = m >= 0
  145|     out[pos] = 1.0 / (1.0 + np.exp(-m[pos]))
  146|     e = np.exp(m[~pos])
  147|     out[~pos] = e / (1.0 + e)
  148|     return out
  149| 
  150| 
  151| def run_folder(variant, seed):
  152|     return f"{TRAINED[variant][0]}_s{seed}"
  153| 
  154| 
  155| class DevScorecards:
  156|     """Per-run score vectors on every development set, aligned to the reference run's item order."""
  157| 
  158|     def __init__(self, root):
  159|         self.root = pathlib.Path(root) / "scorecards" / "dev"
  160|         self.labels, self.row_ids, self.score, self.margin = {}, {}, {}, {}
  161|         self.trial_full_aegis = {}
  162|         all_sets = TWO_CLASS_SETS + [BENIGN_ONLY_SET]
  163|         for s in all_sets:
  164|             card = self._card(REFERENCE_RUN, s)
  165|             self.labels[s] = np.array([int(r["label"]) for r in card["rows"]], int)
  166|             self.row_ids[s] = [r["src"] for r in card["rows"]]
  167|         folders = [run_folder(v, seed) for v in TRAINED for seed in SEEDS] + TRIAL_RUNS
  168|         for run in folders:
  169|             for s in all_sets:
  170|                 rows = self._card(run, s)["rows"]
  171|                 if run in TRIAL_RUNS and s == "aegis_val":
  172|                     self.trial_full_aegis[run] = rows
  173|                     by_id = {r["src"]: r for r in rows}
  174|                     rows = [by_id[x] for x in self.row_ids[s]]
  175|                 assert [r["src"] for r in rows] == self.row_ids[s], (run, s, "item order differs")
  176|                 assert all(int(r["label"]) == y for r, y in zip(rows, self.labels[s])), (run, s, "labels differ")
  177|                 self.score[(run, s)] = np.array([r["score"] for r in rows], float)
  178|                 self.margin[(run, s)] = np.array([r["margin"] for r in rows], float)
  179|         self.runs_of = {v: [run_folder(v, seed) for seed in SEEDS] for v in TRAINED}
  180|         self.runs_of[TRIAL[0]] = list(TRIAL_RUNS)
  181|         for key, (_, pairs) in AVERAGES.items():
  182|             self.runs_of[key] = []
  183|             for (va, sa), (vb, sb) in pairs:
  184|                 ra, rb = run_folder(va, sa), run_folder(vb, sb)
  185|                 name = f"{ra} + {rb}"
  186|                 for s in all_sets:
  187|                     m = (self.margin[(ra, s)] + self.margin[(rb, s)]) / 2.0
  188|                     self.margin[(name, s)], self.score[(name, s)] = m, probability_from_margin(m)
  189|                 self.runs_of[key].append(name)
  190| 
  191|     def _card(self, run, s):
  192|         p = self.root / run / f"{s}.json"
  193|         if not p.exists():
  194|             raise SystemExit(f"missing scorecard {p}")
  195|         card = json.loads(p.read_text())
  196|         if "labels_removed" in card:
  197|             raise SystemExit(f"{p} has no labels: run attach_labels.py and pass the labelled copy")
  198|         return card
  199| 
  200|     def all_runs(self):
  201|         return [r for v in self.runs_of for r in self.runs_of[v]]
  202| 
  203| 
  204| def plain_name(variant):
  205|     if variant == TRIAL[0]:
  206|         return TRIAL[1]
  207|     if variant in TRAINED:
  208|         return TRAINED[variant][1]
  209|     return AVERAGES[variant][0]
  210| 
  211| 
  212| # ----------------------------------------------------------------------------------------------- point metrics
  213| def strict_cut(benign_scores):
  214|     """The (k+1)-th largest benign score, k = floor(0.01 n); a prompt is flagged when its score is above it."""
  215|     b = np.sort(np.asarray(benign_scores, float))[::-1]
  216|     return float(b[int(math.floor(0.01 * b.size + EPS))])
  217| 
  218| 
  219| def point_metrics(dev, run):
  220|     out = {}
  221|     for s in TWO_CLASS_SETS:
  222|         out[f"{s}.auroc"] = float(roc_auc_score(dev.labels[s], dev.score[(run, s)]))
  223|     out["intent_score"] = (out["twins_prompt.auroc"] + out["dev_pair2.auroc"]) / 2.0
  224|     out["tc_val.auprc"] = float(average_precision_score(dev.labels["tc_val"], dev.score[(run, "tc_val")]))
  225|     benign = np.concatenate([dev.score[(run, s)][dev.labels[s] == 0] for s in ("tc_val", "aegis_val")])
  226|     out[f"{BENIGN_ONLY_SET}.flagged"] = float((dev.score[(run, BENIGN_ONLY_SET)] > strict_cut(benign)).mean())
  227|     return out
  228| 
  229| 
  230| # ----------------------------------------------------------------------------------------------- bootstrap
  231| def set_rng(name):
  232|     return np.random.default_rng([BASE_SEED, zlib.crc32(name.encode("utf-8"))])
  233| 
  234| 
  235| def counts(indices, n):
  236|     """Per-draw multiplicities of each item (rows = draws)."""
  237|     r = indices.shape[0]
  238|     flat = (indices + (np.arange(r) * n)[:, None]).ravel()
  239|     return np.bincount(flat, minlength=r * n).reshape(r, n)
  240| 
  241| 
  242| def bootstrap_auroc(benign, unsafe, benign_counts, unsafe_counts):
  243|     order = np.argsort(benign, kind="mergesort")
  244|     sorted_benign = benign[order]
  245|     left = np.searchsorted(sorted_benign, unsafe, "left")
  246|     right = np.searchsorted(sorted_benign, unsafe, "right")
  247|     cum = np.concatenate([np.zeros((benign_counts.shape[0], 1)), np.cumsum(benign_counts[:, order], 1)], 1)
  248|     below, at_or_below = cum[:, left], cum[:, right]
  249|     return (unsafe_counts * (below + 0.5 * (at_or_below - below))).sum(1) / (len(benign) * len(unsafe))
  250| 
  251| 
  252| def bootstrap_auprc(labels, scores, weights):
  253|     """Average precision with per-draw item multiplicities (rows = draws); row by row it equals sklearn's
  254|     average_precision_score(labels, scores, sample_weight=row)."""
  255|     order = np.argsort(-scores, kind="mergesort")
  256|     s_sorted, y_sorted = scores[order], labels[order].astype(float)
  257|     group_ends = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]
  258|     w = weights[:, order].astype(float)
  259|     tp = np.cumsum(w * y_sorted, 1)[:, group_ends]
  260|     fp = np.cumsum(w * (1 - y_sorted), 1)[:, group_ends]
  261|     precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
  262|     recall = tp / tp[:, -1:]
  263|     return (np.diff(np.concatenate([np.zeros((w.shape[0], 1)), recall], 1), axis=1) * precision).sum(1)
  264| 
  265| 
  266| def run_bootstrap(dev, draws, chunk=500):
  267|     runs = dev.all_runs()
  268|     per_run = {r: {} for r in runs}
  269|     benign_counts_for_threshold = {}
  270|     for s in TWO_CLASS_SETS:
  271|         y = dev.labels[s]
  272|         benign_idx, unsafe_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
  273|         nb, nu = len(benign_idx), len(unsafe_idx)
  274|         rng = set_rng(s)
  275|         benign_draws = np.vstack([np.arange(nb)[None], rng.integers(0, nb, (draws, nb))])
  276|         unsafe_draws = np.vstack([np.arange(nu)[None], rng.integers(0, nu, (draws, nu))])
  277|         auroc_parts = {r: [] for r in runs}
  278|         auprc_parts = {r: [] for r in runs}
  279|         kept_benign = []
  280|         for c in range(0, draws + 1, chunk):
  281|             cb, cu = counts(benign_draws[c:c + chunk], nb), counts(unsafe_draws[c:c + chunk], nu)
  282|             if s in ("tc_val", "aegis_val"):
  283|                 kept_benign.append(cb.astype(np.int32))
  284|             if s == "tc_val":
  285|                 weights = np.zeros((cb.shape[0], len(y)), np.int64)
  286|                 weights[:, benign_idx], weights[:, unsafe_idx] = cb, cu
  287|             for r in runs:
  288|                 sc = dev.score[(r, s)]
  289|                 auroc_parts[r].append(bootstrap_auroc(sc[benign_idx], sc[unsafe_idx], cb, cu))
  290|                 if s == "tc_val":
  291|                     auprc_parts[r].append(bootstrap_auprc(y, sc, weights))
  292|         for r in runs:
  293|             per_run[r][f"{s}.auroc"] = np.concatenate(auroc_parts[r])
  294|             if s == "tc_val":
  295|                 per_run[r]["tc_val.auprc"] = np.concatenate(auprc_parts[r])
  296|         if kept_benign:
  297|             benign_counts_for_threshold[s] = np.concatenate(kept_benign)
  298|         del benign_draws, unsafe_draws
  299|     n5 = len(dev.labels[BENIGN_ONLY_SET])
  300|     p5_counts = counts(np.vstack([np.arange(n5)[None], set_rng(BENIGN_ONLY_SET).integers(0, n5, (draws, n5))]), n5)
  301|     pooled_counts = np.hstack([benign_counts_for_threshold["tc_val"], benign_counts_for_threshold["aegis_val"]])
  302|     k = int(math.floor(0.01 * pooled_counts.shape[1] + EPS))
  303|     for r in runs:
  304|         benign = np.concatenate([dev.score[(r, s)][dev.labels[s] == 0] for s in ("tc_val", "aegis_val")])
  305|         order = np.argsort(-benign, kind="mergesort")
  306|         cum = np.cumsum(pooled_counts[:, order], 1)
  307|         cut = benign[order][np.argmax(cum > k, axis=1)]
  308|         p5 = dev.score[(r, BENIGN_ONLY_SET)]
  309|         per_run[r][f"{BENIGN_ONLY_SET}.flagged"] = (p5_counts * (p5[None, :] > cut[:, None])).sum(1) / n5
  310|         per_run[r]["intent_score"] = (per_run[r]["twins_prompt.auroc"] + per_run[r]["dev_pair2.auroc"]) / 2.0
  311|     per_variant = {}
  312|     cols = np.arange(draws + 1)
  313|     for v, rs in dev.runs_of.items():
  314|         n_runs = len(rs)
  315|         picks = np.vstack([np.arange(n_runs)[None], set_rng(f"training_runs:{v}").integers(0, n_runs, (draws, n_runs))])
  316|         per_variant[v] = {m: np.stack([per_run[r][m] for r in rs])[picks, cols[:, None]].mean(1) for m, _, _ in METRICS}
  317|     return per_run, per_variant
  318| 
  319| 
  320| # ----------------------------------------------------------------------------------------------- formatting
  321| def signed(x, dp=3):
  322|     s = f"{abs(x):.{dp}f}"
  323|     if float(s) == 0:
  324|         return f"+{s}" if x >= 0 else f"−{s}"
  325|     return ("+" if x > 0 else "−") + s
  326| 
  327| 
  328| def fmt_value(x, kind):
  329|     return f"{100 * x:.1f}%" if kind == "flagged" else f"{x:.3f}"
  330| 
  331| 
  332| def fmt_difference(d, kind):
  333|     lo, hi = np.nanpercentile(d[1:], [2.5, 97.5])
  334|     if kind == "flagged":
  335|         return f"{signed(100 * d[0], 1)} [{signed(100 * lo, 1)}, {signed(100 * hi, 1)}]"
  336|     return f"{signed(d[0])} [{signed(lo)}, {signed(hi)}]"
  337| 
  338| 
  339| # ----------------------------------------------------------------------------------------------- report
  340| def build_report(dev, points, per_variant):
  341|     trial_aegis_full = ", ".join(
  342|         fmt_value(float(roc_auc_score([int(x["label"]) for x in dev.trial_full_aegis[r]],
  343|                                       [x["score"] for x in dev.trial_full_aegis[r]])), "auc") for r in TRIAL_RUNS)
  344|     lines = [
  345|         "# Where the gain came from: development-set analysis", "",
  346|         "Generated by `dev_ablation.py` from the development scorecards in `scorecards/dev/` (labelled copy). Every "
  347|         "number below is recomputed by that script, which exits non-zero if this file or the card section differs "
  348|         "from what it computes.", "",
  349|         "## Read this first", "",
  350|         "- **Descriptive and after the fact.** This analysis was not planned before the training runs. It re-reads the "
  351|         "development scorecards produced when the release 2.0 recipe was chosen; no model was trained or scored for "
  352|         "it, and no test set is used. The differences describe these training runs on these sets; they are not "
  353|         "measured causal effects.",
  354|         "- **The recipe variants were not built to isolate one change each.** Only adding FalseReject and adding the "
  355|         "bag-of-words correction change one thing. Adding teacher labels also adds 100,000 prompts; adding the "
  356|         "alarming-but-benign data adds two pools at once; the base recipe against the release 1.0 trial checkpoints "
  357|         "changes data, starting weights and training settings at once. No comparison separates the warm start from "
  358|         "the data.",
  359|         "- **Same number of steps, different passes over the data.** Every training run of the recipe variants took "
  360|         "9,396 optimizer steps at batch 32, so a variant with fewer rows sees each row more often: without the "
  361|         "alarming-but-benign data 2.81 times, without FalseReject 2.21, the base recipe and the bag-of-words "
  362|         "variant 2.00, with teacher labels 1.20. This is part of every comparison except the bag-of-words one. The "
  363|         "release 1.0 trial checkpoints trained for up to 5 epochs on 13,132 rows at batch 16 and each kept the "
  364|         "checkpoint with the best ToxicChat validation AUPRC, which favours them on that split.",
  365|         "- **The held-back rows are in-domain for the variants that train on their source.** The WildGuardMix, "
  366|         "WildJailbreak and FalseReject held-back rows were held back from the same sources as the training pools. So "
  367|         "adding FalseReject, and adding the alarming-but-benign data, compare on the FalseReject held-back rows a "
  368|         "variant that trained on FalseReject with one that did not; adding the alarming-but-benign data likewise on "
  369|         "WildJailbreak's benign held-back rows; and every variant against the release 1.0 trial checkpoints on all "
  370|         "three (the trial checkpoints trained on none of these sources).",
  371|         "- **The paired prompts were generated in-house** for this project: 602 pairs of a harmful and a benign "
  372|         "prompt on the same topic. No training run saw them, but they are not an independent public benchmark.",
  373|         "- **The teachers were calibrated on two development sets.** Each teacher's score was Platt-scaled (two "
  374|         "parameters) on the labels of the ToxicChat validation split and the Aegis 2.0 validation set before it set "
  375|         "the teacher-label variant's targets, so that variant's figures on those two sets are not fully independent "
  376|         "of its training targets.",
  377|         "- **Small sets and few runs.** The ToxicChat validation split has 38 unsafe prompts, so its AUPRC is noisy. "
  378|         "The release 1.0 trial checkpoints are two runs with the same seed (they differ only by GPU "
  379|         "nondeterminism), not three seeds, so their run-to-run spread is understated.",
  380|         "- **The trial checkpoints' Aegis 2.0 validation set.** They were scored on an earlier 1,199-row version; 40 "
  381|         "rows that match test items were removed before the other runs were scored. They are evaluated here on the "
  382|         "1,159 shared rows.",
  383|         "",
  384|         "## The recipe variants", "",
  385|         "| Recipe variant | Run folders | Training data | Rows | Times each row is seen |",
  386|         "|---|---|---|---|---|",
  387|         "| release 1.0 trial checkpoints | A0_s13, A0_s13_repeat | release 1.0's recipe: ToxicChat train plus "
  388|         "relabelled response rows, warm-started from `cooperleong00/deberta-v3-large_toxicity-scorer` | 13,132 | up "
  389|         "to 5 (best ToxicChat validation checkpoint kept) |",
  390|         "| base recipe | R1_s13, R1_s42, R1_s99 | ToxicChat train (three copies), Aegis 2.0, WildGuardMix, "
  391|         "WildJailbreak (all four types), FalseReject; from the base model, no warm start | 150,316 | 2.00 |",
  392|         "| base recipe without the alarming-but-benign data | R0_s13, R0_s42, R0_s99 | the base recipe without "
  393|         "FalseReject and without WildJailbreak's benign rows | 107,031 | 2.81 |",
  394|         "| base recipe without FalseReject | R4_s13, R4_s42, R4_s99 | the base recipe without FalseReject | 136,131 | "
  395|         "2.21 |",
  396|         "| base recipe + bag-of-words correction | R2_s13, R2_s42, R2_s99 | the base recipe's rows; during training a "
  397|         "bag-of-words model's logit is added to the encoder's | 150,316 | 2.00 |",
  398|         "| base recipe + teacher labels (released; R3_s13 is the published weights) | R3_s13, R3_s42, R3_s99 | the "
  399|         "base recipe's rows with 2,182 teacher-disputed labels trained toward 0.5, plus 100,000 extra prompts "
  400|         "(30,000 WildJailbreak vanilla, 425 of them disputed; 30,000 HarmAug generated instructions, 40,000 WildChat-1M "
  401|         "first turns; teacher targets for the 70,000 without labels) | 250,316 | 1.20 |",
  402|         "| two-model averages | built from the runs above | no training: the mean logit margin of two training runs "
  403|         "(base + base with different seeds; base + bag-of-words; base + teacher labels; bag-of-words + teacher "
  404|         "labels) | n/a | n/a |",
  405|         "",
  406|         "The variant without the alarming-but-benign data trains on a WildJailbreak file (`p4_r0` in the training "
  407|         "code) that keeps only the `vanilla_harmful` and `adversarial_harmful` rows (29,100) and drops the 29,100 "
  408|         "`vanilla_benign` and `adversarial_benign` rows; that variant also has no FalseReject pool. The three teachers "
  409|         "are Qwen3Guard-Gen-8B, granite-guardian-4.1-8b and Shieldstral-1.0-3B.",
  410|         "",
  411|         "## Results by recipe variant (mean over training runs [lowest, highest])", "",
  412|         "AUROC unless marked. The FalseReject held-back rows are all benign, so they have no AUROC: their column is "
  413|         "the share flagged at the strict threshold (1% false positives on the benign rows of the ToxicChat "
  414|         "validation split and the Aegis 2.0 validation set; lower is better). The intent score is the mean AUROC on "
  415|         "the paired prompts and on the PHTest, HarmfulQA and CatQA set.", ""]
  416|     lines += ["| Recipe variant | Training runs | " + " | ".join(lab for _, lab, _ in METRICS) + " |",
  417|               "|" + "---|" * (len(METRICS) + 2)]
  418|     for v in DISPLAY_ORDER:
  419|         rs = dev.runs_of[v]
  420|         cells = []
  421|         for m, _, kind in METRICS:
  422|             vals = [points[r][m] for r in rs]
  423|             cells.append(f"{fmt_value(np.mean(vals), kind)} [{fmt_value(min(vals), kind)}, {fmt_value(max(vals), kind)}]")
  424|         lines.append(f"| {plain_name(v)} | {len(rs)} | " + " | ".join(cells) + " |")
  425|     lines += ["", "Every training run (two-model averages: the two run folders averaged):", "",
  426|               "| Recipe variant | Run folder(s) | " + " | ".join(lab for _, lab, _ in METRICS) + " |",
  427|               "|" + "---|" * (len(METRICS) + 2)]
  428|     for v in DISPLAY_ORDER:
  429|         for r in dev.runs_of[v]:
  430|             lines.append(f"| {plain_name(v)} | {r} | " +
  431|                          " | ".join(fmt_value(points[r][m], kind) for m, _, kind in METRICS) + " |")
  432|     lines += ["", "Missing training runs: none. Every recipe variant has all three training runs (seeds 13, 42 and "
  433|               "99) scored on all seven development sets; the release 1.0 trial checkpoints have both their runs "
  434|               "scored on all seven (on the Aegis 2.0 validation set, the 1,159 shared rows). Their Aegis 2.0 "
  435|               f"validation AUROC as first recorded, on the 1,199-row version: {trial_aegis_full}. Their intent score, "
  436|               "as first recorded, is the same as here.", ""]
  437|     lines += ["## Comparisons (difference of variant means [95% paired-bootstrap interval])", "",
  438|               "AUROC and AUPRC differences are in AUROC units; the FalseReject held-back rows in percentage points of "
  439|               "rows flagged (negative = fewer false flags). An interval that includes zero is a tie at this sample "
  440|               "size.", ""]
  441|     lines += ["| Comparison | " + " | ".join(lab for _, lab, _ in METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
  442|     for name, x, y, _ in CONTRASTS:
  443|         lines.append(f"| {name} | " + " | ".join(
  444|             fmt_difference(per_variant[x][m] - per_variant[y][m], kind) for m, _, kind in METRICS) + " |")
  445|     lines += ["", "What each comparison changes (first variant minus second):", ""]
  446|     for name, x, y, what in CONTRASTS:
  447|         lines.append(f"- **{name}** ({plain_name(x)} minus {plain_name(y)}): {what}.")
  448|     lines += ["", "## Method", "",
  449|               "- Point estimates: AUROC (sklearn `roc_auc_score`) and AUPRC (`average_precision_score`) on "
  450|               "P(unsafe); a variant's value is the mean over its training runs. A two-model average's score is the "
  451|               "sigmoid of the two runs' mean logit margin.",
  452|               "- Intervals: 10,000 paired bootstrap draws. Benign and unsafe items are resampled separately within "
  453|               "each set, with the same item draws for every training run; each set has its own stream, "
  454|               "`numpy.random.default_rng([13, zlib.crc32(set file name)])`. Inside each draw a variant's training "
  455|               "runs are resampled with replacement (stream `training_runs:<variant key>`). For the FalseReject "
  456|               "held-back rows the strict threshold is recomputed in every draw from that draw's benign rows of the "
  457|               "ToxicChat validation split and the Aegis 2.0 validation set.",
  458|               "- An interval is the 2.5th to 97.5th percentile of the 10,000 draws of the difference.", ""]
  459|     return "\n".join(lines)
  460| 
  461| 
  462| # ----------------------------------------------------------------------------------------------- card section
  463| def card_checks(per_run, per_variant, runs_of):
  464|     """(card wording, value printed on the card, recomputed value). Edit together with the card."""
  465|     def diff(x, y, m, kind="auc"):
  466|         return fmt_difference(per_variant[x][m] - per_variant[y][m], kind)
  467| 
  468|     def value(v, m):
  469|         return f"{per_variant[v][m][0]:.3f}"
  470| 
  471|     return [
  472|         ("Intent score of the release 1.0 trial checkpoints (mean of their two runs)", "0.715",
  473|          value(TRIAL[0], "intent_score")),
  474|         ("Intent score of release 2.0's recipe (mean of three training runs)", "0.884",
  475|          value("teacher_labels", "intent_score")),
  476|         ("Release 1.0's recipe to the base recipe (base recipe minus trial checkpoints) [95% CI]",
  477|          "+0.128 [+0.088, +0.170]", diff("base", TRIAL[0], "intent_score")),
  478|         ("Teacher labels, added with 100,000 extra prompts: intent score [95% CI]", "+0.041 [+0.029, +0.051]",
  479|          diff("teacher_labels", "base", "intent_score")),
  480|         ("Same step on ToxicChat validation AUPRC: a tie [95% CI]", "+0.032 [−0.009, +0.080]",
  481|          diff("teacher_labels", "base", "tc_val.auprc")),
  482|         ("A bag-of-words correction lowered the intent score [95% CI]", "−0.053 [−0.072, −0.034]",
  483|          diff("bag_of_words", "base", "intent_score")),
  484|         ("The alarming-but-benign data lowered it too [95% CI]", "−0.054 [−0.067, −0.041]",
  485|          diff("base", "no_alarming_benign", "intent_score")),
  486|         ("Same step on ToxicChat validation AUPRC, with the data minus without [95% CI]", "+0.072 [−0.001, +0.150]",
  487|          diff("base", "no_alarming_benign", "tc_val.auprc")),
  488|         ("Intent score of the variant without the alarming-but-benign data (mean of three runs)", "0.897",
  489|          value("no_alarming_benign", "intent_score")),
  490|         ("That variant trails the base recipe's ToxicChat validation AUPRC by (the selection rule's check)", "0.072",
  491|          f"{per_variant['base']['tc_val.auprc'][0] - per_variant['no_alarming_benign']['tc_val.auprc'][0]:.3f}"),
  492|         ("Base recipe's run-to-run range in ToxicChat validation AUPRC", "0.055",
  493|          f"{np.ptp([per_run[r]['tc_val.auprc'][0] for r in runs_of['base']]):.3f}"),
  494|     ]
  495| 
  496| 
  497| # ----------------------------------------------------------------------------------------------- main
  498| def main():
  499|     ap = argparse.ArgumentParser()
  500|     ap.add_argument("bundle")
  501|     ap.add_argument("--quick", action="store_true")
  502|     ap.add_argument("--write-report", action="store_true")
  503|     args = ap.parse_args()
  504|     draws = 1000 if args.quick else 10000
  505|     dev = DevScorecards(args.bundle)
  506|     points = {r: point_metrics(dev, r) for r in dev.all_runs()}
  507|     per_run, per_variant = run_bootstrap(dev, draws)
  508|     mismatches = 0
  509|     for r in dev.all_runs():   # draw 0 is the original sample: it must equal the direct computation
  510|         for m, _, _ in METRICS:
  511|             if abs(per_run[r][m][0] - points[r][m]) > 1e-9:
  512|                 print(f"INTERNAL MISMATCH {r} {m}: bootstrap {per_run[r][m][0]!r} != direct {points[r][m]!r}")
  513|                 mismatches += 1
  514|     print("## Card section: Where the gain came from (development sets)")
  515|     checks = card_checks(per_run, per_variant, dev.runs_of)
  516|     for wording, card, got in checks:
  517|         ok = card == got
  518|         mismatches += not ok
  519|         print(f"  [{'ok ' if ok else 'BAD'}] {wording}: card {card} | recomputed {got}")
  520|     report = build_report(dev, points, per_variant) + "\n"
  521|     path = pathlib.Path(args.bundle) / "reports" / "dev_ablation.md"
  522|     if args.write_report:
  523|         path.write_text(report)
  524|         print(f"wrote {path}")
  525|     else:
  526|         have = path.read_text().splitlines() if path.exists() else []
  527|         want = report.splitlines()
  528|         differing = [(i + 1, h, w) for i, (h, w) in enumerate(zip(have, want)) if h != w]
  529|         if len(have) != len(want):
  530|             differing.append((min(len(have), len(want)) + 1, f"{len(have)} lines", f"{len(want)} lines"))
  531|         for line_no, h, w in differing[:10]:
  532|             print(f"  [BAD] reports/dev_ablation.md line {line_no}:\n        bundle:     {h}\n        recomputed: {w}")
  533|         print(f"## reports/dev_ablation.md: {len(want)} lines recomputed, {len(differing)} differing")
  534|         mismatches += len(differing)
  535|     print(f"\n{len(checks)} card figures and the full report checked; {mismatches} mismatches.")
  536|     sys.exit(1 if mismatches else 0)
  537| 
  538| 
  539| if __name__ == "__main__":
  540|     main()
```

## `evidence/latency/time_encoder.py`

```python
    1| """time_encoder.py: the timing code that produced latency/release_2_A10_fp32.json.
    2| 
    3| These are the functions exactly as they ran, on Modal, on the published weights (model.safetensors sha256
    4| e688e56c...). The Modal launcher and the dataset loader (`baselines_scorecard`) are not included, so this
    5| file documents the method rather than running on its own.
    6| 
    7| Method: FP32, batch size 1. Inputs are OR-Bench-Hard prompts joined and cut to exactly 128 or 512 tokens
    8| (special tokens included). One cold call is recorded separately, then 100 warm-up requests, then 1,000 timed
    9| requests per length. Each time is one forward pass, from tokenised tensors already on the GPU to the logits,
   10| with torch.cuda.synchronize() before and after; tokenisation is not timed. Percentiles are nearest-rank.
   11| Static-batch throughput (batches 1-64, 50 timed batches each) is recorded alongside.
   12| Environment: torch 2.12.1, transformers 5.12.1, Python 3.11; the GPU reported itself as "NVIDIA A10".
   13| """
   14| import time
   15| 
   16| LENGTHS = (128, 512)
   17| BATCHES = (1, 8, 16, 32, 64)
   18| WARMUP, TIMED, TIMED_BATCHES = 100, 1000, 50
   19| P95_TARGET_MS = 100.0
   20| 
   21| 
   22| def _pct(xs, q):
   23|     xs = sorted(xs)
   24|     return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)
   25| 
   26| 
   27| def _summary(ms):
   28|     return {"n": len(ms), "p50": _pct(ms, .5), "p95": _pct(ms, .95), "p99": _pct(ms, .99),
   29|             "mean": round(sum(ms) / len(ms), 2)}
   30| 
   31| 
   32| def make_inputs(tok, n_tokens, count):
   33|     """`count` distinct strings of exactly n_tokens content tokens from OR-Bench-Hard prompts."""
   34|     import baselines_scorecard as bs   # loads OR-Bench-Hard-1K prompts (not included here)
   35|     prompts = [it["text"] for it in bs.load_set("orbench_hard")]
   36|     ids_all = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
   37|     out, j = [], 0
   38|     while len(out) < count:
   39|         buf = []
   40|         while len(buf) < n_tokens:
   41|             buf += ids_all[j % len(ids_all)] + tok(" ", add_special_tokens=False)["input_ids"]
   42|             j += 1
   43|         out.append(tok.decode(buf[:n_tokens]))
   44|     return out
   45| 
   46| 
   47| def _sync(dev):
   48|     import torch
   49|     if str(dev).startswith("cuda"):
   50|         torch.cuda.synchronize()
   51| 
   52| 
   53| def bench_encoder(key, dtype_name, dev="cuda"):
   54|     import torch
   55|     from transformers import AutoModelForSequenceClassification, AutoTokenizer
   56|     import baselines_scorecard as bs
   57|     cfg = bs.get_cfg(key)
   58|     if cfg.get("path"):   # v6 checkpoint on the read-only volume (tokenizer alongside)
   59|         tok = AutoTokenizer.from_pretrained(cfg["path"])
   60|         model = AutoModelForSequenceClassification.from_pretrained(cfg["path"])
   61|     else:
   62|         # v1.3 peers: DuoGuard uses the card's Qwen2.5 tokenizer with pad = eos and no special tokens
   63|         tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", cfg["hub"]),
   64|                                             revision=cfg.get("tokenizer_rev", cfg["rev"]))
   65|         model = AutoModelForSequenceClassification.from_pretrained(cfg["hub"], revision=cfg["rev"])
   66|     if tok.pad_token is None:
   67|         tok.pad_token = tok.eos_token
   68|     model.config.pad_token_id = tok.pad_token_id
   69|     dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
   70|     model.to(dev, dtype=dtype).eval()
   71|     n_special = len(tok("x")["input_ids"]) - len(tok("x", add_special_tokens=False)["input_ids"])
   72|     res = {}
   73|     for n in LENGTHS:
   74|         texts = make_inputs(tok, n - n_special, TIMED + WARMUP + 64 * 2)   # specials -> exactly n
   75| 
   76|         def run(batch):
   77|             enc = tok(batch, return_tensors="pt", truncation=True, max_length=n,
   78|                       padding="max_length").to(dev)
   79|             _sync(dev)
   80|             t0 = time.perf_counter()
   81|             with torch.no_grad():
   82|                 model(**enc)
   83|             _sync(dev)
   84|             return (time.perf_counter() - t0) * 1000
   85| 
   86|         cold = run(texts[:1])
   87|         for t in texts[:WARMUP]:
   88|             run([t])
   89|         bs1 = [run([t]) for t in texts[WARMUP:WARMUP + TIMED]]
   90|         res[n] = {"cold_ms": round(cold, 1), "bs1": _summary(bs1), **_throughput(run, texts)}
   91|     return res
   92| 
   93| 
   94| def _throughput(run, texts):
   95|     table, best = {}, None
   96|     for b in BATCHES:
   97|         batch = texts[:b]
   98|         try:
   99|             for _ in range(5):
  100|                 run(batch)
  101|             ms = [run(batch) for _ in range(TIMED_BATCHES)]
  102|         except Exception as e:   # CUDA OOM: this and larger batches are infeasible here
  103|             if "out of memory" not in str(e).lower():
  104|                 raise
  105|             import torch
  106|             torch.cuda.empty_cache()
  107|             table[b] = {"oom": True}
  108|             break
  109|         p95 = _pct(ms, .95)
  110|         table[b] = {"p95_ms": p95, "items_per_s": round(b / (sum(ms) / len(ms) / 1000), 1)}
  111|         if p95 <= P95_TARGET_MS:
  112|             best = b
  113|     fit = [b for b in table if "items_per_s" in table[b]]
  114|     top = max(fit, key=lambda b: table[b]["items_per_s"]) if fit else None
  115|     return {"batches": table, "best_batch_at_p95_target": best,
  116|             "items_per_s_at_target": table[best]["items_per_s"] if best else None,
  117|             # unconstrained: best items/s over the batches that fit in memory (labelled)
  118|             "items_per_s_unconstrained": table[top]["items_per_s"] if top else None,
  119|             "unconstrained_batch": top}
```

## `evidence/latency/time_guard.py`

```python
    1| """time_guard.py: the timing code that produced latency/qwen3guard_gen_0.6b_A10G_bf16.json.
    2| 
    3| The function exactly as it ran, on Modal (gpu="A10G"), on Qwen/Qwen3Guard-Gen-0.6B @ fada3b2f655b89601929198343c94cd2f64d93cc
    4| in bf16, with the model's own chat template and greedy decoding. It shares its helpers and settings with
    5| time_encoder.py, which timed release 2.0 in the same harness. The Modal launcher and the scoring harness
    6| (`baselines_scorecard`, which loads the model and builds its prompt) are not included, so this file documents the
    7| method rather than running on its own.
    8| 
    9| Method: inputs are OR-Bench-Hard prompts joined and cut to exactly 128 or 512 tokens of prompt text, inside
   10| Qwen3Guard's template (417 and 781 tokens in total). Batch size 1: one cold call recorded separately, then 100 warm-up
   11| requests, then 1,000 timed requests per length. "decision" (time to verdict) stops generating as soon as the verdict
   12| word after "Safety:" has been produced, which is when a filter can act; "full" generates the whole output, up to 128
   13| new tokens. Tokenisation is not timed. Static-batch throughput is recorded alongside, for "decision".
   14| Environment: torch 2.8.0, transformers 4.57.6, Python 3.11; the GPU reported itself as "NVIDIA A10G" (release 2.0's
   15| container on the same Modal GPU class reported "NVIDIA A10").
   16| """
   17| import time
   18| 
   19| from time_encoder import LENGTHS, TIMED, WARMUP, _summary, _sync, _throughput, make_inputs
   20| 
   21| 
   22| def bench_generative_hf(key, dev="cuda"):
   23|     import torch
   24|     from transformers import StoppingCriteria, StoppingCriteriaList
   25|     import baselines_scorecard as bs   # loads the model and builds its chat prompt (not included here)
   26|     cfg = bs.MODELS[key]
   27|     tok, model = bs._load_generative(key, cfg, dev)
   28| 
   29|     class VerdictStop(StoppingCriteria):
   30|         def __init__(self, plen):
   31|             self.plen = plen
   32| 
   33|         def __call__(self, input_ids, scores, **kw):
   34|             texts = tok.batch_decode(input_ids[:, self.plen:], skip_special_tokens=True)
   35|             if "verdict" in cfg:   # stop once the verdict word after the field ("Safety:") is out
   36|                 import re
   37|                 alt = "|".join(sorted(bs.VERDICT_WORDS[cfg["verdict"]], key=len, reverse=True))
   38|                 pat = re.compile(cfg["field_prompt"] + r"(" + alt + r")\b", re.IGNORECASE)
   39|                 return torch.tensor([bool(pat.search(t)) for t in texts], device=input_ids.device)
   40|             return torch.tensor([("safe" in t) for t in texts], device=input_ids.device)
   41| 
   42|     res = {}
   43|     for n in LENGTHS:
   44|         contents = make_inputs(tok, n, TIMED + WARMUP + 64 * 2)
   45|         prompts = [bs.build_prompt(key, tok, {"text": c, "response": None}, [0]) for c in contents]
   46|         total_tokens = len(tok(prompts[0], add_special_tokens=False)["input_ids"])
   47| 
   48|         def runner(mode):
   49|             def run(batch):
   50|                 enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
   51|                 plen = enc["input_ids"].shape[1]
   52|                 kw = {"stopping_criteria": StoppingCriteriaList([VerdictStop(plen)])} \
   53|                     if mode == "decision" else {}
   54|                 _sync(dev)
   55|                 t0 = time.perf_counter()
   56|                 with torch.no_grad():
   57|                     model.generate(**enc, max_new_tokens=cfg.get("max_new_tokens", bs.MAX_NEW_TOKENS),
   58|                                    do_sample=False,
   59|                                    temperature=None, top_p=None, top_k=None,
   60|                                    pad_token_id=tok.pad_token_id, **kw)
   61|                 _sync(dev)
   62|                 return (time.perf_counter() - t0) * 1000
   63|             return run
   64| 
   65|         entry = {"total_input_tokens": total_tokens}
   66|         for mode in ("decision", "full"):
   67|             run = runner(mode)
   68|             cold = run(prompts[:1])
   69|             for p in prompts[:WARMUP]:
   70|                 run([p])
   71|             bs1 = [run([p]) for p in prompts[WARMUP:WARMUP + TIMED]]
   72|             entry[mode] = {"cold_ms": round(cold, 1), "bs1": _summary(bs1)}
   73|             if mode == "decision":
   74|                 entry[mode].update(_throughput(run, prompts))
   75|         res[n] = entry
   76|     return res
```

## `evidence/plot_frr_vs_recall.py`

```python
    1| #!/usr/bin/env python3
    2| """plot_frr_vs_recall.py: draw the model card's figure, frr_vs_recall.png, from this bundle's scorecards.
    3| 
    4| Harmless OR-Bench-Hard prompts wrongly blocked (false-refusal rate, 1,319 prompts) against harmful OR-Bench-Toxic
    5| prompts caught (recall, 655 prompts). Item labels for OR-Bench are carried in the scorecards, so this runs on the
    6| bundle as shipped (attach_labels.py is not needed).
    7| 
    8|     pip install numpy matplotlib
    9|     python3 plot_frr_vs_recall.py <bundle folder> [--out frr_vs_recall.png]
   10| 
   11| Curves (release 2.0's published weights, omni-moderation snapshot omni-moderation-2024-09-26, and GPT-4o
   12| gpt-4o-2024-11-20 with the project's guard prompt): for each k = 1..655 the threshold is the k-th highest harmful
   13| score; recall is the share of harmful prompts scoring at or above it (with tied scores, that can be more than k)
   14| and the false-refusal rate the share of harmless prompts at or above it. omni's score is the highest of its 13
   15| category scores; GPT-4o's is P(block), its probability of answering BLOCK.
   16| Bands: 95% bootstrap intervals for each curve's false-refusal rate at each of its thresholds, resampling the 1,319
   17| harmless prompts with the thresholds fixed. The draws are reproduce_card.py's OR-Bench harmless-prompt draws
   18| (numpy default_rng([13, zlib.crc32("orbench")]), the first 10,000 x 1,319 integers), the same for every curve, so
   19| a band read at a table threshold equals that table's interval.
   20| Points: release 2.0's three thresholds on its curve (0.90, the default 0.455 and the pre-screen 0.043, each
   21| labelled with its values); omni at its own `flagged` decision; GPT-4o at its own decision (P(block) >= 0.5); and
   22| release 2.0 at 0.455 with GPT-4o reviewing only what it flags (a prompt is blocked only if both flag it). The
   23| legend gives the comparison points' values.
   24| """
   25| import argparse
   26| import json
   27| import pathlib
   28| import zlib
   29| 
   30| import matplotlib
   31| matplotlib.use("Agg")
   32| import matplotlib.pyplot as plt  # noqa: E402
   33| import numpy as np  # noqa: E402
   34| 
   35| PUB = "classifier__v6_R3_s13"                               # release 2.0, published weights
   36| OMNI = "classifier__openai_omni-moderation-2024-09-26"
   37| GPT4O = "guard__gpt-4o-2024-11-20"
   38| T_BAL = 0.45514049134720697                                 # release 2.0's default (balanced) threshold
   39| T_SCREEN = 0.04342702474724448                              # release 2.0's pre-screen threshold
   40| B = 10000                                                   # bootstrap draws for the bands
   41| 
   42| # Palette: categorical slots 1-4 of the reference palette, validated in light mode on #fcfcfb (all checks pass;
   43| # aqua and yellow are below 3:1 contrast, so every point has its own marker shape, an outline and a legend entry
   44| # that gives its values).
   45| SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
   46| C_V2, C_OMNI, C_G, C_PAIR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
   47| 
   48| 
   49| def load(sc, model, split):
   50|     rows = json.loads((sc / f"{model}__{split}.json").read_text())["rows"]
   51|     return np.array([r["score"] for r in rows], float), np.array([int(r["decision"]) for r in rows], int)
   52| 
   53| 
   54| def curve(benign, unsafe):
   55|     order = np.sort(unsafe)[::-1]
   56|     thr = order[np.arange(len(unsafe))]
   57|     rec = np.array([(unsafe >= t).mean() for t in thr])
   58|     frr = np.array([(benign >= t).mean() for t in thr])
   59|     return rec, frr, thr
   60| 
   61| 
   62| def band(benign, thr, I, chunk=1000):
   63|     """95% interval of the false-refusal rate at each fixed threshold, over the harmless-prompt draws I."""
   64|     n = len(benign)
   65|     o = np.argsort(benign, kind="mergesort")
   66|     pos = np.searchsorted(benign[o], thr, "left")          # harmless prompts below each threshold
   67|     draws = []
   68|     for c in range(0, I.shape[0], chunk):
   69|         idx = I[c:c + chunk]
   70|         m = idx.shape[0]
   71|         cnt = np.bincount((idx + (np.arange(m) * n)[:, None]).ravel(), minlength=m * n).reshape(m, n)
   72|         cum = np.concatenate([np.zeros((m, 1)), np.cumsum(cnt[:, o], 1)], 1)
   73|         draws.append((n - cum[:, pos]) / n)
   74|     d = np.concatenate(draws)
   75|     return np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)
   76| 
   77| 
   78| def main():
   79|     ap = argparse.ArgumentParser()
   80|     ap.add_argument("bundle")
   81|     ap.add_argument("--out", default="frr_vs_recall.png")
   82|     a = ap.parse_args()
   83|     sc = pathlib.Path(a.bundle) / "scorecards" / "test"
   84| 
   85|     vb, _ = load(sc, PUB, "orbench_hard")
   86|     vu, _ = load(sc, PUB, "orbench_toxic")
   87|     ob, odb = load(sc, OMNI, "orbench_hard")
   88|     ou, odu = load(sc, OMNI, "orbench_toxic")
   89|     gsb, gb = load(sc, GPT4O, "orbench_hard")
   90|     gsu, gu = load(sc, GPT4O, "orbench_toxic")
   91|     assert ((gsb >= 0.5) == (gb == 1)).all() and ((gsu >= 0.5) == (gu == 1)).all()   # decision = P(block) >= 0.5
   92| 
   93|     rv, fv, tv = curve(vb, vu)
   94|     ro, fo, to = curve(ob, ou)
   95|     rg, fg, tg = curve(gsb, gsu)
   96|     # reproduce_card.py's OR-Bench stream: its first draws are the harmless prompts (B x 1,319), the same for every model
   97|     I = np.random.default_rng([13, zlib.crc32(b"orbench")]).integers(0, len(vb), (B, len(vb)))
   98|     bands = [band(vb, tv, I), band(ob, to, I), band(gsb, tg, I)]
   99| 
  100|     plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
  101|                          "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
  102|     fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=160)
  103|     fig.patch.set_facecolor(SURFACE)
  104|     ax.set_facecolor(SURFACE)
  105|     ax.grid(True, color=GRID, linewidth=0.8)
  106|     ax.set_axisbelow(True)
  107|     for s in ("top", "right"):
  108|         ax.spines[s].set_visible(False)
  109|     for s in ("left", "bottom"):
  110|         ax.spines[s].set_color(INK2)
  111| 
  112|     for (lo, hi), r, c in zip(bands, (rv, ro, rg), (C_V2, C_OMNI, C_G)):
  113|         ax.fill_between(100 * r, 100 * lo, 100 * hi, color=c, alpha=0.16, linewidth=0, step="post")
  114|     ax.plot(100 * rv, 100 * fv, color=C_V2, linewidth=2, label="release 2.0, published weights (curve, 95% band)")
  115|     ax.plot(100 * ro, 100 * fo, color=C_OMNI, linewidth=2, label="omni-moderation-2024-09-26 (curve, 95% band)")
  116|     ax.plot(100 * rg, 100 * fg, color=C_G, linewidth=2, linestyle=(0, (5, 2)),
  117|             label="GPT-4o as a guard, score P(block) (curve, 95% band)")
  118| 
  119|     def point(r, f, marker, color, size, text, edge=SURFACE):
  120|         ax.plot(100 * r, 100 * f, marker, markersize=size, color=color, markeredgecolor=edge, markeredgewidth=1.5,
  121|                 zorder=5, linestyle="none", label=f"{text}\n{100 * r:.1f}% caught, {100 * f:.1f}% wrongly blocked")
  122| 
  123|     # release 2.0's three thresholds, on its curve: label offsets (points) are placed by hand to clear the curves,
  124|     # the bands and the legend
  125|     for t, name, filled, xy, ha, sep in [(0.90, "0.90 (block)", False, (6, 42), "center", ", "),
  126|                                          (T_BAL, "0.455 (default)", True, (-12, 34), "right", ", "),
  127|                                          (T_SCREEN, "0.043 (pre-screen)", False, (-22, -104), "right", ",\n")]:
  128|         r, f = (vu >= t).mean(), (vb >= t).mean()
  129|         ax.plot(100 * r, 100 * f, "o", markersize=8.5, color=C_V2 if filled else SURFACE, markeredgecolor=C_V2,
  130|                 markeredgewidth=2, zorder=6, linestyle="none")
  131|         rel = {"center": (0.5, 0.0), "right": (1.0, 0.5)}[ha]
  132|         ax.annotate(f"{name}\n{100 * r:.1f}% caught{sep}{100 * f:.1f}% blocked", (100 * r, 100 * f), xytext=xy,
  133|                     textcoords="offset points", fontsize=8.4, color=INK, ha=ha, va="center",
  134|                     arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8, shrinkA=2, shrinkB=5, relpos=rel))
  135|     point(odu.mean(), odb.mean(), "s", C_OMNI, 8, "omni at its flagged decision")
  136|     point(gu.mean(), gb.mean(), "D", C_G, 8, "GPT-4o at its own decision", edge=INK2)
  137|     pu_, pb_ = (vu >= T_BAL) & (gu == 1), (vb >= T_BAL) & (gb == 1)
  138|     point(pu_.mean(), pb_.mean(), "*", C_PAIR, 15, "release 2.0 at 0.455, GPT-4o reviewing its flags", edge=INK2)
  139| 
  140|     ax.set_xlim(0, 100)
  141|     ax.set_ylim(0, 100)
  142|     ax.set_xlabel("Harmful prompts caught: recall on OR-Bench-Toxic (655 prompts), %")
  143|     ax.set_ylabel("Harmless prompts wrongly blocked:\nfalse-refusal rate on OR-Bench-Hard (1,319 prompts), %")
  144|     ax.set_title("Wrongly blocked vs caught on OR-Bench (lower is better)", loc="left", fontsize=12, color=INK)
  145|     leg = ax.legend(loc="upper left", frameon=True, framealpha=1, facecolor=SURFACE, edgecolor=GRID,
  146|                     fontsize=8.6, labelspacing=0.55, borderpad=0.6)
  147|     for t in leg.get_texts():
  148|         t.set_color(INK)
  149|     fig.tight_layout()
  150|     fig.savefig(a.out, facecolor=SURFACE)
  151|     above = int((fv > fo).sum())
  152|     g_below = int((fg < fv).sum())
  153|     lo, hi = band(vb, np.array([T_BAL]), I)                 # equals the card's OR-Bench interval at 0.455
  154|     print(f"harmless prompts wrongly blocked at 0.455: {100 * (vb >= T_BAL).mean():.1f}% [{100 * lo[0]:.1f}, {100 * hi[0]:.1f}]")
  155|     print(f"wrote {a.out}; matched-recall points where release 2.0 blocks more harmless prompts than omni: {above}; "
  156|           f"where GPT-4o blocks fewer than release 2.0: {g_below} of {len(fv)}")
  157| 
  158| 
  159| if __name__ == "__main__":
  160|     main()
```

## `evidence/reports/dev_ablation.md`

```markdown
    1| # Where the gain came from: development-set analysis
    2| 
    3| Generated by `dev_ablation.py` from the development scorecards in `scorecards/dev/` (labelled copy). Every number below is recomputed by that script, which exits non-zero if this file or the card section differs from what it computes.
    4| 
    5| ## Read this first
    6| 
    7| - **Descriptive and after the fact.** This analysis was not planned before the training runs. It re-reads the development scorecards produced when the release 2.0 recipe was chosen; no model was trained or scored for it, and no test set is used. The differences describe these training runs on these sets; they are not measured causal effects.
    8| - **The recipe variants were not built to isolate one change each.** Only adding FalseReject and adding the bag-of-words correction change one thing. Adding teacher labels also adds 100,000 prompts; adding the alarming-but-benign data adds two pools at once; the base recipe against the release 1.0 trial checkpoints changes data, starting weights and training settings at once. No comparison separates the warm start from the data.
    9| - **Same number of steps, different passes over the data.** Every training run of the recipe variants took 9,396 optimizer steps at batch 32, so a variant with fewer rows sees each row more often: without the alarming-but-benign data 2.81 times, without FalseReject 2.21, the base recipe and the bag-of-words variant 2.00, with teacher labels 1.20. This is part of every comparison except the bag-of-words one. The release 1.0 trial checkpoints trained for up to 5 epochs on 13,132 rows at batch 16 and each kept the checkpoint with the best ToxicChat validation AUPRC, which favours them on that split.
   10| - **The held-back rows are in-domain for the variants that train on their source.** The WildGuardMix, WildJailbreak and FalseReject held-back rows were held back from the same sources as the training pools. So adding FalseReject, and adding the alarming-but-benign data, compare on the FalseReject held-back rows a variant that trained on FalseReject with one that did not; adding the alarming-but-benign data likewise on WildJailbreak's benign held-back rows; and every variant against the release 1.0 trial checkpoints on all three (the trial checkpoints trained on none of these sources).
   11| - **The paired prompts were generated in-house** for this project: 602 pairs of a harmful and a benign prompt on the same topic. No training run saw them, but they are not an independent public benchmark.
   12| - **The teachers were calibrated on two development sets.** Each teacher's score was Platt-scaled (two parameters) on the labels of the ToxicChat validation split and the Aegis 2.0 validation set before it set the teacher-label variant's targets, so that variant's figures on those two sets are not fully independent of its training targets.
   13| - **Small sets and few runs.** The ToxicChat validation split has 38 unsafe prompts, so its AUPRC is noisy. The release 1.0 trial checkpoints are two runs with the same seed (they differ only by GPU nondeterminism), not three seeds, so their run-to-run spread is understated.
   14| - **The trial checkpoints' Aegis 2.0 validation set.** They were scored on an earlier 1,199-row version; 40 rows that match test items were removed before the other runs were scored. They are evaluated here on the 1,159 shared rows.
   15| 
   16| ## The recipe variants
   17| 
   18| | Recipe variant | Run folders | Training data | Rows | Times each row is seen |
   19| |---|---|---|---|---|
   20| | release 1.0 trial checkpoints | A0_s13, A0_s13_repeat | release 1.0's recipe: ToxicChat train plus relabelled response rows, warm-started from `cooperleong00/deberta-v3-large_toxicity-scorer` | 13,132 | up to 5 (best ToxicChat validation checkpoint kept) |
   21| | base recipe | R1_s13, R1_s42, R1_s99 | ToxicChat train (three copies), Aegis 2.0, WildGuardMix, WildJailbreak (all four types), FalseReject; from the base model, no warm start | 150,316 | 2.00 |
   22| | base recipe without the alarming-but-benign data | R0_s13, R0_s42, R0_s99 | the base recipe without FalseReject and without WildJailbreak's benign rows | 107,031 | 2.81 |
   23| | base recipe without FalseReject | R4_s13, R4_s42, R4_s99 | the base recipe without FalseReject | 136,131 | 2.21 |
   24| | base recipe + bag-of-words correction | R2_s13, R2_s42, R2_s99 | the base recipe's rows; during training a bag-of-words model's logit is added to the encoder's | 150,316 | 2.00 |
   25| | base recipe + teacher labels (released; R3_s13 is the published weights) | R3_s13, R3_s42, R3_s99 | the base recipe's rows with 2,182 teacher-disputed labels trained toward 0.5, plus 100,000 extra prompts (30,000 WildJailbreak vanilla, 425 of them disputed; 30,000 HarmAug generated instructions, 40,000 WildChat-1M first turns; teacher targets for the 70,000 without labels) | 250,316 | 1.20 |
   26| | two-model averages | built from the runs above | no training: the mean logit margin of two training runs (base + base with different seeds; base + bag-of-words; base + teacher labels; bag-of-words + teacher labels) | n/a | n/a |
   27| 
   28| The variant without the alarming-but-benign data trains on a WildJailbreak file (`p4_r0` in the training code) that keeps only the `vanilla_harmful` and `adversarial_harmful` rows (29,100) and drops the 29,100 `vanilla_benign` and `adversarial_benign` rows; that variant also has no FalseReject pool. The three teachers are Qwen3Guard-Gen-8B, granite-guardian-4.1-8b and Shieldstral-1.0-3B.
   29| 
   30| ## Results by recipe variant (mean over training runs [lowest, highest])
   31| 
   32| AUROC unless marked. The FalseReject held-back rows are all benign, so they have no AUROC: their column is the share flagged at the strict threshold (1% false positives on the benign rows of the ToxicChat validation split and the Aegis 2.0 validation set; lower is better). The intent score is the mean AUROC on the paired prompts and on the PHTest, HarmfulQA and CatQA set.
   33| 
   34| | Recipe variant | Training runs | intent score | paired prompts (generated in-house) | PHTest, HarmfulQA and CatQA set | ToxicChat validation split | Aegis 2.0 validation set | WildGuardMix held-back rows | WildJailbreak held-back rows | FalseReject held-back rows flagged at the strict threshold | ToxicChat validation AUPRC |
   35| |---|---|---|---|---|---|---|---|---|---|---|
   36| | release 1.0 trial checkpoints | 2 | 0.715 [0.685, 0.744] | 0.816 [0.800, 0.832] | 0.614 [0.570, 0.657] | 0.988 [0.985, 0.991] | 0.865 [0.858, 0.872] | 0.826 [0.823, 0.830] | 0.745 [0.725, 0.765] | 2.4% [1.6%, 3.2%] | 0.899 [0.898, 0.901] |
   37| | base recipe | 3 | 0.843 [0.835, 0.853] | 0.938 [0.936, 0.939] | 0.748 [0.731, 0.766] | 0.985 [0.980, 0.993] | 0.939 [0.934, 0.944] | 0.995 [0.995, 0.995] | 0.999 [0.999, 0.999] | 0.2% [0.2%, 0.2%] | 0.907 [0.880, 0.935] |
   38| | base recipe without the alarming-but-benign data | 3 | 0.897 [0.888, 0.904] | 0.955 [0.944, 0.961] | 0.840 [0.831, 0.848] | 0.965 [0.959, 0.975] | 0.936 [0.934, 0.939] | 0.993 [0.992, 0.994] | 0.996 [0.996, 0.996] | 3.6% [1.1%, 7.7%] | 0.835 [0.825, 0.852] |
   39| | base recipe without FalseReject | 3 | 0.875 [0.857, 0.884] | 0.958 [0.949, 0.965] | 0.791 [0.765, 0.809] | 0.978 [0.975, 0.980] | 0.937 [0.935, 0.939] | 0.994 [0.993, 0.994] | 1.000 [0.999, 1.000] | 1.7% [0.9%, 2.5%] | 0.874 [0.858, 0.897] |
   40| | base recipe + bag-of-words correction | 3 | 0.790 [0.776, 0.807] | 0.924 [0.915, 0.935] | 0.656 [0.637, 0.680] | 0.954 [0.944, 0.962] | 0.850 [0.845, 0.855] | 0.745 [0.738, 0.750] | 0.711 [0.698, 0.720] | 0.3% [0.2%, 0.5%] | 0.725 [0.705, 0.758] |
   41| | base recipe + teacher labels (released) | 3 | 0.884 [0.879, 0.887] | 0.972 [0.970, 0.974] | 0.796 [0.788, 0.804] | 0.990 [0.984, 0.994] | 0.934 [0.932, 0.936] | 0.995 [0.994, 0.996] | 0.998 [0.998, 0.998] | 0.2% [0.0%, 0.5%] | 0.939 [0.934, 0.947] |
   42| | two-model average: base recipe, two seeds | 3 | 0.851 [0.847, 0.855] | 0.945 [0.943, 0.945] | 0.758 [0.749, 0.766] | 0.988 [0.985, 0.991] | 0.944 [0.941, 0.947] | 0.996 [0.996, 0.996] | 0.999 [0.999, 0.999] | 0.2% [0.2%, 0.2%] | 0.928 [0.919, 0.934] |
   43| | two-model average: base recipe and + bag-of-words correction | 3 | 0.839 [0.828, 0.848] | 0.949 [0.944, 0.953] | 0.729 [0.713, 0.743] | 0.981 [0.977, 0.989] | 0.914 [0.911, 0.916] | 0.977 [0.974, 0.980] | 0.992 [0.991, 0.992] | 0.2% [0.2%, 0.2%] | 0.849 [0.840, 0.859] |
   44| | two-model average: base recipe and + teacher labels | 3 | 0.875 [0.872, 0.878] | 0.961 [0.958, 0.964] | 0.788 [0.782, 0.798] | 0.991 [0.989, 0.995] | 0.942 [0.939, 0.944] | 0.996 [0.995, 0.997] | 0.999 [0.999, 0.999] | 0.2% [0.0%, 0.2%] | 0.939 [0.929, 0.953] |
   45| | two-model average: + bag-of-words correction and + teacher labels | 3 | 0.870 [0.866, 0.877] | 0.967 [0.962, 0.970] | 0.774 [0.767, 0.784] | 0.986 [0.984, 0.988] | 0.915 [0.913, 0.917] | 0.980 [0.977, 0.982] | 0.992 [0.991, 0.994] | 0.2% [0.2%, 0.2%] | 0.878 [0.860, 0.907] |
   46| 
   47| Every training run (two-model averages: the two run folders averaged):
   48| 
   49| | Recipe variant | Run folder(s) | intent score | paired prompts (generated in-house) | PHTest, HarmfulQA and CatQA set | ToxicChat validation split | Aegis 2.0 validation set | WildGuardMix held-back rows | WildJailbreak held-back rows | FalseReject held-back rows flagged at the strict threshold | ToxicChat validation AUPRC |
   50| |---|---|---|---|---|---|---|---|---|---|---|
   51| | release 1.0 trial checkpoints | A0_s13 | 0.744 | 0.832 | 0.657 | 0.991 | 0.872 | 0.830 | 0.765 | 3.2% | 0.898 |
   52| | release 1.0 trial checkpoints | A0_s13_repeat | 0.685 | 0.800 | 0.570 | 0.985 | 0.858 | 0.823 | 0.725 | 1.6% | 0.901 |
   53| | base recipe | R1_s13 | 0.853 | 0.939 | 0.766 | 0.993 | 0.941 | 0.995 | 0.999 | 0.2% | 0.935 |
   54| | base recipe | R1_s42 | 0.842 | 0.936 | 0.748 | 0.981 | 0.934 | 0.995 | 0.999 | 0.2% | 0.880 |
   55| | base recipe | R1_s99 | 0.835 | 0.938 | 0.731 | 0.980 | 0.944 | 0.995 | 0.999 | 0.2% | 0.906 |
   56| | base recipe without the alarming-but-benign data | R0_s13 | 0.904 | 0.960 | 0.848 | 0.975 | 0.939 | 0.993 | 0.996 | 1.1% | 0.828 |
   57| | base recipe without the alarming-but-benign data | R0_s42 | 0.888 | 0.944 | 0.831 | 0.962 | 0.935 | 0.994 | 0.996 | 7.7% | 0.852 |
   58| | base recipe without the alarming-but-benign data | R0_s99 | 0.901 | 0.961 | 0.840 | 0.959 | 0.934 | 0.992 | 0.996 | 2.1% | 0.825 |
   59| | base recipe without FalseReject | R4_s13 | 0.882 | 0.965 | 0.800 | 0.980 | 0.935 | 0.994 | 1.000 | 2.5% | 0.858 |
   60| | base recipe without FalseReject | R4_s42 | 0.857 | 0.949 | 0.765 | 0.978 | 0.939 | 0.993 | 1.000 | 0.9% | 0.867 |
   61| | base recipe without FalseReject | R4_s99 | 0.884 | 0.960 | 0.809 | 0.975 | 0.937 | 0.994 | 0.999 | 1.8% | 0.897 |
   62| | base recipe + bag-of-words correction | R2_s13 | 0.807 | 0.935 | 0.680 | 0.944 | 0.850 | 0.750 | 0.720 | 0.5% | 0.712 |
   63| | base recipe + bag-of-words correction | R2_s42 | 0.787 | 0.924 | 0.650 | 0.957 | 0.855 | 0.746 | 0.716 | 0.2% | 0.705 |
   64| | base recipe + bag-of-words correction | R2_s99 | 0.776 | 0.915 | 0.637 | 0.962 | 0.845 | 0.738 | 0.698 | 0.2% | 0.758 |
   65| | base recipe + teacher labels (released) | R3_s13 | 0.887 | 0.970 | 0.804 | 0.984 | 0.934 | 0.994 | 0.998 | 0.0% | 0.935 |
   66| | base recipe + teacher labels (released) | R3_s42 | 0.879 | 0.971 | 0.788 | 0.992 | 0.936 | 0.996 | 0.998 | 0.5% | 0.934 |
   67| | base recipe + teacher labels (released) | R3_s99 | 0.884 | 0.974 | 0.795 | 0.994 | 0.932 | 0.996 | 0.998 | 0.2% | 0.947 |
   68| | two-model average: base recipe, two seeds | R1_s13 + R1_s42 | 0.855 | 0.943 | 0.766 | 0.991 | 0.941 | 0.996 | 0.999 | 0.2% | 0.934 |
   69| | two-model average: base recipe, two seeds | R1_s42 + R1_s99 | 0.847 | 0.945 | 0.749 | 0.985 | 0.944 | 0.996 | 0.999 | 0.2% | 0.919 |
   70| | two-model average: base recipe, two seeds | R1_s99 + R1_s13 | 0.851 | 0.945 | 0.758 | 0.989 | 0.947 | 0.996 | 0.999 | 0.2% | 0.929 |
   71| | two-model average: base recipe and + bag-of-words correction | R1_s13 + R2_s13 | 0.848 | 0.953 | 0.743 | 0.989 | 0.916 | 0.974 | 0.991 | 0.2% | 0.859 |
   72| | two-model average: base recipe and + bag-of-words correction | R1_s42 + R2_s42 | 0.841 | 0.951 | 0.731 | 0.978 | 0.911 | 0.976 | 0.992 | 0.2% | 0.840 |
   73| | two-model average: base recipe and + bag-of-words correction | R1_s99 + R2_s99 | 0.828 | 0.944 | 0.713 | 0.977 | 0.914 | 0.980 | 0.992 | 0.2% | 0.849 |
   74| | two-model average: base recipe and + teacher labels | R1_s13 + R3_s13 | 0.878 | 0.958 | 0.798 | 0.995 | 0.943 | 0.995 | 0.999 | 0.0% | 0.953 |
   75| | two-model average: base recipe and + teacher labels | R1_s42 + R3_s42 | 0.874 | 0.964 | 0.784 | 0.989 | 0.939 | 0.997 | 0.999 | 0.2% | 0.929 |
   76| | two-model average: base recipe and + teacher labels | R1_s99 + R3_s99 | 0.872 | 0.962 | 0.782 | 0.990 | 0.944 | 0.997 | 0.999 | 0.2% | 0.934 |
   77| | two-model average: + bag-of-words correction and + teacher labels | R2_s13 + R3_s13 | 0.877 | 0.970 | 0.784 | 0.988 | 0.913 | 0.977 | 0.991 | 0.2% | 0.860 |
   78| | two-model average: + bag-of-words correction and + teacher labels | R2_s42 + R3_s42 | 0.868 | 0.968 | 0.767 | 0.984 | 0.917 | 0.980 | 0.994 | 0.2% | 0.866 |
   79| | two-model average: + bag-of-words correction and + teacher labels | R2_s99 + R3_s99 | 0.866 | 0.962 | 0.770 | 0.985 | 0.914 | 0.982 | 0.992 | 0.2% | 0.907 |
   80| 
   81| Missing training runs: none. Every recipe variant has all three training runs (seeds 13, 42 and 99) scored on all seven development sets; the release 1.0 trial checkpoints have both their runs scored on all seven (on the Aegis 2.0 validation set, the 1,159 shared rows). Their Aegis 2.0 validation AUROC as first recorded, on the 1,199-row version: 0.875, 0.861. Their intent score, as first recorded, is the same as here.
   82| 
   83| ## Comparisons (difference of variant means [95% paired-bootstrap interval])
   84| 
   85| AUROC and AUPRC differences are in AUROC units; the FalseReject held-back rows in percentage points of rows flagged (negative = fewer false flags). An interval that includes zero is a tie at this sample size.
   86| 
   87| | Comparison | intent score | paired prompts (generated in-house) | PHTest, HarmfulQA and CatQA set | ToxicChat validation split | Aegis 2.0 validation set | WildGuardMix held-back rows | WildJailbreak held-back rows | FalseReject held-back rows flagged at the strict threshold | ToxicChat validation AUPRC |
   88| |---|---|---|---|---|---|---|---|---|---|
   89| | Adding teacher labels (with 100,000 extra prompts) | +0.041 [+0.029, +0.051] | +0.034 [+0.023, +0.045] | +0.047 [+0.028, +0.067] | +0.005 [−0.008, +0.018] | −0.006 [−0.013, +0.003] | +0.000 [−0.002, +0.003] | −0.001 [−0.002, +0.000] | +0.0 [−0.5, +0.5] | +0.032 [−0.009, +0.080] |
   90| | Adding FalseReject | −0.032 [−0.048, −0.013] | −0.020 [−0.033, −0.008] | −0.043 [−0.069, −0.014] | +0.007 [−0.004, +0.024] | +0.002 [−0.006, +0.010] | +0.001 [−0.001, +0.004] | −0.001 [−0.002, +0.000] | −1.5 [−9.6, +0.0] | +0.033 [−0.023, +0.101] |
   91| | Adding the alarming-but-benign data | −0.054 [−0.067, −0.041] | −0.017 [−0.030, −0.003] | −0.091 [−0.111, −0.072] | +0.020 [+0.002, +0.048] | +0.003 [−0.005, +0.011] | +0.002 [−0.000, +0.005] | +0.003 [−0.000, +0.007] | −3.4 [−12.8, −0.5] | +0.072 [−0.001, +0.150] |
   92| | Adding WildJailbreak's benign rows | −0.023 [−0.040, −0.007] | +0.003 [−0.010, +0.018] | −0.049 [−0.073, −0.027] | +0.012 [−0.003, +0.037] | +0.001 [−0.006, +0.008] | +0.001 [−0.002, +0.004] | +0.004 [+0.001, +0.007] | −1.9 [−8.8, +4.1] | +0.039 [−0.016, +0.098] |
   93| | Adding the bag-of-words correction | −0.053 [−0.072, −0.034] | −0.013 [−0.029, +0.002] | −0.093 [−0.122, −0.063] | −0.031 [−0.063, −0.007] | −0.090 [−0.108, −0.072] | −0.250 [−0.277, −0.224] | −0.288 [−0.313, −0.263] | +0.1 [−0.5, +0.9] | −0.182 [−0.291, −0.068] |
   94| | Base recipe against the release 1.0 trial checkpoints | +0.128 [+0.088, +0.170] | +0.122 [+0.091, +0.156] | +0.135 [+0.076, +0.194] | −0.003 [−0.020, +0.012] | +0.074 [+0.054, +0.097] | +0.168 [+0.147, +0.191] | +0.254 [+0.220, +0.289] | −2.2 [−4.7, −0.2] | +0.007 [−0.058, +0.075] |
   95| | Released recipe against the release 1.0 trial checkpoints | +0.169 [+0.131, +0.208] | +0.156 [+0.126, +0.188] | +0.182 [+0.128, +0.237] | +0.002 [−0.014, +0.016] | +0.069 [+0.049, +0.091] | +0.169 [+0.147, +0.191] | +0.253 [+0.220, +0.289] | −2.2 [−4.6, −0.1] | +0.040 [−0.016, +0.102] |
   96| 
   97| What each comparison changes (first variant minus second):
   98| 
   99| - **Adding teacher labels (with 100,000 extra prompts)** (base recipe + teacher labels (released) minus base recipe): two changes together. (1) 2,182 of the base recipe's WildGuardMix, WildJailbreak and FalseReject rows are trained toward 0.5 where the teachers confidently disputed the label. (2) 100,000 prompts are added: 30,000 more WildJailbreak vanilla rows (their own labels, unless disputed: 425 were, which with the 2,182 above makes the card's 2,607 disputed rows), 30,000 instructions from the HarmAug generated dataset and 40,000 WildChat-1M first turns (teacher targets only); 35,761 of the 100,000 are unsafe. 250,316 rows seen 1.20 times against 150,316 rows seen 2.00 times.
  100| - **Adding FalseReject** (base recipe minus base recipe without FalseReject): the FalseReject pool alone: 14,185 benign prompts. 150,316 rows seen 2.00 times against 136,131 seen 2.21 times.
  101| - **Adding the alarming-but-benign data** (base recipe minus base recipe without the alarming-but-benign data): a combined change: the FalseReject pool (14,185 rows) and WildJailbreak's benign rows (29,100: 14,550 vanilla_benign and 14,550 adversarial_benign) together. 150,316 rows seen 2.00 times against 107,031 seen 2.81 times.
  102| - **Adding WildJailbreak's benign rows** (base recipe without FalseReject minus base recipe without the alarming-but-benign data): WildJailbreak's benign rows alone (29,100), FalseReject absent from both. 136,131 rows seen 2.21 times against 107,031 seen 2.81 times. The variant without FalseReject was added after the first development results of the base recipe, the variant without the alarming-but-benign data and the bag-of-words variant had been seen.
  103| - **Adding the bag-of-words correction** (base recipe + bag-of-words correction minus base recipe): the bag-of-words correction alone: the same rows, seen the same number of times (2.00).
  104| - **Base recipe against the release 1.0 trial checkpoints** (base recipe minus release 1.0 trial checkpoints): everything between release 1.0's recipe and the base recipe at once: the training data, the starting weights (a toxicity-scorer checkpoint against the plain base model) and the training settings.
  105| - **Released recipe against the release 1.0 trial checkpoints** (base recipe + teacher labels (released) minus release 1.0 trial checkpoints): the two previous steps together: base recipe against the trial checkpoints, plus adding teacher labels.
  106| 
  107| ## Method
  108| 
  109| - Point estimates: AUROC (sklearn `roc_auc_score`) and AUPRC (`average_precision_score`) on P(unsafe); a variant's value is the mean over its training runs. A two-model average's score is the sigmoid of the two runs' mean logit margin.
  110| - Intervals: 10,000 paired bootstrap draws. Benign and unsafe items are resampled separately within each set, with the same item draws for every training run; each set has its own stream, `numpy.random.default_rng([13, zlib.crc32(set file name)])`. Inside each draw a variant's training runs are resampled with replacement (stream `training_runs:<variant key>`). For the FalseReject held-back rows the strict threshold is recomputed in every draw from that draw's benign rows of the ToxicChat validation split and the Aegis 2.0 validation set.
  111| - An interval is the 2.5th to 97.5th percentile of the 10,000 draws of the difference.
  112| 
```
