"""v6_data.py: source loaders for the v6 training pools, dev sets and decontamination references.

Runs on Modal only (downloads happen there). Every loader returns plain Python lists/DataFrames;
nothing here prints or persists text. Revisions are resolved once and recorded in the manifest.
"""
import csv
import gzip
import io
import json

import numpy as np
import pandas as pd

AEGIS = ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c")
WGM = "allenai/wildguardmix"
WJB = "allenai/wildjailbreak"
PHTEST = "furonghuang-lab/PHTest"
HQA = "declare-lab/HarmfulQA"
CATQA = "declare-lab/CategoricalHarmfulQA"
JBB = "JailbreakBench/JBB-Behaviors"
HARMAUG = "hbseong/HarmAug_generated_dataset"
WILDCHAT = "allenai/WildChat-1M"
SST = ("Bertievidgen/SimpleSafetyTests", "sst_test_cases.csv",
       "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5")
HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
                 "8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/"
                 "harmbench_behaviors_text_all.csv")
OPENAI_MOD_REPO = "openai/moderation-api-release"
OPENAI_MOD_FILE = "data/samples-1680.jsonl.gz"
OPENAI_MOD_CATS = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")
DEBERTA = "microsoft/deberta-v3-large"


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def revision(repo, repo_type="dataset"):
    info = _api().repo_info(repo, repo_type=repo_type)
    return info.sha


def hub_file(repo, fname, rev=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset", revision=rev)


def http_bytes(url):
    import requests
    r = requests.get(url, timeout=180, headers={"User-Agent": "v6-build"})
    r.raise_for_status()
    return r.content


def github_head(repo):
    import requests
    r = requests.get(f"https://api.github.com/repos/{repo}/commits/HEAD", timeout=60,
                     headers={"User-Agent": "v6-build", "Accept": "application/vnd.github+json"})
    r.raise_for_status()
    return r.json()["sha"]


def _summ(df, label_cols=()):
    out = {"rows": int(len(df)), "columns": list(map(str, df.columns))}
    for c in label_cols:
        if c in df.columns:
            out[f"counts_{c}"] = {str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()}
    return out


# ------------------------------------------------------------------------------------------- #
# loaders                                                                                       #
# ------------------------------------------------------------------------------------------- #
def load_aegis(split_file):
    data = json.load(open(hub_file(AEGIS[0], split_file, AEGIS[1]), encoding="utf-8"))
    return pd.DataFrame(data)


def load_wgm(config):
    from datasets import load_dataset
    rev = revision(WGM)
    ds = load_dataset(WGM, config, revision=rev)
    split = list(ds.keys())[0]
    return ds[split].to_pandas(), rev, split


def load_wjb():
    rev = revision(WJB)
    p = hub_file(WJB, "train/train.tsv", rev)
    df = pd.read_csv(p, sep="\t", keep_default_na=False, dtype=str, quoting=csv.QUOTE_MINIMAL)
    return df, rev


def load_hf_any(repo, config=None, rev=None):
    from datasets import load_dataset
    rev = rev or revision(repo)
    ds = load_dataset(repo, config, revision=rev) if config else load_dataset(repo, revision=rev)
    return {k: v.to_pandas() for k, v in ds.items()}, rev


def load_openai_mod():
    sha = github_head(OPENAI_MOD_REPO)
    raw = http_bytes(f"https://raw.githubusercontent.com/{OPENAI_MOD_REPO}/{sha}/{OPENAI_MOD_FILE}")
    rows = [json.loads(x) for x in gzip.decompress(raw).decode("utf-8").splitlines() if x.strip()]
    return rows, sha


def load_harmbench_std():
    rows = list(csv.DictReader(io.StringIO(http_bytes(HARMBENCH_URL).decode("utf-8"))))
    return [r["Behavior"] for r in rows if r["FunctionalCategory"] == "standard"]


def load_sst():
    with open(hub_file(*SST), newline="", encoding="utf-8") as f:
        return [r["prompt"] for r in csv.DictReader(f)]


# ------------------------------------------------------------------------------------------- #
# inspection                                                                                    #
# ------------------------------------------------------------------------------------------- #
def inspect_all():
    out = {}

    def guard(name, fn):
        try:
            out[name] = fn()
        except Exception as e:  # report and keep going
            out[name] = {"error": repr(e)[:500]}

    def aegis():
        from huggingface_hub import list_repo_files
        files = list_repo_files(AEGIS[0], repo_type="dataset", revision=AEGIS[1])
        r = {"files": files}
        for f in ("train.json", "validation.json", "test.json"):
            if f in files:
                r[f] = _summ(load_aegis(f), ["prompt_label"])
        return r

    def wgm():
        r = {}
        for c in ("wildguardtrain", "wildguardtest"):
            df, rev, split = load_wgm(c)
            r[c] = _summ(df, ["prompt_harm_label", "adversarial"]) | {"revision": rev, "split": split}
        return r

    def wjb():
        from huggingface_hub import list_repo_files
        rev = revision(WJB)
        files = list_repo_files(WJB, repo_type="dataset", revision=rev)
        df, _ = load_wjb()
        r = _summ(df, ["data_type"]) | {"revision": rev, "files": files}
        r["empty_vanilla"] = int((df["vanilla"].str.len() == 0).sum()) if "vanilla" in df else None
        r["empty_adversarial"] = (int((df["adversarial"].str.len() == 0).sum())
                                  if "adversarial" in df else None)
        return r

    def generic(repo, config=None, label_cols=()):
        def f():
            splits, rev = load_hf_any(repo, config)
            return {"revision": rev, "splits": {k: _summ(v, label_cols) for k, v in splits.items()}}
        return f

    def openai_mod():
        rows, sha = load_openai_mod()
        keys = sorted({k for r in rows for k in r})
        unsafe = sum(1 for r in rows if any(r.get(c) == 1 for c in OPENAI_MOD_CATS))
        return {"commit": sha, "rows": len(rows), "keys": keys, "unsafe_any_flag": unsafe}

    guard("aegis", aegis)
    guard("wildguardmix", wgm)
    guard("wildjailbreak", wjb)
    guard("phtest", generic(PHTEST, label_cols=("label", "harmfulness", "category")))
    guard("harmfulqa", generic(HQA, label_cols=("topic",)))
    guard("catqa", generic(CATQA, label_cols=("Category",)))
    guard("jbb", generic(JBB, "behaviors", label_cols=("Category",)))
    guard("openai_mod", openai_mod)
    guard("harmbench_std", lambda: {"rows": len(load_harmbench_std())})
    guard("sst", lambda: {"rows": len(load_sst())})
    guard("deberta_rev", lambda: revision(DEBERTA, "model"))
    guard("harmaug_cols", lambda: {"revision": revision(HARMAUG), "files": _api().list_repo_files(
        HARMAUG, repo_type="dataset")[:20]})
    return out
