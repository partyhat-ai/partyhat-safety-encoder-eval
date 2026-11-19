#!/usr/bin/env python3
"""attach_labels.py: rebuild the item labels stripped from the evidence bundle, from the public datasets.

The label-stripped bundle omits item labels for the ToxicChat, Aegis 2.0 and WildGuardTest test sets and for five
development sets (tc_val from ToxicChat, aegis_val from Aegis 2.0, dev_holdout_p3 from WildGuardMix,
dev_holdout_p4 from WildJailbreak and dev_holdout_p5 from FalseReject), so that no third-party labels are
redistributed. This script rebuilds them from the public datasets at the pinned revisions, with YOUR Hugging Face
access (accept the gated terms of allenai/wildguardmix and allenai/wildjailbreak yourself first), and verifies
every rebuilt label vector against the sha256 the bundle carries (reports/label_hashes.json). It refuses to write a
vector whose hash differs.

    pip install pandas scikit-learn huggingface_hub pyarrow
    python3 attach_labels.py <stripped bundle folder> <output folder>

The output folder is a copy of the bundle with labels attached; then run
    python3 reproduce_card.py <output folder>
    python3 dev_ablation.py <output folder>
"""
import csv
import hashlib
import json
import pathlib
import shutil
import sys

PINS = {
    "toxicchat": ("lmsys/toxic-chat", "29df8e4dba60e1f4af4b4075c0705c5b313548a8"),
    "aegis": ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"),
    "wildguardmix": ("allenai/wildguardmix", "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"),
    "wildjailbreak": ("allenai/wildjailbreak", "5ddc12a7894f842b0619b8e1c7ee496b198af009"),
    "falsereject": ("AmazonScience/FalseReject", "493ba967714ea54c6f01067e1f61e389cc2c9b3e"),
}
_CACHE = {}


def hub(key, fname):
    from huggingface_hub import hf_hub_download
    repo, rev = PINS[key]
    return hf_hub_download(repo, fname, repo_type="dataset", revision=rev)


def toxicchat(split):
    if split not in _CACHE:
        with open(hub("toxicchat", f"data/0124/toxic-chat_annotation_{split}.csv"), newline="", encoding="utf-8") as f:
            _CACHE[split] = list(csv.DictReader(f))
    return _CACHE[split]


def aegis(fname):
    if fname not in _CACHE:
        _CACHE[fname] = json.load(open(hub("aegis", fname), encoding="utf-8"))
    return _CACHE[fname]


def labels_toxicchat_test(rows):
    src = toxicchat("test")
    return [{"label": int(src[r["src"]]["toxicity"]), "jailbreaking": int(src[r["src"]]["jailbreaking"])} for r in rows]


def labels_aegis2_test_prompt(rows):
    data = aegis("test.json")
    out = []
    for r in rows:
        d = data[r["src"]]
        cats = [c.strip() for c in str(d.get("violated_categories") or "").split(",") if c.strip()]
        nc_only = cats == ["Needs Caution"]
        shipped = int(d["prompt_label"] == "unsafe")
        out.append({"label": 1 if nc_only else shipped, "label_shipped": shipped, "gt_nc_only": nc_only})
    return out


def labels_wildguardtest_prompt(rows):
    import pandas as pd
    if "wgt" not in _CACHE:
        _CACHE["wgt"] = pd.read_parquet(hub("wildguardmix", "test/wildguard_test.parquet"))
    df = _CACHE["wgt"]
    return [{"label": int(df.iloc[r["src"]]["prompt_harm_label"] == "harmful"),
             "adversarial": bool(df.iloc[r["src"]]["adversarial"])} for r in rows]


def labels_tc_val(rows):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    if "tcval" not in _CACHE:
        df = pd.DataFrame(toxicchat("train"))
        df["toxicity"] = df["toxicity"].astype(int)
        _, val = train_test_split(df, test_size=0.1, random_state=13, stratify=df["toxicity"])
        _CACHE["tcval"] = val.reset_index(drop=True)
    val = _CACHE["tcval"]
    return [{"label": int(val.iloc[int(str(r["src"]).split(":")[1])]["toxicity"])} for r in rows]


def labels_aegis_val(rows):
    data = {d["id"]: d for d in aegis("validation.json")}
    return [{"label": int(data[str(r["src"]).split(":", 1)[1]]["prompt_label"] == "unsafe")} for r in rows]


def labels_dev_holdout_p3(rows):
    import pandas as pd
    if "wgm" not in _CACHE:
        _CACHE["wgm"] = pd.read_parquet(hub("wildguardmix", "train/wildguard_train.parquet"))
    df = _CACHE["wgm"]
    return [{"label": int(df.iloc[int(str(r["src"]).split(":")[1])]["prompt_harm_label"] == "harmful")} for r in rows]


def labels_dev_holdout_p4(rows):
    import pandas as pd
    if "wjb" not in _CACHE:
        _CACHE["wjb"] = pd.read_csv(hub("wildjailbreak", "train/train.tsv"), sep="\t", keep_default_na=False,
                                    usecols=["data_type"])
    df = _CACHE["wjb"]
    return [{"label": int(str(df.iloc[int(str(r["src"]).split(":")[1])]["data_type"]).endswith("_harmful"))} for r in rows]


def labels_dev_holdout_p5(rows):
    """Every dev_holdout_p5 row is a prompt from FalseReject's train split, which the dataset defines as benign
    prompts that look harmful (its card at the pinned revision: "A benign but high-risk-looking prompt"). The label
    is therefore 0 for every row; the card is fetched to confirm the pinned revision, and the hash check below
    confirms the vector."""
    card = pathlib.Path(hub("falsereject", "README.md")).read_text(encoding="utf-8")
    if "benign but high-risk-looking prompt" not in card:
        raise SystemExit("FalseReject card at the pinned revision no longer defines its train prompts as benign")
    return [{"label": 0} for _ in rows]


BUILDERS = {"toxicchat_test": labels_toxicchat_test, "aegis2_test_prompt": labels_aegis2_test_prompt,
            "wildguardtest_prompt": labels_wildguardtest_prompt, "tc_val": labels_tc_val,
            "aegis_val": labels_aegis_val, "dev_holdout_p3": labels_dev_holdout_p3,
            "dev_holdout_p4": labels_dev_holdout_p4, "dev_holdout_p5": labels_dev_holdout_p5}


def label_sha(rows):
    return hashlib.sha256(json.dumps([int(r["label"]) for r in rows], separators=(",", ":")).encode()).hexdigest()


def main():
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    if dst.exists():
        sys.exit(f"{dst} exists; choose a new output folder")
    shutil.copytree(src, dst)
    hashes = json.loads((dst / "reports" / "label_hashes.json").read_text())
    done = bad = 0
    for rel, want in sorted(hashes.items()):
        p = dst / rel
        card = json.loads(p.read_text())
        rows = card["rows"]
        set_key = p.stem.split("__")[-1] if "__" in p.stem else p.stem
        built = BUILDERS[set_key](rows)
        merged = [dict(r, **b) for r, b in zip(rows, built)]
        got = label_sha(merged)
        if got != want:
            print(f"MISMATCH {rel}: rebuilt labels sha256 {got[:12]} != bundle {want[:12]}; not written")
            bad += 1
            continue
        card["rows"] = merged
        card.pop("labels_removed", None)
        card["labels_rebuilt"] = "attach_labels.py from the public dataset at the pinned revision; sha256 verified"
        p.write_text(json.dumps(card))
        done += 1
    print(f"labels rebuilt and verified for {done} files; {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
