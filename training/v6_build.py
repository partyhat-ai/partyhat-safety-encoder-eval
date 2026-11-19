"""v6_build.py: builds the training pools and the dev sets. Runs on Modal only.

The pools and dev sets are built from source, de-duplicated (within and across pools),
decontaminated against every reference set (exact normalised match; Jaccard >= 0.8 on 4-word
shingles; MiniLM-L6-v2 cosine >= 0.9), sampled, held-back rows set aside and soft targets
attached. Text is written only to the private Volume; the returned manifest holds counts, ids
and hashes.

Pool keys (kept as they are, since they name files and manifest entries):
  P1 ToxicChat (p1_copy.jsonl, written by v6_modal.build_p1), P2 Aegis 2.0, P3 WildGuardMix,
  P4 WildJailbreak, P5 FalseReject. p4_r0.jsonl is the WildJailbreak pool without its benign
  rows, used by recipe variant R0.
Dev sets: twins_prompt = paired prompts generated in-house; dev_pair2 = PHTest (harmless) with
HarmfulQA and CatQA (harmful); dev_holdout_p3/p4/p5 = held-back rows of the WildGuardMix,
WildJailbreak and FalseReject pools; aegis_val and tc_val = the Aegis 2.0 and ToxicChat
validation splits.
"""
import hashlib
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from decontam import normalize_text, shingles

JACCARD, SHINGLE, EMB_T = 0.8, 4, 0.9
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
P3_CAP, P4_PER_TYPE, HOLDOUT = 60_000, 15_000, 0.03   # WildGuardMix cap, WildJailbreak per type, held-back share
# soft targets (benign, unsafe) per pool
SOFT = {"P2": (0.02, 0.98), "P3": (0.05, 0.95), "P4": (0.05, 0.95), "P5": (0.05, 0.95)}
STOP_SHARE = 0.05
TEST_REFS = ("xstest", "jbb", "orbench", "openai_mod", "harmbench_std", "sst", "toxicchat_test",
             "aegis2_test_prompt", "wildguardtest_prompt")
DEV_REFS = ("twins_prompt", "dev_pair2", "aegis_val", "tc_val")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": path, "rows": len(rows),
            "unsafe": int(sum(int(r["toxicity"]) for r in rows if r.get("toxicity") is not None)),
            "sha256": sha_file(path)}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ------------------------------------------------------------------------------------------- #
# reference index: exact + shingle-Jaccard (lexical) and MiniLM cosine (embedding), per set     #
# ------------------------------------------------------------------------------------------- #
class RefIndex:
    def __init__(self, refs):
        """refs: {set_name: [text, ...]}"""
        self.names = list(refs)
        self.texts, self.sid = [], []
        for k, n in enumerate(self.names):
            self.texts += [str(t) for t in refs[n]]
            self.sid += [k] * len(refs[n])
        self.sid = np.asarray(self.sid)
        self.exact = defaultdict(set)
        self.inv = defaultdict(list)
        self.size = np.zeros(len(self.texts), dtype=np.int32)
        for i, t in enumerate(self.texts):
            n = normalize_text(t)
            if not n:
                continue
            self.exact[n].add(int(self.sid[i]))
            sh = shingles(n, SHINGLE)
            self.size[i] = len(sh)
            for g in sh:
                self.inv[g].append(i)
        self.emb = None

    def lexical_one(self, text):
        """{set_id: kind} for one candidate: 'exact' or 'nahash' (Jaccard >= 0.8)."""
        n = normalize_text(text)
        if not n:
            return {}
        hit = {s: "exact" for s in self.exact.get(n, ())}
        sh = shingles(n, SHINGLE)
        c = Counter()
        for g in sh:
            p = self.inv.get(g)
            if p:
                c.update(p)
        a = len(sh)
        for r, inter in c.items():
            s = int(self.sid[r])
            if s in hit:
                continue
            if inter / (a + self.size[r] - inter) >= JACCARD:
                hit[s] = "nahash"
        return hit

    def embed_refs(self, model):
        self.emb = _embed(model, self.texts)


def _embed(model, texts):
    import torch
    e = model.encode([str(t) for t in texts], batch_size=256, convert_to_tensor=True,
                     show_progress_bar=False, normalize_embeddings=True)
    return e.to(torch.float32)


_IDX = None


def _lex_chunk(texts):
    return [_IDX.lexical_one(t) for t in texts]


def flag(index, texts, model, procs=16):
    """Per candidate: {set_name: kind}. Lexical in a process pool, embedding on GPU."""
    global _IDX
    import multiprocessing as mp

    import torch
    _IDX = index
    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    ctx = mp.get_context("fork")
    with ctx.Pool(procs) as pool:
        lex = [h for part in pool.map(_lex_chunk, chunks) for h in part]
    hits = [{index.names[s]: k for s, k in h.items()} for h in lex]
    if index.emb is None:
        index.embed_refs(model)
    ce = _embed(model, texts)
    sid = torch.as_tensor(index.sid, device=index.emb.device)
    for st in range(0, ce.shape[0], 4096):
        sims = ce[st:st + 4096] @ index.emb.T          # [b, R]
        for k, name in enumerate(index.names):
            m = sid == k
            if not bool(m.any()):
                continue
            best = sims[:, m].max(dim=1).values.cpu().numpy()
            for j in np.flatnonzero(best >= EMB_T):
                hits[st + j].setdefault(name, "embedding")
    return hits


def summarise_hits(hits, n):
    per = defaultdict(Counter)
    for h in hits:
        for s, k in h.items():
            per[s][k] += 1
    flagged = sum(1 for h in hits if h)
    return {"n": n, "flagged_any": flagged,
            "per_ref": {s: {"flagged": sum(c.values()), "share": round(sum(c.values()) / max(n, 1), 5),
                            "by_kind": dict(c)} for s, c in sorted(per.items())}}


# ------------------------------------------------------------------------------------------- #
# de-duplication within and across pools                                                      #
# ------------------------------------------------------------------------------------------- #
def dedup_within(df):
    """Normalise; drop empty; a text with both labels is dropped entirely; keep first copy."""
    df = df.copy()
    df["_n"] = [normalize_text(t) for t in df["user_input"]]
    n0 = len(df)
    df = df[df["_n"] != ""]
    nl = df.groupby("_n")["toxicity"].nunique()
    conflict = set(nl[nl > 1].index)
    df_c = df[~df["_n"].isin(conflict)]
    out = df_c.drop_duplicates("_n", keep="first")
    return out.reset_index(drop=True), {"in": n0, "empty": int(n0 - len(df)),
                                        "conflict_texts": len(conflict),
                                        "conflict_rows": int(len(df) - len(df_c)),
                                        "dup_rows": int(len(df_c) - len(out)), "out": len(out)}


def dedup_across(p1_norm_label, pools):
    """pools: ordered {name: df with _n}. The ToxicChat pool (P1) wins any disputed label; among
    the other pools (P2-P5) a disputed label drops every copy, otherwise the first occurrence (pool
    order) is kept."""
    labels = defaultdict(set)
    first = {}
    for name, df in pools.items():
        for n, y in zip(df["_n"], df["toxicity"]):
            labels[n].add(int(y))
            first.setdefault(n, name)
    out, rep = {}, {}
    for name, df in pools.items():
        in_p1 = df["_n"].isin(p1_norm_label.keys())
        conflict = df["_n"].map(lambda n: len(labels[n]) > 1)
        later = df["_n"].map(lambda n: first[n] != name)
        keep = ~in_p1 & ~conflict & ~later
        rep[name] = {"in": len(df), "dup_or_conflict_with_P1": int(in_p1.sum()),
                     "conflict_P2_P5": int((~in_p1 & conflict).sum()),
                     "dup_of_earlier_pool": int((~in_p1 & ~conflict & later).sum()),
                     "out": int(keep.sum())}
        out[name] = df[keep].reset_index(drop=True)
    return out, rep


# ------------------------------------------------------------------------------------------- #
# sampling caps, held-back rows and soft targets                                              #
# ------------------------------------------------------------------------------------------- #
def stratified_cap(df, strata_cols, cap, rng):
    """Proportional stratified sample of `cap` rows (largest remainder); all rows if fewer."""
    if len(df) <= cap:
        return df.reset_index(drop=True), {"cap_applied": False}
    groups = sorted(df.groupby(strata_cols).groups.items(), key=lambda kv: str(kv[0]))
    sizes = np.array([len(ix) for _, ix in groups], float)
    raw = cap * sizes / sizes.sum()
    take = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - take), kind="stable")[: cap - take.sum()]:
        take[i] += 1
    parts = []
    for (key, ix), k in zip(groups, take):
        ix = np.asarray(sorted(ix))
        parts.append(df.loc[np.sort(rng.choice(ix, size=k, replace=False))])
    return pd.concat(parts).reset_index(drop=True), {"cap_applied": True, "strata": {
        str(key): int(k) for (key, _), k in zip(groups, take)}}


def holdout(df, strata_cols, rng, share=HOLDOUT):
    is_h = np.zeros(len(df), bool)
    for key, ix in sorted(df.groupby(strata_cols).groups.items(), key=lambda kv: str(kv[0])):
        ix = np.asarray(sorted(ix))
        k = int(round(share * len(ix)))
        is_h[rng.permutation(ix)[:k]] = True
    return df[~is_h].reset_index(drop=True), df[is_h].reset_index(drop=True)


def to_rows(df, pool, soft=None, hard_soft=False):
    rows = []
    for r in df.itertuples(index=False):
        y = int(r.toxicity)
        d = {"row_id": r.row_id, "user_input": r.user_input, "toxicity": y, "pool": pool,
             "type": getattr(r, "type", pool)}
        if soft is not None:
            d["toxicity_soft"] = soft[1] if y else soft[0]
        elif hard_soft:
            d["toxicity_soft"] = float(y)
        rows.append(d)
    return rows


# ------------------------------------------------------------------------------------------- #
# orchestration                                                                                #
# ------------------------------------------------------------------------------------------- #
def load_references_test(vol):
    import v6_data as vd
    from tc_common import INPUT_FIELD, load_test
    j = lambda n: json.load(open(f"{vol}/sets/{n}.json"))  # noqa: E731
    refs, meta = {}, {}
    refs["xstest"] = j("xstest_safe") + j("xstest_unsafe")
    refs["orbench"] = j("orbench_hard") + j("orbench_toxic")
    jb, rev = vd.load_hf_any(vd.JBB, "behaviors")
    refs["jbb"] = list(jb["harmful"]["Goal"]) + list(jb["benign"]["Goal"])
    meta["jbb_revision"] = rev
    om, sha = vd.load_openai_mod()
    refs["openai_mod"] = [r["prompt"] for r in om]
    meta["openai_mod_commit"] = sha
    refs["harmbench_std"] = vd.load_harmbench_std()
    refs["sst"] = vd.load_sst()
    refs["toxicchat_test"] = [str(t) for t in load_test()[INPUT_FIELD]]
    at = vd.load_aegis("test.json")
    refs["aegis2_test_prompt"] = [p for p in at["prompt"] if p != "REDACTED"]
    wt, rev, _ = vd.load_wgm("wildguardtest")
    refs["wildguardtest_prompt"] = list(wt.loc[wt["prompt_harm_label"].notna(), "prompt"])
    meta["wildguardmix_revision"] = rev
    meta["n"] = {k: len(v) for k, v in refs.items()}
    return refs, meta


def aegis_prompts(df):
    df = df[(df["prompt"] != "REDACTED") & df["prompt_label"].isin(["safe", "unsafe"])]
    df = df[df["prompt"].astype(str).str.strip().str.len() >= 3]
    return pd.DataFrame({"row_id": "aegis:" + df["id"].astype(str), "user_input": df["prompt"].astype(str),
                         "toxicity": (df["prompt_label"] == "unsafe").astype(int), "type": "aegis"})


def build_all(vol, out_root, procs=16):
    import v6_data as vd
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cuda")
    man = {"params": {"jaccard": JACCARD, "shingle": SHINGLE, "emb_threshold": EMB_T,
                      "emb_model": EMB_MODEL, "p3_cap": P3_CAP, "p4_per_type": P4_PER_TYPE,
                      "holdout": HOLDOUT, "rng": "numpy.random.default_rng(13), one stream: "
                      "P3 cap, P4 cap (types in sorted order), holdouts P3, P4, P5"}}

    # ---- test references ---------------------------------------------------------------- #
    test_refs, man["references"] = load_references_test(vol)
    idx_test = RefIndex(test_refs)

    # ---- dev sets ------------------------------------------------------------------------- #
    calm = read_jsonl(f"{vol}/data/calm_train.jsonl") + read_jsonl(f"{vol}/data/calm_dev.jsonl")
    tw = [r for r in calm if r["axis"] == "prompt"]
    h = flag(idx_test, [r["user_input"] for r in tw], model, procs)
    bad_pairs = {r["pair_id"] for r, x in zip(tw, h) if x}
    man["twins_decontam"] = summarise_hits(h, len(tw)) | {"pairs_dropped": len(bad_pairs)}
    twins = [{"row_id": f"calm:{r['pair_id']}:{r['toxicity']}", "user_input": r["user_input"],
              "toxicity": int(r["toxicity"]), "pair_id": r["pair_id"], "category": r["category"]}
             for r in tw if r["pair_id"] not in bad_pairs]

    ph, ph_rev = vd.load_hf_any(vd.PHTEST)
    ph = ph["train"]
    man["phtest_harmfulness_counts"] = {str(k): int(v) for k, v in ph["Harmfulness"].value_counts().items()}
    ph = ph[ph["Harmfulness"].astype(str).str.lower() == "harmless"]
    assert len(ph) > 0, "PHTest: no rows labelled harmless"
    hq, hq_rev = vd.load_hf_any(vd.HQA)
    cq, cq_rev = vd.load_hf_any(vd.CATQA)
    pair2 = pd.concat([
        pd.DataFrame({"row_id": "phtest:" + ph["ID"].astype(str), "user_input": ph["Request"].astype(str),
                      "toxicity": 0, "type": "phtest_harmless"}),
        pd.DataFrame({"row_id": "harmfulqa:" + hq["train"]["id"].astype(str),
                      "user_input": hq["train"]["question"].astype(str), "toxicity": 1, "type": "harmfulqa"}),
        pd.DataFrame({"row_id": [f"catqa_en:{i}" for i in range(len(cq["en"]))],
                      "user_input": cq["en"]["Question"].astype(str), "toxicity": 1, "type": "catqa_en"}),
    ], ignore_index=True)
    pair2, man["dev_pair2_dedup"] = dedup_within(pair2)
    h = flag(idx_test, list(pair2["user_input"]), model, procs)
    man["dev_pair2_decontam"] = summarise_hits(h, len(pair2))
    pair2 = pair2[[not x for x in h]].reset_index(drop=True)
    man["dev_pair2_sources"] = {"phtest": ph_rev, "harmfulqa": hq_rev, "catqa": cq_rev,
                                "columns": {"phtest": "Request (Harmfulness == harmless)",
                                            "harmfulqa": "question", "catqa": "Question (split en)"},
                                "counts": {k: int(v) for k, v in pair2["type"].value_counts().items()}}

    av = aegis_prompts(vd.load_aegis("validation.json"))
    h = flag(idx_test, list(av["user_input"]), model, procs)
    man["aegis_val_vs_test_refs_report_only"] = summarise_hits(h, len(av))

    tc_val = read_jsonl(f"{out_root}/dev/tc_val.jsonl")
    p1 = read_jsonl(f"{out_root}/pools/p1_copy.jsonl")

    dev_refs = {"twins_prompt": [r["user_input"] for r in twins], "dev_pair2": list(pair2["user_input"]),
                "aegis_val": list(av["user_input"]), "tc_val": [r["user_input"] for r in tc_val]}
    all_refs = dict(test_refs) | dev_refs
    idx_all = RefIndex(all_refs)
    os.makedirs(f"{out_root}/refs", exist_ok=True)
    with open(f"{out_root}/refs/refs.json", "w", encoding="utf-8") as f:
        json.dump(all_refs, f, ensure_ascii=False)

    # ---- pools P2-P5 (Aegis 2.0, WildGuardMix, WildJailbreak, FalseReject): load, dedup ------ #
    p2 = aegis_prompts(vd.load_aegis("train.json"))
    g, wgm_rev, _ = vd.load_wgm("wildguardtrain")
    g = g[g["prompt_harm_label"].isin(["harmful", "unharmful"])]
    p3 = pd.DataFrame({"row_id": [f"wgm:{i}" for i in g.index], "user_input": g["prompt"].astype(str),
                       "toxicity": (g["prompt_harm_label"] == "harmful").astype(int).values,
                       "type": np.where(g["adversarial"].astype(bool), "adversarial", "vanilla")})
    w, wjb_rev = vd.load_wjb()
    txt = np.where(w["data_type"].str.startswith("adversarial"), w["adversarial"], w["vanilla"])
    p4 = pd.DataFrame({"row_id": [f"wjb:{i}" for i in w.index], "user_input": txt,
                       "toxicity": w["data_type"].str.endswith("_harmful").astype(int).values,
                       "type": w["data_type"].values})
    p4 = p4[p4["user_input"].str.len() > 0]
    fr = read_jsonl(f"{vol}/v6/src/falsereject_pool.clean.jsonl")
    p5 = pd.DataFrame({"row_id": [f"falsereject:{i}" for i in range(len(fr))],
                       "user_input": [r["user_input"] for r in fr],
                       "toxicity": [int(r["toxicity"]) for r in fr], "type": "falsereject"})
    man["sources"] = {"P2": {"hub": vd.AEGIS[0], "revision": vd.AEGIS[1], "file": "train.json",
                             "rows_after_rules": int(len(p2))},
                      "P3": {"hub": vd.WGM, "revision": wgm_rev, "config": "wildguardtrain",
                             "rows_labelled": int(len(p3))},
                      "P4": {"hub": vd.WJB, "revision": wjb_rev, "file": "train/train.tsv",
                             "rows_raw": int(len(w)), "rows_nonempty_text": int(len(p4))},
                      "P5": {"file": "gen_data/falsereject_pool.clean.jsonl", "rows": int(len(p5)),
                             "sha256": sha_file(f"{vol}/v6/src/falsereject_pool.clean.jsonl")}}
    pools, man["dedup_within"] = {}, {}
    for name, df in (("P2", p2), ("P3", p3), ("P4", p4), ("P5", p5)):
        pools[name], man["dedup_within"][name] = dedup_within(df)
    p1_norm = {normalize_text(r["user_input"]): r["toxicity"] for r in p1}
    pools, man["dedup_across"] = dedup_across(p1_norm, pools)

    # ---- decontaminate -------------------------------------------------------------------- #
    man["decontam"] = {}
    h = flag(idx_all, [r["user_input"] for r in p1], model, procs)
    h = [{k: v for k, v in x.items() if k not in ("toxicchat_test", "tc_val")} for x in h]
    man["decontam"]["P1_report_only"] = summarise_hits(h, len(p1))
    stops = []
    for name in ("P2", "P3", "P4", "P5"):
        df = pools[name]
        h = flag(idx_all, list(df["user_input"]), model, procs)
        s = summarise_hits(h, len(df))
        man["decontam"][name] = s
        stops += [(name, r, v["share"]) for r, v in s["per_ref"].items() if v["share"] > STOP_SHARE]
        pools[name] = df[[not x for x in h]].reset_index(drop=True)
    if stops:
        # Stop when more than 5% of a pool matches one reference set.
        man["STOP"] = {"rule": "more than 5% of a pool flagged against one reference set",
                       "hits": [{"pool": p, "ref": r, "share": s} for p, r, s in stops]}
        return man

    # ---- sample, hold out, soft targets ----------------------------------------------------- #
    rng = np.random.default_rng(13)
    pools["P3"], man["p3_cap"] = stratified_cap(pools["P3"], ["type", "toxicity"], P3_CAP, rng)
    parts, unused = [], []
    for t in sorted(pools["P4"]["type"].unique()):
        d = pools["P4"][pools["P4"]["type"] == t].reset_index(drop=True)
        pick = np.sort(rng.choice(len(d), size=min(P4_PER_TYPE, len(d)), replace=False))
        mask = np.zeros(len(d), bool)
        mask[pick] = True
        parts.append(d[mask])
        unused.append(d[~mask])
    pools["P4"] = pd.concat(parts).reset_index(drop=True)
    p4_unused = pd.concat(unused).reset_index(drop=True)
    hold = {}
    for name in ("P3", "P4", "P5"):
        pools[name], hold[name] = holdout(pools[name], ["type", "toxicity"], rng)

    files = {}
    pdir, ddir = f"{out_root}/pools", f"{out_root}/dev"
    for name in ("P2", "P3", "P4", "P5"):
        files[name] = write_jsonl(f"{pdir}/{name.lower()}.jsonl", to_rows(pools[name], name, SOFT[name]))
    r0p4 = pools["P4"][pools["P4"]["type"].isin(["vanilla_harmful", "adversarial_harmful"])]
    files["P4_R0"] = write_jsonl(f"{pdir}/p4_r0.jsonl", to_rows(r0p4, "P4", SOFT["P4"]))
    for name in ("P3", "P4", "P5"):
        files[f"dev_holdout_{name.lower()}"] = write_jsonl(f"{ddir}/dev_holdout_{name.lower()}.jsonl",
                                                           to_rows(hold[name], name))
    files["twins_prompt"] = write_jsonl(f"{ddir}/twins_prompt.jsonl", twins)
    files["dev_pair2"] = write_jsonl(f"{ddir}/dev_pair2.jsonl", to_rows(pair2, "dev_pair2"))
    files["aegis_val"] = write_jsonl(f"{ddir}/aegis_val.jsonl", to_rows(av, "aegis_val"))
    files["p4_vanilla_unused_for_r3"] = write_jsonl(
        f"{out_root}/r3/p4_vanilla_unused.jsonl",
        to_rows(p4_unused[p4_unused["type"].str.startswith("vanilla")], "WJB_EXTRA", SOFT["P4"]))
    man["files"] = files
    man["type_counts"] = {n: {f"{t}|{y}": int(c) for (t, y), c in
                              pools[n].groupby(["type", "toxicity"]).size().items()} for n in pools}
    n_p1 = len(p1)
    man["N_R1"] = 3 * n_p1 + sum(files[k]["rows"] for k in ("P2", "P3", "P4", "P5"))
    man["N_R0"] = 3 * n_p1 + files["P2"]["rows"] + files["P3"]["rows"] + files["P4_R0"]["rows"]
    man["unsafe_R1"] = 3 * sum(r["toxicity"] for r in p1) + sum(
        files[k]["unsafe"] for k in ("P2", "P3", "P4", "P5"))
    return man


def reverse_audit(out_root, procs=16):
    """Reverse audit: every reference row against the base recipe's (R1) final training files (the
    four pool files plus the ToxicChat copy). The ToxicChat copy is not checked against the
    ToxicChat references, which come from the same dataset by design."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cuda")
    refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
    pools = {n: [r["user_input"] for r in read_jsonl(f"{out_root}/pools/{n}.jsonl")]
             for n in ("p2", "p3", "p4", "p5")}
    p1 = [r["user_input"] for r in read_jsonl(f"{out_root}/pools/p1_copy.jsonl")]
    idx = RefIndex(pools | {"p1_copy": p1})
    out = {}
    for name, texts in refs.items():
        h = flag(idx, texts, model, procs)
        if name in ("toxicchat_test", "tc_val"):
            h = [{k: v for k, v in x.items() if k != "p1_copy"} for x in h]
        out[name] = summarise_hits(h, len(texts))
    out["total_flagged"] = int(sum(v["flagged_any"] for v in out.values() if isinstance(v, dict)))
    return out


# ------------------------------------------------------------------------------------------- #
# extra rows for R3 (base recipe + teacher labels), collected best effort                     #
# ------------------------------------------------------------------------------------------- #
R3_WJB_PER_TYPE, R3_HARMAUG, R3_WILDCHAT = 15_000, 30_000, 40_000


def _wildchat_first_turns(rng, want, max_shards=None):
    """English first user turns, 20-2000 chars, at most 3 per hashed_ip. Only the first turn's
    content, `language` and `hashed_ip` are accessed; no assistant text or moderation field."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    import v6_data as vd
    rev = vd.revision(vd.WILDCHAT)
    files = sorted(f for f in HfApi().list_repo_files(vd.WILDCHAT, repo_type="dataset", revision=rev)
                   if f.endswith(".parquet"))
    rows, per_ip, used = [], Counter(), []
    for f in files[:max_shards]:
        t = pq.read_table(hf_hub_download(vd.WILDCHAT, f, repo_type="dataset", revision=rev),
                          columns=["conversation", "language", "hashed_ip", "conversation_hash"])
        used.append(f)
        lang, ip, conv, ch = (t.column(c).to_pylist() for c in
                              ("language", "hashed_ip", "conversation", "conversation_hash"))
        order = rng.permutation(len(lang))
        for i in order:
            if lang[i] != "English" or not conv[i]:
                continue
            first = conv[i][0]
            if first.get("role") != "user":
                continue
            txt = str(first.get("content") or "")
            if not (20 <= len(txt) <= 2000) or per_ip[ip[i]] >= 3:
                continue
            per_ip[ip[i]] += 1
            rows.append({"row_id": f"wildchat:{ch[i]}", "user_input": txt})
        del t, conv
        if len(rows) >= want:
            break
    return rows, {"revision": rev, "shards_read": used, "shards_total": len(files),
                  "eligible_rows_collected": len(rows)}


def p1_matched_heldout(out_root, procs=8):
    """Sensitivity check: the held-out reference items (orbench, jbb, sst) that match the ToxicChat
    pool (p1_copy), as (card set, row index, sha256 of the item text). No text is returned."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cuda")
    refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
    p1 = [r["user_input"] for r in read_jsonl(f"{out_root}/pools/p1_copy.jsonl")]
    idx = RefIndex({"p1_copy": p1})
    # refs.json order (load_references_test): orbench = hard (1,319) + toxic (655);
    # jbb = harmful (100) + benign (100); sst = file order
    split = {"orbench": [("orbench_hard", 1319), ("orbench_toxic", 655)],
             "jbb": [("jbb_harmful", 100), ("jbb_benign", 100)], "sst": [("sst", len(refs["sst"]))]}
    out = []
    for name, parts in split.items():
        texts = refs[name]
        assert len(texts) == sum(n for _, n in parts), name
        h = flag(idx, texts, model, procs)
        for k in [i for i, x in enumerate(h) if x]:
            off = 0
            for card_set, n in parts:
                if k < off + n:
                    out.append({"set": card_set, "i": k - off, "kind": h[k]["p1_copy"],
                                "text_sha256": hashlib.sha256(texts[k].encode("utf-8")).hexdigest()})
                    break
                off += n
    return {"items": out, "n": len(out),
            "note": "row index i is the item's position in the pinned set file order used by both "
                    "harnesses; verify by text_sha256 against the card's source before excluding"}


def filter_aegis_val(out_root, procs=16):
    """Drop aegis_val rows that match any test set, because aegis_val is used to set the frozen
    thresholds. The pre-filter file is kept as aegis_val_prefilter.jsonl."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cuda")
    refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
    idx = RefIndex({k: refs[k] for k in TEST_REFS})
    src = f"{out_root}/dev/aegis_val.jsonl"
    pre = f"{out_root}/dev/aegis_val_prefilter.jsonl"
    if not os.path.exists(pre):
        os.rename(src, pre)
    rows = read_jsonl(pre)
    h = flag(idx, [r["user_input"] for r in rows], model, procs)
    kept = [r for r, x in zip(rows, h) if not x]
    out = {"prefilter": {"rows": len(rows), "sha256": sha_file(pre)},
           "decontam": summarise_hits(h, len(rows)),
           "aegis_val": write_jsonl(src, kept)}
    out["aegis_val"]["safe"] = out["aegis_val"]["rows"] - out["aegis_val"]["unsafe"]
    return out


def build_r3_training(out_root, targets_path):
    """R3 training files from the committee's targets (v6_committee_targets.py --out): the
    ToxicChat and Aegis 2.0 files are unchanged; WildGuardMix, WildJailbreak and FalseReject rows
    take the committee target; the extra rows go to r3_extra.jsonl."""
    tg = {r["row_id"]: r for r in read_jsonl(targets_path)}
    out, rep = {}, {"targets_rows": len(tg), "targets_sha256": sha_file(targets_path)}
    d = f"{out_root}/r3/pools"
    for n in ("p1_copy", "p2"):
        rows = read_jsonl(f"{out_root}/pools/{n}.jsonl")
        out[n] = write_jsonl(f"{d}/{n}.jsonl", rows)
    for n in ("p3", "p4", "p5"):
        rows, changed, missing = [], 0, 0
        for r in read_jsonl(f"{out_root}/pools/{n}.jsonl"):
            t = tg.get(r["row_id"])
            if t is None:
                missing += 1
            else:
                changed += int(float(t["toxicity_soft"]) != float(r["toxicity_soft"]))
                r = dict(r, toxicity=int(t["toxicity"]), toxicity_soft=float(t["toxicity_soft"]))
            rows.append(r)
        out[n] = write_jsonl(f"{d}/{n}.jsonl", rows) | {"target_changed": changed, "not_in_targets": missing}
    extra = [dict(row_id=t["row_id"], user_input=t["user_input"], toxicity=int(t["toxicity"]),
                  toxicity_soft=float(t["toxicity_soft"]), pool=t["pool"], type=t.get("type", t["pool"]))
             for t in tg.values() if t["pool"] in ("WJB_EXTRA", "HARMAUG", "WILDCHAT")]
    out["r3_extra"] = write_jsonl(f"{d}/r3_extra.jsonl", extra)
    out["r3_extra"]["per_pool"] = dict(Counter(r["pool"] for r in extra))
    assert all(r["toxicity_soft"] is not None for r in extra)
    rep["files"] = out
    rep["N_R3"] = 2 * out["p1_copy"]["rows"] + sum(out[k]["rows"] for k in out)
    return rep


def build_r3(vol, out_root, procs=16):
    import v6_data as vd
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cuda")
    rng = np.random.default_rng(13)
    man = {"rng": "default_rng(13): WJB_EXTRA per vanilla type (sorted), HarmAug, WildChat"}
    all_refs = json.load(open(f"{out_root}/refs/refs.json", encoding="utf-8"))
    idx = RefIndex(all_refs)
    base = {n: read_jsonl(f"{out_root}/pools/{n}.jsonl") for n in ("p1_copy", "p2", "p3", "p4", "p5")}
    held = [r for n in ("p3", "p4", "p5") for r in read_jsonl(f"{out_root}/dev/dev_holdout_{n}.jsonl")]
    seen = {normalize_text(r["user_input"]) for rs in base.values() for r in rs}
    seen |= {normalize_text(r["user_input"]) for r in held}

    # extra vanilla WildJailbreak rows: already de-duplicated and decontaminated with the WildJailbreak pool
    un = pd.DataFrame(read_jsonl(f"{out_root}/r3/p4_vanilla_unused.jsonl"))
    parts = []
    for t in sorted(un["type"].unique()):
        d = un[un["type"] == t].reset_index(drop=True)
        parts.append(d.iloc[np.sort(rng.choice(len(d), size=min(R3_WJB_PER_TYPE, len(d)), replace=False))])
    wjb = pd.concat(parts).reset_index(drop=True)
    man["WJB_EXTRA"] = {"available": int(len(un)), "taken": {k: int(v) for k, v in wjb["type"].value_counts().items()}}

    def clean_unlabelled(rows, name, cap):
        df = pd.DataFrame(rows)
        df["toxicity"] = 0   # placeholder for dedup_within only; never written
        df, dd = dedup_within(df)
        n0 = len(df)
        df = df[~df["_n"].isin(seen)].reset_index(drop=True)
        cross = n0 - len(df)
        h = flag(idx, list(df["user_input"]), model, procs)
        s = summarise_hits(h, len(df))
        df = df[[not x for x in h]].reset_index(drop=True)
        if len(df) > cap:
            df = df.iloc[np.sort(rng.choice(len(df), size=cap, replace=False))].reset_index(drop=True)
        seen.update(df["_n"])
        man[name] = {"dedup_within": dd, "dup_of_r1_or_holdout": int(cross), "decontam": s,
                     "kept": int(len(df))}
        return [{"row_id": r.row_id, "user_input": r.user_input, "toxicity": None,
                 "toxicity_soft": None, "pool": name, "type": name} for r in df.itertuples(index=False)]

    from huggingface_hub import HfApi, hf_hub_download
    ha_rev = vd.revision(vd.HARMAUG)
    ha_files = [f for f in HfApi().list_repo_files(vd.HARMAUG, repo_type="dataset", revision=ha_rev)
                if f.endswith(".parquet")]
    ha = pd.concat([pd.read_parquet(hf_hub_download(vd.HARMAUG, f, repo_type="dataset", revision=ha_rev))
                    for f in ha_files], ignore_index=True)
    man["HARMAUG_source"] = {"revision": ha_rev, "columns": list(map(str, ha.columns)), "rows": int(len(ha))}
    col = next((c for c in ("instruction", "prompt", "question", "query") if c in ha.columns), None)
    if col is None:
        man["STOP_HARMAUG"] = "no instruction-like column; HarmAug skipped"
        harmaug = []
    else:
        man["HARMAUG_source"]["text_column"] = col
        txt = ha[col].astype(str)
        ha_rows = [{"row_id": f"harmaug:{i}", "user_input": t} for i, t in zip(ha.index, txt)
                   if 3 <= len(t.strip())]
        sub = rng.permutation(len(ha_rows))[: 2 * R3_HARMAUG]
        harmaug = clean_unlabelled([ha_rows[i] for i in sorted(sub)], "HARMAUG", R3_HARMAUG)

    wc, man["WILDCHAT_source"] = _wildchat_first_turns(rng, int(1.5 * R3_WILDCHAT))
    wildchat = clean_unlabelled(wc, "WILDCHAT", R3_WILDCHAT)

    rows = []
    for n, pool in (("p1_copy", "P1"), ("p2", "P2"), ("p3", "P3"), ("p4", "P4"), ("p5", "P5")):
        for r in base[n]:
            rows.append({k: r.get(k) for k in ("row_id", "user_input", "toxicity", "toxicity_soft", "type")}
                        | {"pool": pool})
    for r in wjb.to_dict("records"):
        rows.append({"row_id": r["row_id"], "user_input": r["user_input"], "toxicity": int(r["toxicity"]),
                     "toxicity_soft": float(r["toxicity_soft"]), "type": r["type"], "pool": "WJB_EXTRA"})
    rows += harmaug + wildchat
    ids = [r["row_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate row_id in R3 rows"
    man["r3_rows"] = write_jsonl(f"{out_root}/r3/r3_rows.jsonl", rows)
    man["r3_rows"]["per_pool"] = dict(Counter(r["pool"] for r in rows))
    return man
