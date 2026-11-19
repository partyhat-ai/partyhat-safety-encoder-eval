#!/usr/bin/env python3
"""score_test_sets.py: score the published weights (or an open comparison guard) on the card's twelve test sets
and write per-item scorecards in the evidence bundle's schema, then optionally compare them with the bundle.

This is the link from the published weights to the scorecards: `reproduce_card.py` recomputes the card from the
scorecards; this script recomputes the scorecards from the model. It is a standalone copy of the scoring paths of
the harness that produced the bundle (same set loaders and row order, same encoder settings, same generative-guard
score), with the project's private infrastructure removed. Text is read from the public datasets at pinned
revisions and is never written out: a scorecard row holds the row index, label, score, margin and decision.

    pip install -r requirements-scoring.txt
    # release 2.0, the published weights (gated: accept the model's access terms first)
    python3 score_test_sets.py --out scorecards_mine --compare evidence_with_labels
    # a quick check on 32 prompts per set (CPU is fine)
    python3 score_test_sets.py --cap 32 --device cpu --out scorecards_quick --compare evidence_with_labels
    # an open comparison guard (needs a GPU; Llama Guard 3 is gated by Meta)
    python3 score_test_sets.py --model llamaguard3 --sets xstest_safe,xstest_unsafe --out sc --compare evidence

--compare checks every written scorecard against the bundle's: the same rows in the same order, the same labels
where the bundle has them, the largest margin difference, and the decisions. Floating-point results differ across
GPUs, batch compositions and library versions, so margins are compared with a tolerance: 0.02 for release 2.0 (FP32;
the parity tolerance the project used between its two scorers) and 0.5 for the generative guards, which run in
bfloat16, where a logit moves in steps of 0.0625-0.25 at the magnitudes involved. For release 2.0 a decision may
differ only where the score sits within 1e-4 of the threshold; a generative guard's decision, verdict and parse status
must match exactly. It exits non-zero if any scorecard fails. On a CPU check of 32 prompts per set, release 2.0's
margins matched the bundle's (scored on an A100) within 1.4e-5 on all twelve sets.

Data: OR-Bench and XSTest are rebuilt as JSON prompt lists from the pinned Hub revisions and checked against the
sha256 the harness locked them with; openai_mod, JailbreakBench and WildGuardTest files are sha256-checked too.
WildGuardTest is gated (accept allenai/wildguardmix's terms first).
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import pathlib
import re
import sys
import time
import urllib.request

ENCODER_MAX_LEN = 512
MAX_NEW_TOKENS = 100            # generative guards: decode budget
GEN_CONTENT_MAX_TOKENS = 8000   # generative guards: per-field truncation

# ------------------------------------------------------------------------------------------------ sets
ORBENCH = ("bench-llms/or-bench", "fd6ee135ee63ff6c4f3ff72c0e39627bf0a7f314")
XSTEST = ("natolambert/xstest-v2-copy", "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5")
LOCKED_PROMPTS_SHA256 = {   # sha256 of json.dumps(list of prompts), as locked by the harness
    "orbench_hard": "e590a3740cdb67b0c97015ccae4dda46fd401e5d9ac5334e5ff99ef2f87ccf3d",
    "orbench_toxic": "2441e94e845226e617360663259e8d56d10a345e132b00546192dc3551e8c066",
    "xstest_safe": "f089f2dc728d8d6eba9853bb3469d29df88af812ece6ed3b802914d1d6b6c568",
    "xstest_unsafe": "f440a773d9201132a9ec5d82275679977f173f23276369d38240cc2311684855",
}
TOXICCHAT = ("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_test.csv", "29df8e4dba60e1f4af4b4075c0705c5b313548a8")
AEGIS = ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test.json", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c")
SST = ("Bertievidgen/SimpleSafetyTests", "sst_test_cases.csv", "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5")
HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
                 "8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/harmbench_behaviors_text_all.csv")
OPENAI_MOD_URL = ("https://raw.githubusercontent.com/openai/moderation-api-release/"
                  "f4ab51b5edd3bfbcb349a56324274235b674e0e4/data/samples-1680.jsonl.gz")
OPENAI_MOD_SHA256 = "ef3f7de86ee394fe337b0a566ba4b9c7db058f3caa8d6a4a0006138b663ab9a8"
OPENAI_MOD_CATS = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")
JBB_REPO, JBB_REV = "JailbreakBench/JBB-Behaviors", "886acc352a31533ffbcf4ef22c744658688086fc"
JBB_FILES = {"jbb_benign": ("data/benign-behaviors.csv", "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468"),
             "jbb_harmful": ("data/harmful-behaviors.csv", "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1")}
WGT = ("allenai/wildguardmix", "test/wildguard_test.parquet", "d29c47f41c8b51348b5c8e8c81c039b3132b66d1")
WGT_SHA256 = "6ccc2909c1ae6d41424fac69f1fc32535b1de39cb8f80407d81e8bc64a0bebca"

SETS = {   # set key -> role (benign / unsafe / mixed: per-row labels); every set is prompt-only
    "orbench_hard": "benign", "orbench_toxic": "unsafe", "xstest_safe": "benign", "xstest_unsafe": "unsafe",
    "jbb_benign": "benign", "jbb_harmful": "unsafe", "harmbench_std": "unsafe", "sst": "unsafe",
    "openai_mod": "mixed", "toxicchat_test": "mixed", "aegis2_test_prompt": "mixed", "wildguardtest_prompt": "mixed",
}


def _sha_check(raw, want, key):
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SystemExit(f"[data] {key}: sha256 {got[:12]} != pinned {want[:12]}; refusing to score")
    return raw


def _hub_file(repo, fname, rev):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset", revision=rev)


def _http_bytes(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _locked_prompts(key):
    """OR-Bench and XSTest prompt lists, rebuilt exactly as the harness cached them, then sha256-checked."""
    from datasets import load_dataset
    if key.startswith("orbench"):
        cfg = "or-bench-hard-1k" if key == "orbench_hard" else "or-bench-toxic"
        ds = load_dataset(ORBENCH[0], cfg, split="train", revision=ORBENCH[1])
        prompts = [str(x) for x in ds["prompt"]]
    else:
        xs = load_dataset(XSTEST[0], split="prompts", revision=XSTEST[1]).to_pandas()
        lab = "type" if "type" in xs.columns else "label"
        is_contrast = xs[lab].astype(str).str.startswith("contrast")
        keep = is_contrast if key == "xstest_unsafe" else ~is_contrast
        prompts = xs[keep]["prompt"].astype(str).tolist()
    _sha_check(json.dumps(prompts).encode(), LOCKED_PROMPTS_SHA256[key], key)
    return prompts


def load_set(key):
    """[{src, text, label, aux}], in the bundle's row order. Text stays in memory only."""
    role = SETS[key]
    if key in LOCKED_PROMPTS_SHA256:
        lab = int(role == "unsafe")
        return [{"src": i, "text": t, "label": lab, "aux": {}} for i, t in enumerate(_locked_prompts(key))]
    if key == "toxicchat_test":
        with open(_hub_file(*TOXICCHAT), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [{"src": i, "text": r["user_input"], "label": int(r["toxicity"]),
                 "aux": {"jailbreaking": int(r["jailbreaking"])}} for i, r in enumerate(rows)]
    if key == "aegis2_test_prompt":
        items = []
        for i, r in enumerate(json.load(open(_hub_file(*AEGIS), encoding="utf-8"))):
            if r.get("prompt") == "REDACTED":
                continue
            cats = [c.strip() for c in str(r.get("violated_categories") or "").split(",") if c.strip()]
            nc_only = cats == ["Needs Caution"]
            shipped = int(r["prompt_label"] == "unsafe")
            items.append({"src": i, "text": r["prompt"], "label": 1 if nc_only else shipped,
                          "aux": {"label_shipped": shipped, "gt_nc_only": nc_only}})
        return items
    if key == "sst":
        with open(_hub_file(*SST), newline="", encoding="utf-8") as f:
            return [{"src": i, "text": r["prompt"], "label": 1, "aux": {}} for i, r in enumerate(csv.DictReader(f))]
    if key == "harmbench_std":
        rows = [r for r in csv.DictReader(io.StringIO(_http_bytes(HARMBENCH_URL).decode("utf-8")))
                if r["FunctionalCategory"] == "standard"]
        return [{"src": i, "text": r["Behavior"], "label": 1, "aux": {}} for i, r in enumerate(rows)]
    if key == "openai_mod":
        raw = _sha_check(_http_bytes(OPENAI_MOD_URL), OPENAI_MOD_SHA256, key)
        rows = [json.loads(l) for l in gzip.decompress(raw).decode("utf-8").splitlines() if l.strip()]
        return [{"src": i, "text": r["prompt"], "label": int(any(r.get(c) == 1 for c in OPENAI_MOD_CATS)),
                 "aux": {}} for i, r in enumerate(rows)]
    if key in JBB_FILES:
        fname, want = JBB_FILES[key]
        p = _hub_file(JBB_REPO, fname, JBB_REV)
        _sha_check(open(p, "rb").read(), want, key)
        with open(p, newline="", encoding="utf-8") as f:
            return [{"src": i, "text": r["Goal"], "label": int(role == "unsafe"), "aux": {"jbb_source": r.get("Source")}}
                    for i, r in enumerate(csv.DictReader(f))]
    if key == "wildguardtest_prompt":
        import pandas as pd
        p = _hub_file(*WGT)
        _sha_check(open(p, "rb").read(), WGT_SHA256, key)
        items = []
        for i, r in enumerate(pd.read_parquet(p).itertuples(index=False)):
            if r.prompt_harm_label not in ("harmful", "unharmful"):
                continue                        # 26 rows carry no prompt label
            items.append({"src": i, "text": r.prompt, "label": int(r.prompt_harm_label == "harmful"),
                          "aux": {"adversarial": bool(r.adversarial)}})
        return items
    raise KeyError(key)


# ------------------------------------------------------------------------------------------------ models
RELEASE2 = dict(kind="release2", mode="classifier", hub="partyhat-hf/partyhat-safety-encoder",
                card_model="v6_R3_s13")
GUARDS = {   # the open comparison guards on the card, as the harness scored them
    "llamaguard3": dict(kind="generative", mode="guard", hub="meta-llama/Llama-Guard-3-8B",
                        rev="7327bd9f6efbbe6101dc6cc4736302b3cbb6e425", verdict="binary",
                        field_prompt=r"^\s*", cats=r"\n\s*(S\d+(?:,\s*S\d+)*)"),
    "qwen3guard06_strict": dict(kind="generative", mode="guard", hub="Qwen/Qwen3Guard-Gen-0.6B",
                                rev="fada3b2f655b89601929198343c94cd2f64d93cc", verdict="3way_strict",
                                max_new_tokens=128, card_suffix="strict",
                                field_prompt=r"Safety\s*:\s*", cats=r"Categories\s*:\s*([^\n]*)"),
    "qwen3guard06_loose": dict(kind="generative", mode="guard", hub="Qwen/Qwen3Guard-Gen-0.6B",
                               rev="fada3b2f655b89601929198343c94cd2f64d93cc", verdict="3way_loose",
                               max_new_tokens=128, card_suffix="loose",
                               field_prompt=r"Safety\s*:\s*", cats=r"Categories\s*:\s*([^\n]*)"),
}


_LOADED = {}   # loaded models, reused across sets within one run


def _sigmoid(m):
    return 1.0 / (1.0 + math.exp(-m)) if m > -700 else 0.0


def _thresholds(model_path, revision):
    """T_bal (decision) and T_gate (decision_gate) from the model repo's training_config.json."""
    p = pathlib.Path(model_path) / "training_config.json"
    if not p.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(model_path, "training_config.json", revision=revision)
    t = json.loads(pathlib.Path(p).read_text())["thresholds"]
    return t["T_bal"]["value"], t["T_gate"]["value"]


def score_release2(cfg, items, dev, batch_size=64):
    """FP32 weights and activations, TF32 off, prompt text only, truncation at 512 tokens, batches of 64 sorted by
    character length (stable); margin = logits[:,1] - logits[:,0] in float64; score = sigmoid(margin)."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    key = ("release2", cfg["hub"], cfg.get("rev"), str(dev))
    if key not in _LOADED:
        kw = {"revision": cfg["rev"]} if cfg.get("rev") else {}
        tok = AutoTokenizer.from_pretrained(cfg["hub"], **kw)
        model = AutoModelForSequenceClassification.from_pretrained(cfg["hub"], dtype=torch.float32, **kw).to(dev).eval()
        if model.config.id2label.get(1, model.config.id2label.get("1")) != "unsafe":
            raise SystemExit("expected label 1 = unsafe")
        _LOADED[key] = (tok, model, *_thresholds(cfg["hub"], cfg.get("rev")))
    tok, model, t_bal, t_gate = _LOADED[key]
    texts = [it["text"] or " " for it in items]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    margins = [None] * len(texts)
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", truncation=True, max_length=ENCODER_MAX_LEN,
                      padding=True).to(dev)
            lg = model(**enc).logits.float()
            for i, m in zip(idx, (lg[:, 1] - lg[:, 0]).double().tolist()):
                margins[i] = m
    rows = [{"score": _sigmoid(m), "margin": m, "decision": int(_sigmoid(m) >= t_bal),
             "decision_gate": int(_sigmoid(m) >= t_gate)} for m in margins]
    return rows, {"input": "prompt string", "precision": "fp32, tf32 off", "batch": batch_size,
                  "transformers": transformers.__version__, "torch": torch.__version__,
                  "thresholds": {"T_bal": t_bal, "T_gate": t_gate}}


# ---- generative guards: greedy decode, then score the verdict token's logits -----------------------------------
_STRIP = " \t\r\n\"'`“”‘’"
VERDICT_WORDS = {"binary": ("unsafe", "safe"), "3way_strict": ("unsafe", "safe", "controversial"),
                 "3way_loose": ("unsafe", "safe", "controversial")}


def vocab_partition_words(tok, words):
    """One token-id list per verdict word: the token's text, leading whitespace/quotes stripped and lower-cased, is
    a non-empty prefix of the word. Words start with distinct letters, so the lists are disjoint."""
    parts = [[] for _ in words]
    for tid in range(len(tok)):
        piece = tok.decode([tid]).lstrip(_STRIP).lower()
        if piece:
            for k, w in enumerate(words):
                if w.startswith(piece):
                    parts[k].append(tid)
                    break
    if any(not p for p in parts):
        raise SystemExit(f"degenerate verdict partition: {dict(zip(words, map(len, parts)))}")
    return parts


def verdict_sides(verdict, parts):
    """(positive ids, negative ids, positive verdict values). Strict counts Controversial as unsafe, loose as safe."""
    if verdict == "binary":
        return parts[0], parts[1], ("unsafe",)
    u, s, c = parts
    return (u + c, s, ("unsafe", "controversial")) if verdict == "3way_strict" else (u, s + c, ("unsafe",))


def locate_value_step(tok, gen_ids, field_regex, words):
    """(decode step whose token starts the verdict value, the value) or (None, None) if the field is absent."""
    text = tok.decode(gen_ids, skip_special_tokens=True)
    alt = "|".join(sorted(words, key=len, reverse=True))
    m = re.search(field_regex + r"(?P<val>" + alt + r")\b", text, flags=re.IGNORECASE)
    if not m:
        return None, None
    pos = m.start("val")
    for k in range(len(gen_ids)):
        if len(tok.decode(gen_ids[:k + 1], skip_special_tokens=True)) > pos:
            return k, m.group("val").lower()
    return None, None


def score_generative(cfg, items, dev, max_batch_tokens=24000, max_batch=32):
    """margin_v12 = logsumexp(positive-start logits) - logsumexp(negative-start logits) at the verdict step.
    For the 3-way guards, margin (v1.3) compares group log-sum-exps: strict max(U, C) - S, loose U - max(S, C)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    verdict, words = cfg["verdict"], VERDICT_WORDS[cfg["verdict"]]
    key = ("guard", cfg["hub"], cfg["rev"], verdict, str(dev))
    if key not in _LOADED:
        tok = AutoTokenizer.from_pretrained(cfg["hub"], revision=cfg["rev"])
        model = AutoModelForCausalLM.from_pretrained(cfg["hub"], revision=cfg["rev"],
                                                     dtype=torch.bfloat16).to(dev).eval()
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _LOADED[key] = (tok, model, vocab_partition_words(tok, words))
    tok, model, parts = _LOADED[key]
    pos_ids, neg_ids, pos_values = verdict_sides(verdict, parts)
    classified = set().union(*map(set, parts))
    pos_t, neg_t = torch.tensor(pos_ids), torch.tensor(neg_ids)
    trunc = [0]

    def prompt(text):
        ids = tok(text or " ", add_special_tokens=False)["input_ids"]
        if len(ids) > GEN_CONTENT_MAX_TOKENS:
            trunc[0] += 1
            text = tok.decode(ids[:GEN_CONTENT_MAX_TOKENS])
        return tok.apply_chat_template([{"role": "user", "content": text or " "}], tokenize=False)

    prompts = [prompt(it["text"]) for it in items]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lens[i])
    eos = model.generation_config.eos_token_id
    stop = {tok.pad_token_id} | (set(eos) if isinstance(eos, list) else {eos})
    bf = lambda x: torch.tensor(x).to(torch.bfloat16).float().item()   # noqa: E731
    rows, b = [None] * len(prompts), 0
    while b < len(order):
        e = b + 1
        while e < len(order) and e - b < max_batch and lens[order[e]] * (e - b + 1) <= max_batch_tokens:
            e += 1
        idx = order[b:e]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=cfg.get("max_new_tokens", MAX_NEW_TOKENS), do_sample=False,
                                 temperature=None, top_p=None, top_k=None, output_logits=True,
                                 return_dict_in_generate=True, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        plen = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            gen = out.sequences[j, plen:].tolist()
            gen = gen[:next((k for k, t in enumerate(gen) if t in stop), len(gen))]
            text = tok.decode(gen, skip_special_tokens=True)
            step, val = locate_value_step(tok, gen, cfg["field_prompt"], words)
            row = {"score": None, "margin": None, "decision": 0, "parse": "fail", "gen_value": val, "nc_only": False}
            if step is not None:
                if gen[step] in classified:
                    lrow = out.logits[step][j]
                    m = float(torch.logsumexp(lrow[pos_ids].float(), 0) - torch.logsumexp(lrow[neg_ids].float(), 0))
                    row["score"], row["margin"] = _sigmoid(m), m
                    lu = lrow[pos_t.to(lrow.device)].max().to(torch.bfloat16).float().item()
                    ls = lrow[neg_t.to(lrow.device)].max().to(torch.bfloat16).float().item()
                    row.update(l_unsafe_top=lu, l_safe_top=ls, tie=lu == ls)
                    if len(words) > 2:
                        lse = {w: float(torch.logsumexp(lrow[p].float(), 0)) for w, p in zip(words, parts)}
                        row["lse_parts"] = lse
                        row["margin_v12"], row["score_v12"], row["tie_v12"] = row["margin"], row["score"], row["tie"]
                        u, s, c = lse["unsafe"], lse["safe"], lse["controversial"]
                        hi, lo = (max(u, c), s) if verdict == "3way_strict" else (u, max(s, c))
                        row["margin"] = hi - lo
                        row["score"] = _sigmoid(row["margin"])
                        row["tie"] = row["tie_v13"] = bf(hi) == bf(lo)
                    row["parse"] = "ok"
                else:
                    row["parse"] = "value_token_unclassified"
                row["decision"] = int(val in pos_values)
                cm = re.search(cfg["cats"], text)
                cats = [c.strip() for c in cm.group(1).split(",") if c.strip()] if cm else []
                row["nc_only"] = row["decision"] == 1 and cats == ["Needs Caution"]
            rows[i] = row
        b = e
        print(f"  {sum(r is not None for r in rows)}/{len(rows)}", flush=True)
    return rows, {"truncated_fields": trunc[0], "field_regex": cfg["field_prompt"], "input": "prompt only",
                  "max_new_tokens": cfg.get("max_new_tokens", MAX_NEW_TOKENS), "dtype": "bfloat16", "decode": "greedy",
                  "verdict": verdict, "partition_sizes": {w: len(p) for w, p in zip(words, parts)}}


# ------------------------------------------------------------------------------------------------ run
def score(model_key, cfg, sets, cap, dev, out_dir):
    import torch
    hw = torch.cuda.get_device_name(0) if str(dev).startswith("cuda") else str(dev)
    written = []
    for sk in sets:
        items = load_set(sk)[:cap] if cap else load_set(sk)
        print(f"[score] {model_key} on {sk}: n={len(items)} ({hw})", flush=True)
        t0 = time.time()
        rows, meta = (score_release2 if cfg["kind"] == "release2" else score_generative)(cfg, items, dev)
        out_rows = []
        for i, (it, r) in enumerate(zip(items, rows)):
            row = {"i": i, "src": it["src"], "label": it["label"], "latency_ms": None, **r}
            row.update({k: v for k, v in it["aux"].items()})
            out_rows.append(row)
        meta.update({"hub_id": cfg["hub"], "revision": cfg.get("rev") or "main", "model_key": model_key,
                     "hardware": hw, "wall_seconds": round(time.time() - t0, 1), "cap": cap,
                     "scorer": "score_test_sets.py"})
        name = cfg.get("card_model") or (cfg["hub"] + (f"-{cfg['card_suffix']}" if cfg.get("card_suffix") else ""))
        card = {"mode": cfg["mode"], "model": name, "set": sk, "role": SETS[sk], "kind": "prompt",
                "n": len(out_rows), "rows": out_rows, "meta": meta}
        p = out_dir / f"{cfg['mode']}__{name.replace('/', '_')}__{sk}.json"
        p.write_text(json.dumps(card))
        written.append(p)
        print(f"[done] {sk}: {time.time() - t0:.0f}s -> {p}", flush=True)
    return written


def compare(written, bundle, cfg, edge=1e-4):
    """Compare fresh scorecards with the bundle's. Returns the number of failing scorecards."""
    sc = pathlib.Path(bundle) / "scorecards" / "test"
    margin_tol = 0.02 if cfg["kind"] == "release2" else 0.5
    fails = 0
    for p in written:
        mine = json.loads(p.read_text())
        ref_p = sc / p.name
        if not ref_p.exists():
            print(f"[compare] {p.name}: not in the bundle; skipped")
            continue
        ref = json.loads(ref_p.read_text())
        ref_rows = ref["rows"][:len(mine["rows"])]
        problems = []
        if [(r["i"], r["src"]) for r in mine["rows"]] != [(r["i"], r["src"]) for r in ref_rows]:
            problems.append("row order or source index differs")
        if "label" in ref_rows[0] and [r["label"] for r in mine["rows"]] != [r["label"] for r in ref_rows]:
            problems.append("labels differ")
        if "labels_removed" in ref:
            print(f"[compare] {p.name}: the bundle's labels are stripped here; labels not compared "
                  f"(run attach_labels.py first to compare them)")
        pairs = [(a, b) for a, b in zip(mine["rows"], ref_rows)
                 if a.get("margin") is not None and b.get("margin") is not None]
        dmax = max((abs(a["margin"] - b["margin"]) for a, b in pairs), default=0.0)
        if dmax > margin_tol:
            problems.append(f"largest margin difference {dmax:.4g} > {margin_tol}")
        thr = (ref.get("meta") or {}).get("thresholds", {})
        flips = 0
        for key, tname in (("decision", "T_bal"), ("decision_gate", "T_gate")):
            for a, b in zip(mine["rows"], ref_rows):
                if key in a and key in b and a[key] != b[key]:
                    t = thr.get(tname)
                    if cfg["kind"] != "release2" or t is None or a.get("score") is None or abs(a["score"] - t) > edge:
                        flips += 1
        if flips:
            problems.append(f"{flips} decisions differ" + (" away from the threshold" if cfg["kind"] == "release2" else ""))
        for k in ("gen_value", "parse"):
            n_k = sum(1 for a, b in zip(mine["rows"], ref_rows) if k in b and a.get(k) != b.get(k))
            if n_k:
                problems.append(f"{n_k} rows differ in {k}")
        missing = sum(1 for a, b in zip(mine["rows"], ref_rows) if (a.get("margin") is None) != (b.get("margin") is None))
        if missing:
            problems.append(f"{missing} rows parsed on one side only")
        status = "FAIL" if problems else "ok"
        fails += bool(problems)
        print(f"[compare] {status} {p.name}: {len(mine['rows'])} rows, largest margin difference {dmax:.3g}"
              + (f"; {'; '.join(problems)}" if problems else ""))
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="release2", choices=["release2", *GUARDS],
                    help="release2 (the published weights; default) or an open comparison guard")
    ap.add_argument("--weights", default=RELEASE2["hub"],
                    help="release2 only: Hub id or local folder of the weights (with training_config.json)")
    ap.add_argument("--revision", default=None, help="release2 only: Hub revision of the weights")
    ap.add_argument("--sets", default=",".join(SETS), help="comma-separated set keys (default: all twelve)")
    ap.add_argument("--cap", type=int, default=None, help="score only the first N prompts of each set")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda if available)")
    ap.add_argument("--out", required=True, help="folder for the new scorecards")
    ap.add_argument("--compare", default=None, help="evidence bundle folder to compare with (labelled copy best)")
    a = ap.parse_args()
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(RELEASE2, hub=a.weights, rev=a.revision) if a.model == "release2" else GUARDS[a.model]
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    bad = [s for s in sets if s not in SETS]
    if bad:
        raise SystemExit(f"unknown sets {bad}; known: {list(SETS)}")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = score(a.model, cfg, sets, a.cap, dev, out)
    if a.compare:
        sys.exit(1 if compare(written, a.compare, cfg) else 0)


if __name__ == "__main__":
    main()
