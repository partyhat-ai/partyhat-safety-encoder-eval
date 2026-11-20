"""time_encoder.py: the timing code that produced latency/release_2_A10_fp32.json.

These are the functions exactly as they ran, on Modal, on the published weights (model.safetensors sha256
e688e56c...). The Modal launcher and the dataset loader (`baselines_scorecard`) are not included, so this
file documents the method rather than running on its own.

Method: FP32, batch size 1. Inputs are OR-Bench-Hard prompts joined and cut to exactly 128 or 512 tokens
(special tokens included). One cold call is recorded separately, then 100 warm-up requests, then 1,000 timed
requests per length. Each time is one forward pass, from tokenised tensors already on the GPU to the logits,
with torch.cuda.synchronize() before and after; tokenisation is not timed. Percentiles are nearest-rank.
Static-batch throughput (batches 1-64, 50 timed batches each) is recorded alongside.
Environment: torch 2.12.1, transformers 5.12.1, Python 3.11; the GPU reported itself as "NVIDIA A10".
"""
import time

LENGTHS = (128, 512)
BATCHES = (1, 8, 16, 32, 64)
WARMUP, TIMED, TIMED_BATCHES = 100, 1000, 50
P95_TARGET_MS = 100.0


def _pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def _summary(ms):
    return {"n": len(ms), "p50": _pct(ms, .5), "p95": _pct(ms, .95), "p99": _pct(ms, .99),
            "mean": round(sum(ms) / len(ms), 2)}


def make_inputs(tok, n_tokens, count):
    """`count` distinct strings of exactly n_tokens content tokens from OR-Bench-Hard prompts."""
    import baselines_scorecard as bs   # loads OR-Bench-Hard-1K prompts (not included here)
    prompts = [it["text"] for it in bs.load_set("orbench_hard")]
    ids_all = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    out, j = [], 0
    while len(out) < count:
        buf = []
        while len(buf) < n_tokens:
            buf += ids_all[j % len(ids_all)] + tok(" ", add_special_tokens=False)["input_ids"]
            j += 1
        out.append(tok.decode(buf[:n_tokens]))
    return out


def _sync(dev):
    import torch
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def bench_encoder(key, dtype_name, dev="cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import baselines_scorecard as bs
    cfg = bs.get_cfg(key)
    if cfg.get("path"):   # v6 checkpoint on the read-only volume (tokenizer alongside)
        tok = AutoTokenizer.from_pretrained(cfg["path"])
        model = AutoModelForSequenceClassification.from_pretrained(cfg["path"])
    else:
        # v1.3 peers: DuoGuard uses the card's Qwen2.5 tokenizer with pad = eos and no special tokens
        tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", cfg["hub"]),
                                            revision=cfg.get("tokenizer_rev", cfg["rev"]))
        model = AutoModelForSequenceClassification.from_pretrained(cfg["hub"], revision=cfg["rev"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    model.to(dev, dtype=dtype).eval()
    n_special = len(tok("x")["input_ids"]) - len(tok("x", add_special_tokens=False)["input_ids"])
    res = {}
    for n in LENGTHS:
        texts = make_inputs(tok, n - n_special, TIMED + WARMUP + 64 * 2)   # specials -> exactly n

        def run(batch):
            enc = tok(batch, return_tensors="pt", truncation=True, max_length=n,
                      padding="max_length").to(dev)
            _sync(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**enc)
            _sync(dev)
            return (time.perf_counter() - t0) * 1000

        cold = run(texts[:1])
        for t in texts[:WARMUP]:
            run([t])
        bs1 = [run([t]) for t in texts[WARMUP:WARMUP + TIMED]]
        res[n] = {"cold_ms": round(cold, 1), "bs1": _summary(bs1), **_throughput(run, texts)}
    return res


def _throughput(run, texts):
    table, best = {}, None
    for b in BATCHES:
        batch = texts[:b]
        try:
            for _ in range(5):
                run(batch)
            ms = [run(batch) for _ in range(TIMED_BATCHES)]
        except Exception as e:   # CUDA OOM: this and larger batches are infeasible here
            if "out of memory" not in str(e).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            table[b] = {"oom": True}
            break
        p95 = _pct(ms, .95)
        table[b] = {"p95_ms": p95, "items_per_s": round(b / (sum(ms) / len(ms) / 1000), 1)}
        if p95 <= P95_TARGET_MS:
            best = b
    fit = [b for b in table if "items_per_s" in table[b]]
    top = max(fit, key=lambda b: table[b]["items_per_s"]) if fit else None
    return {"batches": table, "best_batch_at_p95_target": best,
            "items_per_s_at_target": table[best]["items_per_s"] if best else None,
            # unconstrained: best items/s over the batches that fit in memory (labelled)
            "items_per_s_unconstrained": table[top]["items_per_s"] if top else None,
            "unconstrained_batch": top}
