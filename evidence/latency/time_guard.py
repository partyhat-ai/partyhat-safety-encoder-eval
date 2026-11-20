"""time_guard.py: the timing code that produced latency/qwen3guard_gen_0.6b_A10G_bf16.json.

The function exactly as it ran, on Modal (gpu="A10G"), on Qwen/Qwen3Guard-Gen-0.6B @ fada3b2f655b89601929198343c94cd2f64d93cc
in bf16, with the model's own chat template and greedy decoding. It shares its helpers and settings with
time_encoder.py, which timed release 2.0 in the same harness. The Modal launcher and the scoring harness
(`baselines_scorecard`, which loads the model and builds its prompt) are not included, so this file documents the
method rather than running on its own.

Method: inputs are OR-Bench-Hard prompts joined and cut to exactly 128 or 512 tokens of prompt text, inside
Qwen3Guard's template (417 and 781 tokens in total). Batch size 1: one cold call recorded separately, then 100 warm-up
requests, then 1,000 timed requests per length. "decision" (time to verdict) stops generating as soon as the verdict
word after "Safety:" has been produced, which is when a filter can act; "full" generates the whole output, up to 128
new tokens. Tokenisation is not timed. Static-batch throughput is recorded alongside, for "decision".
Environment: torch 2.8.0, transformers 4.57.6, Python 3.11; the GPU reported itself as "NVIDIA A10G" (release 2.0's
container on the same Modal GPU class reported "NVIDIA A10").
"""
import time

from time_encoder import LENGTHS, TIMED, WARMUP, _summary, _sync, _throughput, make_inputs


def bench_generative_hf(key, dev="cuda"):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    import baselines_scorecard as bs   # loads the model and builds its chat prompt (not included here)
    cfg = bs.MODELS[key]
    tok, model = bs._load_generative(key, cfg, dev)

    class VerdictStop(StoppingCriteria):
        def __init__(self, plen):
            self.plen = plen

        def __call__(self, input_ids, scores, **kw):
            texts = tok.batch_decode(input_ids[:, self.plen:], skip_special_tokens=True)
            if "verdict" in cfg:   # stop once the verdict word after the field ("Safety:") is out
                import re
                alt = "|".join(sorted(bs.VERDICT_WORDS[cfg["verdict"]], key=len, reverse=True))
                pat = re.compile(cfg["field_prompt"] + r"(" + alt + r")\b", re.IGNORECASE)
                return torch.tensor([bool(pat.search(t)) for t in texts], device=input_ids.device)
            return torch.tensor([("safe" in t) for t in texts], device=input_ids.device)

    res = {}
    for n in LENGTHS:
        contents = make_inputs(tok, n, TIMED + WARMUP + 64 * 2)
        prompts = [bs.build_prompt(key, tok, {"text": c, "response": None}, [0]) for c in contents]
        total_tokens = len(tok(prompts[0], add_special_tokens=False)["input_ids"])

        def runner(mode):
            def run(batch):
                enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                plen = enc["input_ids"].shape[1]
                kw = {"stopping_criteria": StoppingCriteriaList([VerdictStop(plen)])} \
                    if mode == "decision" else {}
                _sync(dev)
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(**enc, max_new_tokens=cfg.get("max_new_tokens", bs.MAX_NEW_TOKENS),
                                   do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.pad_token_id, **kw)
                _sync(dev)
                return (time.perf_counter() - t0) * 1000
            return run

        entry = {"total_input_tokens": total_tokens}
        for mode in ("decision", "full"):
            run = runner(mode)
            cold = run(prompts[:1])
            for p in prompts[:WARMUP]:
                run([p])
            bs1 = [run([p]) for p in prompts[WARMUP:WARMUP + TIMED]]
            entry[mode] = {"cold_ms": round(cold, 1), "bs1": _summary(bs1)}
            if mode == "decision":
                entry[mode].update(_throughput(run, prompts))
        res[n] = entry
    return res
