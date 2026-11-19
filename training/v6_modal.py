#!/usr/bin/env python
"""v6_modal.py: the Modal launcher for building, training and dev-scoring the release-2.0 encoder.

Writes under /vol/v6/ on the private Volume VOL_NAME (env V6_VOLUME). Only JSON (counts, hashes,
ids, scores, margins, labels) comes back; text and weights stay on the Volume.

  modal run v6_modal.py --action tests        # the v6 unit tests, on the training image
  modal run v6_modal.py --action inspect      # source columns / counts / revisions (no text)
  modal run v6_modal.py --action build        # the five data pools, the R0 file, dev sets, decontamination
  modal run v6_modal.py --action crossfit     # shortcut audit + bag-of-words files for R2
  modal run v6_modal.py --action smoke        # 300-step base-recipe trial run (weights discarded)
  modal run v6_modal.py --action baseline     # release 1.0 trial checkpoints on the dev sets
  modal run v6_modal.py --action wave --arms R1,R0 --seeds 13,42,99 --steps S
  modal run v6_modal.py --action fetch        # pull JSON results to v6_results/

`--arms` takes recipe-variant keys (R0-R4); RECIPE_NAMES in v6_score.py says what each one is.
"""
import json
import os
import pathlib
import time

import modal

from train_image import CODE_DIR, image as train_image

VOL_NAME = os.environ.get("V6_VOLUME", "v6-data")
APP_NAME = os.environ.get("V6_APP", "v6-train")
MNT = "/vol"
V6 = f"{MNT}/v6"
GPU = "A100-80GB"
MAX_CONTAINERS = 20
LOCAL_DIR = pathlib.Path(__file__).parent
RESULTS_LOCAL = LOCAL_DIR / "v6_results"

_IGNORE = ["__pycache__", "checkpoints", ".git", "*.pyc", "*.jsonl", "*.csv", "label_sheets",
           "chart-app", "demo", "leaderboard", "external_rerun", "harness",
           "release", "*.png", "v6_results"]

# data-build image: sentence-transformers for the MiniLM pass (not the training stack)
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "sentence-transformers>=3.0", "datasets>=2.18",
                 "huggingface_hub>=0.23", "numpy>=1.24", "pandas>=2.0", "scikit-learn>=1.3",
                 "requests>=2.31")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("train_image")
    .add_local_dir(LOCAL_DIR, remote_path=CODE_DIR, ignore=_IGNORE)
)

app = modal.App(APP_NAME, image=train_image)
vol = modal.Volume.from_name(VOL_NAME)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _enter_code_dir():
    """Run from the mounted copy of this folder, so its modules import and relative paths work."""
    import sys
    sys.path.insert(0, CODE_DIR)
    os.chdir(CODE_DIR)


# --------------------------------------------------------------------------------------------- #
# tests                                                                                           #
# --------------------------------------------------------------------------------------------- #
@app.function(cpu=4, timeout=30 * 60)
def run_tests():
    import subprocess
    import sys

    import torch
    import transformers
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "test_v6_bow.py", "test_v6_committee.py", "test_v6_metrics.py", "test_v6_score.py", "test_v6_build.py"],
                       cwd=CODE_DIR, capture_output=True, text=True)
    return {"returncode": r.returncode, "tail": r.stdout[-6000:] + r.stderr[-2000:],
            "transformers": transformers.__version__, "torch": torch.__version__}


# --------------------------------------------------------------------------------------------- #
# source inspection (no text returned)                                                            #
# --------------------------------------------------------------------------------------------- #
@app.function(image=prep_image, secrets=[hf_secret], cpu=4, memory=32768, timeout=40 * 60)
def inspect_sources():
    _enter_code_dir()
    import v6_data as vd
    return vd.inspect_all()


# --------------------------------------------------------------------------------------------- #
# data build                                                                                      #
# --------------------------------------------------------------------------------------------- #
VAL_SPLIT_PREFIX = "e890c3fe"   # first characters of the release's ToxicChat validation-split hash


@app.function(volumes={MNT: vol}, secrets=[hf_secret], cpu=4, memory=16384, timeout=30 * 60)
def build_p1():
    """The ToxicChat pool (P1, p1_copy) = exactly train.py's training split (training image, same
    datasets/sklearn as train.py), and tc_val = its validation split. Asserts the split's hash prefix."""
    _enter_code_dir()
    from tc_common import INPUT_FIELD, LABEL_FIELD, load_train_val
    from train import split_hash
    from v6_build import write_jsonl
    tr, va = load_train_val(val_fraction=0.10, seed=13)
    h = split_hash(va)
    assert h.startswith(VAL_SPLIT_PREFIX), f"val split hash {h} does not start {VAL_SPLIT_PREFIX}"
    p1 = [{"row_id": f"tc_train:{i}", "user_input": str(t), "toxicity": int(y), "pool": "P1",
           "type": "toxicchat", "toxicity_soft": float(int(y))}
          for i, (t, y) in enumerate(zip(tr[INPUT_FIELD], tr[LABEL_FIELD]))]
    tv = [{"row_id": f"tc_val:{i}", "user_input": str(t), "toxicity": int(y), "pool": "tc_val",
           "type": "toxicchat"} for i, (t, y) in enumerate(zip(va[INPUT_FIELD], va[LABEL_FIELD]))]
    out = {"val_carve_sha256": h, "train_carve_sha256": split_hash(tr),
           "p1_copy": write_jsonl(f"{V6}/pools/p1_copy.jsonl", p1),
           "tc_val": write_jsonl(f"{V6}/dev/tc_val.jsonl", tv)}
    vol.commit()
    return out


@app.function(image=prep_image, volumes={MNT: vol}, secrets=[hf_secret], gpu="A10G", cpu=16,
              memory=65536, timeout=3 * 60 * 60)
def build_data():
    _enter_code_dir()
    import v6_build
    vol.reload()
    t0 = time.time()
    man = v6_build.build_all(MNT, V6, procs=16)
    man["build_seconds"] = round(time.time() - t0, 1)
    os.makedirs(f"{V6}/manifest", exist_ok=True)
    with open(f"{V6}/manifest/build_manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    vol.commit()
    return man


@app.function(image=prep_image, volumes={MNT: vol}, gpu="A10G", cpu=16, memory=65536,
              timeout=2 * 60 * 60)
def reverse_audit_job():
    _enter_code_dir()
    import v6_build
    vol.reload()
    out = v6_build.reverse_audit(V6, procs=16)
    with open(f"{V6}/manifest/reverse_audit.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.function(image=prep_image, volumes={MNT: vol}, secrets=[hf_secret], gpu="A10G", cpu=16,
              memory=98304, timeout=3 * 60 * 60)
def build_r3_data():
    _enter_code_dir()
    import v6_build
    vol.reload()
    t0 = time.time()
    man = v6_build.build_r3(MNT, V6, procs=16)
    man["build_seconds"] = round(time.time() - t0, 1)
    with open(f"{V6}/manifest/r3_manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    vol.commit()
    return man


@app.function(image=prep_image, volumes={MNT: vol}, gpu="A10G", cpu=8, memory=16384,
              timeout=20 * 60)
def p1_matched_heldout_job():
    _enter_code_dir()
    import v6_build
    vol.reload()
    return v6_build.p1_matched_heldout(V6, procs=8)


@app.function(image=prep_image, volumes={MNT: vol}, gpu="A10G", cpu=16, memory=32768,
              timeout=30 * 60)
def filter_aegis_val_job():
    _enter_code_dir()
    import v6_build
    vol.reload()
    out = v6_build.filter_aegis_val(V6, procs=16)
    with open(f"{V6}/manifest/aegis_val_filter.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.function(volumes={MNT: vol}, cpu=4, memory=32768, timeout=30 * 60)
def build_r3_training_job(targets_path):
    _enter_code_dir()
    import v6_build
    vol.reload()
    rep = v6_build.build_r3_training(V6, targets_path)
    with open(f"{V6}/manifest/r3_training.json", "w") as f:
        json.dump(rep, f, indent=1)
    vol.commit()
    return rep


# --------------------------------------------------------------------------------------------- #
# base weights pinned by revision (train.py has no --revision flag: pass a local snapshot)        #
# --------------------------------------------------------------------------------------------- #
BASE_DIR = f"{V6}/base/deberta-v3-large"


def _ensure_base():
    from huggingface_hub import snapshot_download
    from v6_score import BASE_MODEL, BASE_REVISION
    marker = f"{BASE_DIR}/REVISION"
    if os.path.exists(marker) and open(marker).read().strip() == BASE_REVISION:
        return BASE_DIR
    snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_dir=BASE_DIR,
                      allow_patterns=["*.json", "*.model", "*.txt", "*.safetensors", "*.bin"])
    pathlib.Path(marker).write_text(BASE_REVISION)
    vol.commit()
    return BASE_DIR


# --------------------------------------------------------------------------------------------- #
# shortcut audit + the bag-of-words log-odds for R2 (base recipe + bag-of-words correction)      #
# --------------------------------------------------------------------------------------------- #
@app.function(volumes={MNT: vol}, cpu=16, memory=65536, timeout=2 * 60 * 60)
def crossfit():
    _enter_code_dir()
    import v6_bow_crossfit
    vol.reload()
    names = ["p1_copy", "p2", "p3", "p4", "p5"]
    rep = v6_bow_crossfit.main(["--inputs", *[f"{V6}/pools/{n}.jsonl" for n in names],
                                "--out_dir", f"{V6}/r2", "--report", f"{V6}/r2/bow_crossfit_report.json"])
    from v6_build import sha_file
    rep["r2_files"] = {n: sha_file(f"{V6}/r2/{n}.jsonl") for n in names}
    vol.commit()
    return rep


# --------------------------------------------------------------------------------------------- #
# training + dev scoring                                                                          #
# --------------------------------------------------------------------------------------------- #
def _load_model(path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to("cuda")
    return model, tok


def _score_dev(model_path, run, out_dir, names=None):
    from v6_build import sha_file
    from v6_metrics import dev_report
    from v6_score import DEV_SETS, dev_sets_for_report, dump, load_dev, score_texts, scorecard
    model, tok = _load_model(model_path)
    dev = load_dev(V6, names or DEV_SETS)
    cards = {}
    for n, rows in dev.items():
        p, m = score_texts(model, tok, [r["user_input"] for r in rows], "cuda")
        cards[n] = scorecard(run, n, rows, p, m,
                             meta={"file_sha256": sha_file(f"{V6}/dev/{n}.jsonl")})
        dump(f"{out_dir}/scorecards/{n}.json", cards[n])
    rep = dev_report(dev_sets_for_report(cards))
    del model
    return rep


def _train(variant, seed, steps, batch, accum, work, smoke=False):
    """Runs train.py, streaming its output; returns (returncode, log_tail, step_times, peak_mem)."""
    import re
    import subprocess
    import sys
    import threading

    from v6_score import build_train_cmd
    base = _ensure_base()
    cmd = build_train_cmd(sys.executable, variant, seed, V6, base, f"{work}/model",
                          f"{work}/run_meta.json", steps, batch, accum)
    print("[cmd]", " ".join(cmd[2:]), flush=True)
    peak = [0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
                peak[0] = max(peak[0], int(o.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            stop.wait(5)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    step_t, buf, tail = {}, b"", []
    pat = re.compile(rb"\s(\d+)/(\d+) \[")
    while True:
        ch = pr.stdout.read1(65536) if hasattr(pr.stdout, "read1") else pr.stdout.read(4096)
        if not ch:
            break
        buf += ch
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        now = time.time()
        for ln in parts:
            mm = pat.search(ln)
            if mm and int(mm.group(2)) == steps:
                step_t.setdefault(int(mm.group(1)), now)
            elif ln.strip():
                tail.append(ln.decode("utf-8", "replace"))
                tail = tail[-200:]
                if not smoke:
                    print(ln.decode("utf-8", "replace"), flush=True)
    rc = pr.wait()
    stop.set()
    return rc, "\n".join(tail), step_t, peak[0]


@app.function(gpu=GPU, volumes={MNT: vol}, secrets=[hf_secret], timeout=2 * 60 * 60)
def smoke(steps=300, batch=32, accum=1):
    """A 300-step base-recipe (R1) trial run, seed 13, to check speed and memory. Weights are
    discarded (never leave /tmp, never scored)."""
    import shutil
    _enter_code_dir()
    vol.reload()
    work = "/tmp/v6_smoke"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    rc, tail, st, peak = _train("R1", 13, steps, batch, accum, work, smoke=True)
    out = {"returncode": rc, "batch": batch, "grad_accum": accum, "peak_gpu_mem_mib": peak,
           "oom": "out of memory" in tail.lower(), "log_tail": tail[-3000:]}
    k0 = min([k for k in st if k >= 100], default=None)
    if k0 is not None and steps in st and k0 < steps:
        out["r_steps_per_s_100_to_end"] = (steps - k0) / (st[steps] - st[k0])
        out["r_from_step"] = k0
    out["step_times_rel"] = {k: round(v - min(st.values()), 2) for k, v in sorted(st.items())
                             if k in (1, 50, 100, 150, 200, 250, 300)} if st else {}
    if rc == 0 and os.path.exists(f"{work}/run_meta.json"):
        rm = json.load(open(f"{work}/run_meta.json"))
        out["run_meta"] = {k: v for k, v in rm.items() if k != "log_history"}
        out["log_history_tail"] = rm.get("log_history", [])[-4:]
    shutil.rmtree(work, ignore_errors=True)   # weights discarded
    return out


@app.function(gpu=GPU, volumes={MNT: vol}, timeout=60 * 60, max_containers=4)
def score_trial_checkpoint(run):
    """Score one release 1.0 trial checkpoint on the v6 dev sets (margins stored). The mean intent
    score of these runs is the release 1.0 trial intent score that v6_select.py compares against."""
    _enter_code_dir()
    vol.reload()
    rep = _score_dev(f"{MNT}/runs/{run}/model", run, f"{V6}/baseline/{run}")
    from v6_score import dump
    dump(f"{V6}/baseline/{run}/dev_report.json", rep)
    vol.commit()
    return {"run": run, **rep}


@app.function(gpu=GPU, volumes={MNT: vol}, secrets=[hf_secret], timeout=4 * 60 * 60,
              max_containers=MAX_CONTAINERS)
def train_and_score(variant, seed, steps, batch=32, accum=1):
    import shutil
    _enter_code_dir()
    from v6_score import dump, run_name
    run = run_name(variant, seed)
    rdir = f"{V6}/runs/{run}"
    vol.reload()
    if os.path.exists(f"{rdir}/DONE"):
        return json.load(open(f"{rdir}/summary.json")) | {"skipped": True}
    work = f"/tmp/{run}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    t0 = time.time()
    rc, tail, st, peak = _train(variant, seed, steps, batch, accum, work)
    if rc:
        raise RuntimeError(f"{run}: train.py exited {rc}\n{tail[-3000:]}")
    train_s = time.time() - t0
    for d in pathlib.Path(f"{work}/model").glob("checkpoint-*"):
        shutil.rmtree(d)
    os.makedirs(rdir, exist_ok=True)
    shutil.rmtree(f"{rdir}/model", ignore_errors=True)
    shutil.copytree(f"{work}/model", f"{rdir}/model")
    rm = json.load(open(f"{work}/run_meta.json"))
    t1 = time.time()
    rep = _score_dev(f"{work}/model", run, rdir)
    import torch
    import transformers
    summary = {"run": run, "arm": variant, "seed": seed, "steps": steps, "batch": batch,
               "grad_accum": accum, "dev": rep,
               "run_meta": {k: v for k, v in rm.items() if k != "log_history"},
               "timing_s": {"train": round(train_s, 1), "score": round(time.time() - t1, 1)},
               "peak_gpu_mem_mib": peak, "gpu_name": torch.cuda.get_device_name(0),
               "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
    dump(f"{rdir}/run_meta.json", rm)
    dump(f"{rdir}/summary.json", summary)
    pathlib.Path(f"{rdir}/DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    vol.commit()
    return summary


RELEASE_LEAK = ("threshold.json", "training_args.bin", "results.md", "optimizer.pt", "scheduler.pt",
                "trainer_state.json", "all_results.json", "eval_results.json")


@app.function(gpu=GPU, volumes={MNT: vol}, timeout=60 * 60)
def stage_release(run, target, training_config):
    """Stage the release files for `run` under /v6/release/<run>/stage on the private Volume.
    No Hugging Face call is made. Then reload the staged directory through the v6 scoring path and
    compare its tc_val and aegis_val margins with the run's stored dev scorecards."""
    import hashlib
    import shutil

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _enter_code_dir()
    from v6_score import load_dev, score_texts
    vol.reload()
    src, out = f"{V6}/runs/{run}/model", f"{V6}/release/{run}/stage"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(src)
    src_id2label = {int(k): v for k, v in model.config.id2label.items()}
    assert src_id2label == {0: "non_toxic", 1: "toxic"}, f"unexpected id2label {src_id2label}"
    model.config.id2label = {0: "safe", 1: "unsafe"}
    model.config.label2id = {"safe": 0, "unsafe": 1}
    model.config._name_or_path = target
    if hasattr(model.config, "name_or_path"):
        model.config.name_or_path = target
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    tcp = f"{out}/tokenizer_config.json"
    if os.path.exists(tcp):
        tc = json.load(open(tcp))
        if tc.pop("is_local", None) is not None:
            with open(tcp, "w") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
                f.write("\n")
    with open(f"{out}/training_config.json", "w") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for leaky in RELEASE_LEAK:
        if os.path.exists(f"{out}/{leaky}"):
            os.remove(f"{out}/{leaky}")
    cfg_text = open(f"{out}/config.json").read()
    assert "/vol/" not in cfg_text and "/tmp/" not in cfg_text, "a local path leaked into config.json"
    n_params = int(sum(p.numel() for p in model.parameters()))
    del model

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    files = {f: {"bytes": os.path.getsize(f"{out}/{f}"), "sha256": sha(f"{out}/{f}")}
             for f in sorted(os.listdir(out))}
    # parity: the staged directory through the same scoring path as the stored dev scorecards
    m2 = AutoModelForSequenceClassification.from_pretrained(out, dtype=torch.float32).to("cuda")
    t2 = AutoTokenizer.from_pretrained(out)
    dev = load_dev(V6, ("tc_val", "aegis_val"))
    parity = {}
    for n, rows in dev.items():
        _, m = score_texts(m2, t2, [r["user_input"] for r in rows], "cuda")
        ref = json.load(open(f"{V6}/runs/{run}/scorecards/{n}.json"))["rows"]
        assert [r["src"] for r in ref] == [r["row_id"] for r in rows], n
        d = np.abs(m - np.array([r["margin"] for r in ref]))
        parity[n] = {"n": len(rows), "max_abs_margin_diff": float(d.max()),
                     "bit_identical": bool((d == 0).all())}
    src_w = sha(f"{src}/model.safetensors")
    vol.commit()
    return {"run": run, "stage": out, "target": target, "files": files, "n_params": n_params,
            "source_model_safetensors_sha256": src_w,
            "weights_byte_identical_to_scored": src_w == files["model.safetensors"]["sha256"],
            "config": json.load(open(f"{out}/config.json")), "parity": parity,
            "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
            "gpu": torch.cuda.get_device_name(0)}


@app.function(volumes={MNT: vol}, timeout=30 * 60)
def collect():
    """JSON results only (summaries, dev reports, scorecards of ids/scores/margins/labels)."""
    vol.reload()
    out = {}
    for base in ("runs", "baseline"):
        for rdir in sorted(pathlib.Path(f"{V6}/{base}").glob("*")):
            files = [p for p in rdir.rglob("*.json") if "model" not in p.relative_to(rdir).parts]
            out[f"{base}/{rdir.name}"] = {str(p.relative_to(rdir)): json.load(open(p)) for p in files}
    return out


@app.local_entrypoint()
def main(action: str = "tests", arms: str = "", seeds: str = "13,42,99", steps: int = 0,
         batch: int = 32, accum: int = 1):
    if action == "tests":
        out = run_tests.remote()
        print(out["tail"])
        print(f"[versions] transformers={out['transformers']} torch={out['torch']}")
        if out["returncode"]:
            raise SystemExit(out["returncode"])
    elif action == "inspect":
        print(json.dumps(inspect_sources.remote(), indent=1))
    elif action == "build":
        p1 = build_p1.remote()
        print(json.dumps(p1, indent=1), flush=True)
        man = build_data.remote()
        man["p1"] = p1
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "build_manifest.json").write_text(json.dumps(man, indent=1))
        print(json.dumps({k: v for k, v in man.items() if k not in ("type_counts",)}, indent=1))
    elif action == "reverse":
        out = reverse_audit_job.remote()
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "reverse_audit.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    elif action == "stage":
        run, target = "R3_s13", "partyhat-hf/partyhat-safety-encoder"
        sel = json.loads((RESULTS_LOCAL / "selection.json").read_text())
        thr = sel["thresholds"][run]
        # training_config.json ships with the released model; re-staging writes the published file unchanged.
        tc = json.loads((LOCAL_DIR / "training_config.json").read_text())
        assert tc["operating_threshold"] == thr["t_bal"]
        out = stage_release.remote(run, target, tc)
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "release_stage_R3_s13.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "config"}, indent=1))
    elif action == "d1items":   # ToxicChat-pool items that match held-out evaluation items
        out = p1_matched_heldout_job.remote()
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "d1_p1_matched_heldout.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    elif action == "aegisval":
        out = filter_aegis_val_job.remote()
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "aegis_val_filter.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    elif action == "r3train":
        rep = build_r3_training_job.remote(f"{V6}/r3/r3_committee/r3_targets.jsonl")
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "r3_training.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
    elif action == "r3build":
        man = build_r3_data.remote()
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "r3_manifest.json").write_text(json.dumps(man, indent=1))
        print(json.dumps(man, indent=1))
    elif action == "crossfit":
        rep = crossfit.remote()
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "bow_crossfit_report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
    elif action == "smoke":
        out = smoke.remote(steps or 300, batch, accum)
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / f"smoke_b{batch}x{accum}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "log_tail"}, indent=1))
        print(out["log_tail"][-2500:])
    elif action == "baseline":
        # the release 1.0 trial checkpoints, by their run folder names on the volume
        res = list(score_trial_checkpoint.map(["A0_s13", "A0_s13_repeat"]))
        keys = ("dev_intent", "twins_auroc", "dev_pair2_auroc", "twins_pair_acc", "tc_val_auprc",
                "aegis_val_auroc", "saturation")
        b = sum(r["dev_intent"] for r in res) / len(res)
        out = {"runs": res, "B": b}
        RESULTS_LOCAL.mkdir(exist_ok=True)
        (RESULTS_LOCAL / "baseline_B.json").write_text(json.dumps(out, indent=1))
        for r in res:
            print(r["run"], {k: round(r[k], 4) for k in keys if r.get(k) is not None})
        print(f"[baseline] release 1.0 trial intent score (B) = {b:.4f}")
    elif action == "wave":
        assert steps > 0 and arms, "--arms and --steps are required"
        jobs = [(a, int(s), steps, batch, accum) for a in arms.split(",") for s in seeds.split(",")]
        print(f"[wave] {len(jobs)} runs: {jobs}", flush=True)
        res = list(train_and_score.starmap(jobs, return_exceptions=True))
        from v6_score import RECIPE_NAMES
        for j, r in zip(jobs, res):
            print(j[:2], RECIPE_NAMES.get(j[0], ""),
                  "FAILED: " + repr(r)[:500] if isinstance(r, Exception)
                  else {k: r["dev"].get(k) for k in ("dev_intent", "tc_val_auprc", "saturation")})
    elif action == "fetch":
        allr = collect.remote()
        for run, files in allr.items():
            for f, obj in files.items():
                q = RESULTS_LOCAL / run / f
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_text(json.dumps(obj))
        print(f"[fetch] {len(allr)} dirs -> {RESULTS_LOCAL}")
    else:
        raise SystemExit(f"unknown action {action}")
