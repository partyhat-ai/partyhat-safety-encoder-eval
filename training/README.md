# Training code for partyhat-safety-encoder 2.0

This folder holds the code that built the training data for release 2.0, trained the candidate
encoders on Modal, scored them on the dev sets, and selected the shipped checkpoint. The comments in
the code describe each rule directly.

## Where it comes from

- Everything except `r3_committee/` was taken from commit `e148098` on the private `v6-build` branch.
- `r3_committee/` (the teacher-committee scorer and its test) was taken from the private main
  checkout at commit `d9db3c7`. It uses the `v6_committee_targets.py` in this folder.
- `train.py` started as the base trainer (`train.py` at commit `11dc9e4`) plus `train_v6.patch`
  (sha256 `ba0a9fe9…`), which added three things: `--group_by_length`, a stop on a missing soft
  target, and the bag-of-words correction for R2 (`--bow_logit_field`).
- The published copy is not byte-identical to the private original. It renames internal terms to
  the ones the model card uses (arm → recipe variant, tower → model, two-tower → two-model average,
  replicate → training run, gate → release bar, guardrail → ToxicChat check) and rewords comments
  and printed messages to match, drops section references to unpublished planning documents and
  code the release didn't use, and inlines the small hashing helper behind the selection seal. The training logic for this release is unchanged:
  no data rule, hyperparameter, seed, threshold, metric, selection rule or committee rule was
  changed. Strings stored in data files, run folders or published files (for example the `R3_s13`
  run name, `p4_r0.jsonl`, the `t_bal` key) keep their original form; the glossary below explains
  them.
- Local paths and references to unpublished documents were removed or neutralised, and Modal volume
  and app names became constants that environment variables can override.
- The text that `v6_select.py` writes to `V6_SELECTION.md` (headings, labels, the selection trace and
  the closing "Record") uses the same renamed terms, and the same trace text appears in
  `selection.json`. A regenerated record therefore differs from the sealed one in wording only; the
  selection, bars, thresholds and numbers it computes are unaffected. The `--dev-scorecards` option,
  which re-runs the selection on the evidence bundle, is the one addition.

## Files

| File | What it does |
|---|---|
| `train.py` | Fine-tunes DeBERTa-v3 as a 2-class classifier in FP32. Takes the ToxicChat training split from `tc_common.py`, appends extra JSONL training files, and supports soft targets, frozen class weights, a fixed step budget and the bag-of-words correction. Writes the model, `threshold.json` and a run-meta JSON. |
| `tc_common.py` | ToxicChat dataset id, field names, the seeded train/val split, and the shared metric and threshold code used by `train.py`. |
| `decontam.py` | Text normalisation, word shingles and exact / Jaccard / embedding duplicate detection. `v6_build.py` uses its normaliser and shingler. |
| `v6_data.py` | Loaders for every source dataset and reference set, with revisions resolved and recorded. `inspect_all()` reports columns and counts without text. |
| `v6_build.py` | Builds the training pools and the dev sets: de-duplication within and across pools, decontamination against every test and dev reference set, the sampling caps, the 3% held-back rows and the soft targets. Also the reverse audit, the `aegis_val` filter, and the extra rows and training files for R3. |
| `v6_bow_crossfit.py` | Cross-fitted unigram TF-IDF + logistic-regression log-odds: the shortcut audit and the per-row `bow_logit` for R2. |
| `v6_committee_targets.py` | Turns the teacher committee's margins into R3 training targets: Platt scaling per member, the committee mean, and the disputed-label and clipping rules. CPU only. |
| `r3_committee/r3_committee_modal.py` | Scores the R3 rows and the calibration sets with three open guard models served by vLLM on Modal, writes per-member margin files, and calls `v6_committee_targets.py`. |
| `v6_score.py` | The recipe variants and their readable names, the recipe constants, the exact `train.py` command for each run, the margin-storing dev scorer, and the lock that refuses test-set scoring until `V6_SELECTION.md` is sealed: its sha256 recorded in `V6_SELECTION.sha256` and still matching the file. The evaluation harness that scored the test sets applied the same check. |
| `v6_metrics.py` | Dev metrics, the three frozen thresholds (strict, pre-screen, balanced), the two-model average and the share of scores pinned near 0 or 1. |
| `metrics_core.py` | The small metrics `v6_metrics.py` builds on: AUPRC, the lowest threshold within a false-positive budget, and F1 at a threshold (used by a test). |
| `v6_select.py` | Selection under the fixed rule from per-item dev scorecards, the release bars and the thresholds; writes `V6_SELECTION.md` and `selection.json`. `--dev-scorecards` re-runs it on the evidence bundle's development scorecards. |
| `v6_modal.py` | The Modal launcher: tests, source inspection, data build, cross-fit, trial run, release 1.0 trial scoring, training rounds with dev scoring, release staging and fetching results. |
| `train_image.py` | The Modal training image with its pinned packages, and `CODE_DIR` (`/root/code`), where this folder is mounted in the container. |
| `training_config.json` | The configuration published with the weights; `v6_modal.py stage` writes it unchanged. |
| `test_v6_*.py`, `r3_committee/test_r3_committee.py` | Unit tests. CPU only, synthetic data, no downloads. |

## How the pieces fit

1. **Data build** (`v6_modal.py --action build`, then `reverse`, `aegisval`, `crossfit`): `build_p1`
   writes the ToxicChat training split (P1) and its validation split; `build_data` runs
   `v6_build.build_all` on the sources loaded by `v6_data.py`. Text stays on the Modal volume; only
   counts, ids and hashes come back. `crossfit` runs `v6_bow_crossfit.py` to write the R2 files.
2. **Teacher labels for R3** (`--action r3build`): extra rows are collected. `r3_committee_modal.py`
   (`plan`, `calib`, `score`, `targets`, `publish`) scores them with the committee and writes the
   targets file to the data volume. `v6_modal.py --action r3train` then builds the R3 training files.
3. **Training on Modal** (`--action smoke`, then `--action wave --arms ... --seeds 13,42,99 --steps S`;
   `--arms` takes recipe-variant keys): each run calls `train.py` with the command from
   `v6_score.build_train_cmd`.
4. **Dev scoring**: each run is scored on the seven dev sets right after training. Scorecards store
   the row id, label, score and logit margin, never text. `--action fetch` copies them to
   `v6_results/`.
5. **Selection**: `python3 v6_select.py` applies the selection rule to the fetched scorecards, checks
   the release bars and computes the frozen thresholds for each training run. Sealing the record
   (`v6_score.seal_selection`) unlocks the test sets. `--action stage` then stages the selected run's
   files for release.

Test-set scoring is not in this folder: `../score_test_sets.py` scores the published weights on the test sets
with the evaluation harness's settings.

## Names in the code

Some short keys are part of file names, run folders, stored results or published files, so they
were kept. This is what they mean.

| Key | Plain name |
|---|---|
| `R1` | Recipe variant: the base recipe. |
| `R0` | The base recipe without the alarming-but-benign data (no FalseReject, and only the harmful WildJailbreak rows). |
| `R2` | The base recipe + bag-of-words correction. |
| `R3` | The base recipe + teacher labels. This is the released recipe; `R3_s13` is the shipped run. |
| `R4` | The base recipe without FalseReject. |
| `<variant>_s<seed>` | One training run of a recipe variant, e.g. `R3_s13`. Stored results call it a "replicate". |
| `E11`, `E12`, `E13`, `E23` | Two-model averages (mean logit margin of two runs): base + base (different seeds), base + bag-of-words correction, base + teacher labels, bag-of-words correction + teacher labels. |
| `A0_s13`, `A0_s13_repeat` | The release 1.0 trial checkpoints scored by `--action baseline`. |
| `P1` / `p1_copy` | The ToxicChat pool: train.py's ToxicChat training split, written out as `p1_copy.jsonl` so the recipe can list it twice more. |
| `P2`, `P3`, `P4`, `P5` | The Aegis 2.0, WildGuardMix, WildJailbreak and FalseReject pools (`p2.jsonl` … `p5.jsonl`). |
| `p4_r0` | The WildJailbreak pool without its benign rows, used by R0. |
| `twins_prompt` | Paired prompts generated in-house (one benign, one unsafe per pair). |
| `dev_pair2` | The PHTest (harmless) and HarmfulQA / CatQA (harmful) dev set. |
| `dev_holdout_p3`, `dev_holdout_p4`, `dev_holdout_p5` | Held-back rows of the WildGuardMix, WildJailbreak and FalseReject pools. |
| `tc_val`, `aegis_val` | The ToxicChat and Aegis 2.0 validation splits. |
| `dev_intent` | Intent score: the mean AUROC on `twins_prompt` and `dev_pair2`. |
| `saturation` | Share of scores pinned near 0 or 1 (logit margin beyond ±9). |
| `B` | The release 1.0 trial intent score (`RELEASE_1_TRIAL_INTENT_SCORE` in `v6_select.py`). |
| `G1`, `G2`, `G3` | The release bars: intent score bar, ToxicChat AUPRC bar, share of pinned scores. |
| `T_bal` / `t_bal` | Balanced threshold: maximises mean F1 over `tc_val`, `aegis_val` and `dev_holdout_p3`. The shipped decision threshold. |
| `T_gate` / `t_gate` | Strict threshold: the lowest threshold with at most 1% false positives on the dev benign prompts. |
| `T_screen` / `t_screen` | Pre-screen threshold: escalate when the score reaches it (at least 98% dev recall). |
| `decision_gate` | In the evaluation scorecards: the decision at the strict threshold. |
| `margin_v12`, `score_v12` | In the evaluation scorecards: Qwen3Guard's alternative score definition. |
| `clash`, `clash_rows` | In the committee targets: a disputed label (the committee confidently disagrees with the source label), whose target becomes 0.5. |
| `val_carve_sha256` | The hash of the ToxicChat validation split, in each run's `run_meta.json`. |

## What you need

- A Modal account, with GPU access (A100-80GB for training and scoring, A10G for the data build,
  L40S for the committee).
- A Modal secret named `huggingface-secret` that holds a Hugging Face token. The account behind the
  token must have accepted the terms of the gated datasets the loaders read: `allenai/wildguardmix`
  and `allenai/wildjailbreak` (`lmsys/toxic-chat` is not gated); check the others listed in
  `v6_data.py`.
- Dataset revisions: `v6_data.py` loads each dataset at its current revision. Release 2.0 used these;
  set them when reproducing, or the data may differ:
  - `lmsys/toxic-chat` @ `29df8e4dba60e1f4af4b4075c0705c5b313548a8` (`toxicchat0124`)
  - `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` @ `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`
  - `allenai/wildguardmix` @ `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`
  - `allenai/wildjailbreak` @ `5ddc12a7894f842b0619b8e1c7ee496b198af009`
  - `AmazonScience/FalseReject` (card @ `493ba967714ea54c6f01067e1f61e389cc2c9b3e`)
  - `hbseong/HarmAug_generated_dataset` @ `8898635dce310084cd72036214fddd885a9adc08`
  - `allenai/WildChat-1M` @ `7d6490e462285cf85d91eabea0f9a954fbddcd1f`
  - base model `microsoft/deberta-v3-large` @ `64a8c8eab3e352a784c658aef62be1662607476f`
- The package versions in the image definitions:
  - training image (`train_image.py`): Python 3.11, `torch==2.12.1`, `transformers==5.12.1`,
    `accelerate>=1.1.0`, `datasets>=2.18`, `scikit-learn>=1.3`, `scipy>=1.10`,
    `sentencepiece>=0.1.99`, `protobuf>=3.20`, `huggingface_hub>=0.23`, `numpy>=1.24`,
    `pandas>=2.0`, `pytest>=8`. Only torch and transformers are exact pins.
  - data-build image (`v6_modal.py`, `prep_image`): Python 3.11 with lower bounds only, plus
    `sentence-transformers>=3.0` and `requests>=2.31`.
  - committee image (`r3_committee_modal.py`): CUDA 12.8 base image, Python 3.12, `vllm==0.33.0`,
    plus `hf_transfer`, `scikit-learn` and `httpx` unpinned.
- Inputs that this code reads from the data volume but does not create: the XSTest and OR-Bench set
  files under `/vol/sets/`, the twin-prompt dev files under `/vol/data/`, the FalseReject pool
  under `/vol/v6/src/`, and, for `--action baseline`, the release 1.0 trial checkpoints under
  `/vol/runs/`.

### Names you can override

| Variable | Used in | Default |
|---|---|---|
| `V6_VOLUME` | `v6_modal.py`, `r3_committee_modal.py` | `v6-data` |
| `V6_APP` | `v6_modal.py` | `v6-train` |
| `R3_COMMITTEE_APP` | `r3_committee_modal.py` | `r3-committee` |
| `R3_COMMITTEE_VOLUME` | `r3_committee_modal.py` | `r3-committee` |
| `R3_COMMITTEE_HF_VOLUME` | `r3_committee_modal.py` | `r3-committee-hf` |

Set them in the shell that runs `modal run`. Modal mounts volumes by the ids resolved at launch. Two
report labels in `r3_committee_modal.py` (`targets_path` and the `publish` paths) are computed inside
the container, so they show the default unless the variable is set there too.

## Running the tests

```bash
cd training
python3 -m pytest -q
```

They need `numpy`, `pandas`, `scikit-learn`, `torch`, `transformers`, `datasets`, `tokenizers`,
`modal` (imported by `r3_committee_modal.py`; nothing is sent to Modal) and `pytest`.
`test_v6_bow.py::test_main_end_to_end_with_the_v6_flags` runs the Hugging Face `Trainer` and also
needs `accelerate`. `v6_modal.py --action tests` runs `test_v6_bow.py`, `test_v6_committee.py`,
`test_v6_metrics.py`, `test_v6_score.py` and `test_v6_build.py` on the training image.

## Not included

- `V6_SELECTION.md` and its seal, the selection record `v6_select.py` wrote. Test-set scoring stayed
  locked until it was sealed, so the recipe and thresholds were chosen on dev sets before any test
  result was seen. `v6_select.py --dev-scorecards` recomputes its choice and thresholds from the
  published development scorecards.
- The data itself. Text never leaves the Modal volume.
