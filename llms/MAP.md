# partyhat-safety-encoder-eval: file map

Generated from commit `4067235901f4` (2026-11-20) of github.com/partyhat-ai/partyhat-safety-encoder-eval by `scripts/llms_digest.py`, which `.github/workflows/llms-digest.yml` reruns on every push to `main`, so this text matches the code at that commit.

35 tracked files. 34 are reproduced in full in the `code-N.md` parts; the rest are listed with the reason they are not.

Every path below is relative to https://raw.githubusercontent.com/partyhat-ai/partyhat-safety-encoder-eval/main/ (append the path to fetch the raw file).

| File | Bytes | Lines | Description | In digest |
|---|---|---|---|---|
| `.github/workflows/llms-digest.yml` | 1,287 | 40 | Keeps llms.txt and llms/ (a plain-text copy of this repository for tools that | code-1.md |
| `.gitignore` | 61 | 5 |  | code-1.md |
| `LICENSE` | 1,065 | 21 |  | code-1.md |
| `README.md` | 6,832 | 104 | partyhat-safety-encoder 2.0: evaluation scripts and training code | code-1.md |
| `attach_labels.py` | 7,091 | 165 | attach_labels.py: rebuild the item labels stripped from the evidence bundle, from the public dataset | code-1.md |
| `baselines/README.md` | 5,036 | 69 | How each comparison model was scored | code-1.md |
| `dev_ablation.py` | 33,567 | 540 | dev_ablation.py: recompute the "Where the gain came from" section of the partyhat-safety-encoder 2.0 | code-1.md |
| `reproduce_card.py` | 68,621 | 1,110 | reproduce_card.py: recompute every test-set figure on the partyhat-safety-encoder 2.0 model card, an | code-2.md |
| `requirements-scoring.txt` | 158 | 9 |  | code-2.md |
| `requirements.txt` | 87 | 5 |  | code-2.md |
| `score_test_sets.py` | 27,230 | 473 | score_test_sets.py: score the published weights (or an open comparison guard) on the card's twelve t | code-2.md |
| `scripts/llms_digest.py` | 12,022 | 238 | Build llms.txt and llms/: a plain-text copy of this repository for tools that | code-3.md |
| `training/README.md` | 13,977 | 175 | Training code for partyhat-safety-encoder 2.0 | code-3.md |
| `training/decontam.py` | 24,848 | 584 | decontam.py — text normalisation and duplicate / near-duplicate detection against test sets. | code-3.md |
| `training/metrics_core.py` | 1,168 | 36 | metrics_core.py: small score-based metrics that v6_metrics.py builds on. | code-3.md |
| `training/r3_committee/r3_committee_modal.py` | 37,608 | 775 | r3_committee_modal.py: the teacher committee that labels rows for recipe variant R3 | code-3.md |
| `training/r3_committee/test_r3_committee.py` | 4,763 | 91 | Unit tests for the pure functions of r3_committee_modal.py. Synthetic server output only. | code-3.md |
| `training/tc_common.py` | 8,313 | 215 | tc_common.py: the shared ToxicChat data and metric code. | code-3.md |
| `training/test_v6_bow.py` | 12,076 | 262 | Tests for the bag-of-words correction (recipe variant R2) in train.py and its cross-fit script. | code-4.md |
| `training/test_v6_build.py` | 3,040 | 62 | Tests for v6_build.py: de-duplication, decontamination matching, caps and held-back rows. CPU only;  | code-4.md |
| `training/test_v6_committee.py` | 4,881 | 93 | Tests for v6_committee_targets.py (the teacher-committee targets for R3). CPU only:  python3 -m pyte | code-4.md |
| `training/test_v6_metrics.py` | 3,947 | 97 | Unit tests for v6_metrics.py: dev metrics, the frozen thresholds and the two-model average. CPU, num | code-4.md |
| `training/test_v6_score.py` | 6,295 | 153 | Tests for v6_score.py: the train.py command for each recipe variant, and margin-storing scorecards. | code-4.md |
| `training/test_v6_select.py` | 1,997 | 47 | Tests for v6_select.select (the selection rule and the ToxicChat check) on synthetic candidate summa | code-4.md |
| `training/train.py` | 26,353 | 527 | train.py: fine-tune a DeBERTa-v3 encoder as a 2-class prompt-safety classifier. | code-4.md |
| `training/train_image.py` | 1,367 | 39 | train_image.py: the Modal image used for training, dev scoring and the tests. | code-4.md |
| `training/v6_bow_crossfit.py` | 5,564 | 123 | v6_bow_crossfit.py: cross-fitted bag-of-words log-odds for the training files. | code-4.md |
| `training/v6_build.py` | 33,721 | 664 | v6_build.py: builds the training pools and the dev sets. Runs on Modal only. | code-4.md |
| `training/v6_committee_targets.py` | 7,007 | 147 | v6_committee_targets.py: turns the teacher committee's verdicts into training labels for recipe | code-5.md |
| `training/v6_data.py` | 7,017 | 183 | v6_data.py: source loaders for the v6 training pools, dev sets and decontamination references. | code-5.md |
| `training/v6_metrics.py` | 6,434 | 133 | v6_metrics.py: dev-set metrics, the three frozen thresholds, and the two-model average. | code-5.md |
| `training/v6_modal.py` | 26,602 | 586 | v6_modal.py: the Modal launcher for building, training and dev-scoring the release-2.0 encoder. | code-5.md |
| `training/v6_score.py` | 8,781 | 195 | v6_score.py: the recipe variants, the train.py command for each run, and the dev scorer. | code-5.md |
| `training/v6_select.py` | 14,287 | 253 | v6_select.py: choose the release candidate from per-item dev scorecards (no text). | code-5.md |

### Data files (not reproduced; one sample per folder in `data-samples.md`)

| Folder | Files | Bytes |
|---|---|---|
| `training/` | 1 | 3,252 |
