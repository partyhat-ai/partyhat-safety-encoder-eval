# Evidence for partyhat-safety-encoder 2.0

This folder holds what you need to check every figure on the partyhat-safety-encoder 2.0 model card: the per-item
test scorecards of release 2.0 and of each model the card compares it with, the per-item development scorecards that set the
thresholds and back the card's "Where the gain came from" section, the index files the figures use, and four scripts. `attach_labels.py`
restores the item labels that are not redistributed here, `reproduce_card.py` recomputes each test-set figure and the
thresholds and compares them with the card, `dev_ablation.py` does the same for the development-set section and its full report
(`reports/dev_ablation.md`), and `plot_frr_vs_recall.py` draws the card's figure. Names used in file names and
scorecard fields are explained under "Names in the files".

Everything here is free of dataset text. A scorecard row holds a row index, the gold label (unless stripped), the
model's score and margin, and its decision; no prompt or response text. The texts are in the public datasets listed
under "Datasets" at the pinned revisions; row `i` is the row order there (after the stated exclusions).

The GPT-4o scorecards are as scored in June 2026: each row holds the index, GPT-4o's score, its decision and its
request latency. The score is P(block), GPT-4o's probability of answering BLOCK to the guard prompt (from the answer
token's log-probabilities), and the decision is that score ≥ 0.5 (`reproduce_card.py` checks this on every row), so
GPT-4o has a curve on the card's figure as well as its own decision point. The rows carry no label: each item's
label is its set's (every OR-Bench-Hard and XSTest-safe item is benign, every OR-Bench-Toxic and XSTest-unsafe item
harmful). Their `meta` holds the project's guard prompt, the instruction GPT-4o
was given. Their rows are in the same order as release 2.0's scorecards for those sets; `reproduce_card.py` checks this.

**Labels stripped** for the ToxicChat, Aegis 2.0 and WildGuardTest test sets and for five development sets (the
ToxicChat validation split, the Aegis 2.0 validation set, and the WildGuardMix, WildJailbreak and FalseReject
held-back rows), so that no third-party labels are redistributed. Their scorecards keep index, score, margin and
decision; `attach_labels.py` rebuilds the labels from
the public datasets (see "Rebuilding the labels"). Every other scorecard carries its item label, apart from the GPT-4o
scorecards described above.

## Files

- `scorecards/test/{mode}__{model}__{set}.json`: 75 per-item scorecards. Only the scorecards the card's figures and their
  checks need are included:

  | Model | Scorecard prefix | Sets |
  |---|---|---|
  | release 2.0, three training runs (seed 13 is the published weights) | `classifier__v6_R3_s13`, `classifier__v6_R3_s42`, `classifier__v6_R3_s99` | all 12 |
  | release 1.0 | `classifier__partyhat-hf_partyhat-safety-encoder` | 10 (all but `aegis2_test_prompt`, `wildguardtest_prompt`) |
  | omni-moderation (snapshot `omni-moderation-2024-09-26`) | `classifier__openai_omni-moderation-2024-09-26` | 4 (OR-Bench and XSTest) |
  | GPT-4o (`gpt-4o-2024-11-20`, the project's guard prompt; score P(block), decision P(block) ≥ 0.5) | `guard__gpt-4o-2024-11-20` | 4 (OR-Bench and XSTest) |
  | Llama Guard 3 8B | `guard__meta-llama_Llama-Guard-3-8B` | 7 (the held-out ranking sets) |
  | Qwen3Guard-Gen-0.6B, loose and strict | `guard__Qwen_Qwen3Guard-Gen-0.6B-loose`, `guard__Qwen_Qwen3Guard-Gen-0.6B-strict` | 7 each (the held-out ranking sets) |

  The twelve sets are `toxicchat_test`, `openai_mod`, `aegis2_test_prompt`, `wildguardtest_prompt`,
  `harmbench_std`, `sst`, `orbench_hard`, `orbench_toxic`, `xstest_safe`, `xstest_unsafe`, `jbb_benign` and
  `jbb_harmful`; the held-out ranking sets are the OR-Bench, XSTest and JailbreakBench pairs and `openai_mod`.
  Qwen3Guard cards carry two score definitions (`margin` and `margin_v12`); the card uses whichever is better for
  each figure. omni-moderation cards have no `margin`; their `score` is the highest of the 13 category scores, which
  each row also carries. omni-moderation was scored on all twelve sets; only the sets the card reports are included.
- `scorecards/dev/{run folder}/{set}.json`: 119 per-item development scorecards, the seven development sets for each
  of 17 training runs. Each row holds the row index `i`, the source row id `src`, the label (unless stripped), the
  score, the margin and the decision at 0.5. Only the runs the card's "Where the gain came from" section uses are
  included:

  | Recipe variant | Run folders |
  |---|---|
  | base recipe | `R1_s13`, `R1_s42`, `R1_s99` |
  | base recipe without the alarming-but-benign data (no FalseReject, no WildJailbreak benign rows) | `R0_s13`, `R0_s42`, `R0_s99` |
  | base recipe without FalseReject | `R4_s13`, `R4_s42`, `R4_s99` |
  | base recipe + bag-of-words correction | `R2_s13`, `R2_s42`, `R2_s99` |
  | base recipe + teacher labels: the released recipe (`R3_s13` is the published weights) | `R3_s13`, `R3_s42`, `R3_s99` |
  | release 1.0 trial checkpoints (release 1.0's recipe retrained twice, same seed) | `A0_s13`, `A0_s13_repeat` |

  The two-model averages in the report (two training runs' logit margins averaged) are computed from these runs, so
  they have no scorecards of their own. The trial checkpoints were scored on an earlier, 1,199-row Aegis 2.0
  validation set; `dev_ablation.py` uses the 1,159 rows the other runs were scored on.
- `reports/label_hashes.json`: the sha256 of every stripped label vector (95 scorecards: 10 test, 85 development).
- `reports/toxicchat_dedup_kept_i.json`: the ToxicChat test rows kept after removing near-duplicates of the training
  split (4,610 of 5,083).
- `reports/subset_indices.json`: row indices for subsets defined by dataset metadata or by de-duplication, with no
  harm labels: the de-duplicated ToxicChat rows (as above), ToxicChat's human-annotated rows, WildGuardTest's
  adversarial rows, `pooled_held_out_kept_rows`, the rows of the pooled held-out set (each prompt counted once), and
  `pooled_held_out_set_order`, the order of its sets.
- `reports/d1_exclude.json`: the held-out test rows that match prompts in ToxicChat's training split, for the
  sensitivity figure in the card's contamination paragraph.
- `attach_labels.py`: rebuilds the stripped labels from the public datasets and verifies each against
  `reports/label_hashes.json`.
- `reproduce_card.py`: recomputes every figure on the card that comes from the test scorecards, and the balanced,
  pre-screen and very-low-false-refusal thresholds bit for bit from the development scorecards (numpy and
  scikit-learn only).
- `dev_ablation.py`: recomputes every figure in the card's "Where the gain came from" section and regenerates
  `reports/dev_ablation.md` from the development scorecards (numpy and scikit-learn only).
- `reports/dev_ablation.md`: the full development-set analysis: what each recipe variant trained on, results by
  variant and by training run, every comparison with its 95% interval, and the caveats. It is descriptive and made
  after the fact, not planned before the runs.
- `plot_frr_vs_recall.py`: draws the card's figure, `frr_vs_recall.png`, from the OR-Bench scorecards: curves for
  release 2.0's published weights, omni-moderation and GPT-4o with 95% bands, release 2.0's three thresholds and three
  comparison points (numpy and matplotlib; the OR-Bench labels are in the scorecards, so it runs on this folder as
  shipped). Its bands resample the harmless prompts with `reproduce_card.py`'s OR-Bench draws, so it redraws the
  figure exactly.
- `latency/release_2_A10_fp32.json`: release 2.0's speed measurements quoted on the card, taken on the published
  weights on one NVIDIA A10 GPU (FP32; batch-1 latency at 128 and 512 tokens, plus static-batch throughput).
- `latency/qwen3guard_gen_0.6b_A10G_bf16.json`: Qwen3Guard-Gen-0.6B timed by the same harness on Modal's same GPU
  class (the container reported an NVIDIA A10G), in bf16: time to its verdict and to its full output, at 128 and 512
  tokens of prompt.
- `latency/time_encoder.py` and `latency/time_guard.py`: the timing code that produced them, as it ran. Both use 100
  warm-up and 1,000 timed requests at batch size 1 and exclude tokenisation. The Modal launcher and the harness that
  loads each model are not included, so they document the method rather than running on their own.
  `reproduce_card.py` checks the card's speed figures against both files.
- `MANIFEST.md`: the sha256 and size of every file.

## Rebuilding the labels

    pip install numpy pandas scikit-learn huggingface_hub pyarrow
    python3 attach_labels.py . ../evidence_with_labels      # your own Hub access; accept the access terms of
                                                             # allenai/wildguardmix and allenai/wildjailbreak first

`attach_labels.py` copies this folder, rebuilds each stripped label vector from the public dataset at the pinned
revision (by the row index `src` in each card) and refuses any vector whose sha256 differs from
`reports/label_hashes.json`. It should report 95 files rebuilt and verified, 0 mismatches.

## Recomputing the card

    python3 reproduce_card.py ../evidence_with_labels

For each figure it prints the card's wording, the card's value and the recomputed value; then informational notes,
the figures it does not recompute (with their source), and a summary. It exits non-zero if any figure differs
beyond the card's rounding. It runs in under a minute and needs about 2 GB of memory; `--quick` uses 1,000
bootstrap draws (intervals will then not match).

    python3 dev_ablation.py ../evidence_with_labels

`dev_ablation.py` prints, for each figure in the card's "Where the gain came from" section, the card's wording, the
card's value and the recomputed value; then it regenerates `reports/dev_ablation.md` and compares it line by line. It
exits non-zero on any difference. It runs in about a minute.

    python3 plot_frr_vs_recall.py . --out frr_vs_recall.png

The definitions (held-out AUROC, pooled held-out rates, recall at 1% false positives and at a 10% false-refusal
budget, AUPRC, same catch rate, GPT-4o reviewing release 2.0's flags, the bootstrap) are in `reproduce_card.py`'s opening comment, and the development-set
definitions in `dev_ablation.py`'s. In brief: intervals are 95% paired
bootstrap intervals, 10,000 draws; benign and unsafe items are resampled separately within each set with the same
draws for every model, and the three release 2.0 training runs are resampled inside each draw. Each set has its own
random stream, numpy `default_rng([13, zlib.crc32(set name)])`, so a figure's interval depends only on the sets that
figure uses. Intervals therefore reproduce even if a gated set's labels cannot be rebuilt, except for figures that use
that set. No figure on the card uses WildGuardTest's labels. `dev_ablation.py` builds its intervals the same way,
and also resamples each recipe variant's training runs inside each draw. Rates at a fixed threshold (the card's
threshold and comparison tables) reuse the draws of the set they come from. ToxicChat-T5-large's figures come from
its own model card and are listed as not recomputed.

## Datasets

| Set | Source @ revision |
|---|---|
| `orbench_hard` / `orbench_toxic` | `bench-llms/or-bench` @ `fd6ee135ee63ff6c4f3ff72c0e39627bf0a7f314` |
| `xstest_safe` / `xstest_unsafe` | `natolambert/xstest-v2-copy` @ `b71afe2a6d10e5a6254ea8bcb006c48b095a15d5` |
| `toxicchat_test` | `lmsys/toxic-chat` `data/0124/toxic-chat_annotation_test.csv` @ `29df8e4dba60e1f4af4b4075c0705c5b313548a8` |
| `aegis2_test_prompt` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` `test.json` @ `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c` (the 36 `REDACTED` prompts excluded) |
| `wildguardtest_prompt` | `allenai/wildguardmix` `test/wildguard_test.parquet` @ `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` |
| `openai_mod` | `openai/moderation-api-release` (GitHub) `data/samples-1680.jsonl.gz` @ `f4ab51b5edd3bfbcb349a56324274235b674e0e4` |
| `jbb_benign` / `jbb_harmful` | `JailbreakBench/JBB-Behaviors` @ `886acc352a31533ffbcf4ef22c744658688086fc` |
| `harmbench_std` | `centerforaisafety/HarmBench` (GitHub) `data/behavior_datasets/harmbench_behaviors_text_all.csv` @ `8e1604d1171fe8a48d8febecd22f600e462bdcdd` |
| `sst` | `Bertievidgen/SimpleSafetyTests` `sst_test_cases.csv` @ `98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5` |
| ToxicChat validation split (`tc_val`) | `lmsys/toxic-chat` `data/0124/toxic-chat_annotation_train.csv` @ the revision above, 10% stratified split (`random_state` 13) |
| Aegis 2.0 validation set (`aegis_val`) | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` `validation.json` @ the revision above (rows joined by `src`, the dataset's `id`) |
| WildGuardMix held-back rows (`dev_holdout_p3`) | `allenai/wildguardmix` `train/wildguard_train.parquet` @ `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` |
| WildJailbreak held-back rows (`dev_holdout_p4`) | `allenai/wildjailbreak` `train/train.tsv` @ `5ddc12a7894f842b0619b8e1c7ee496b198af009` |
| FalseReject held-back rows (`dev_holdout_p5`) | `AmazonScience/FalseReject` train split (card @ `493ba967714ea54c6f01067e1f61e389cc2c9b3e`); every train prompt is benign by the dataset's definition |
| PHTest, HarmfulQA and CatQA set (`dev_pair2`) | benign: `furonghuang-lab/PHTest` @ `edb80210812ecc7dab2219aa2e70921710c01888`, `Request` where `Harmfulness == "harmless"`; unsafe: `declare-lab/HarmfulQA` @ `6f1a78aed47d16c0695e4595d0159abc38197bfd`, `question`, and `declare-lab/CategoricalHarmfulQA` @ `321e068b391b7d1aeeb882ac468356a9513c9167`, split `en`, `Question` |
| paired prompts (`twins_prompt`) | generated in-house for this project; no third-party source |

## Licences and attribution

Licences are as stated on each dataset's card at the pinned revision (for the two GitHub sources, the
repository's LICENSE file at the pinned commit). The development sets in the table (the ToxicChat validation split,
the Aegis 2.0 validation set, the WildGuardMix, WildJailbreak and FalseReject held-back rows, the PHTest, HarmfulQA and CatQA
set and the paired prompts) were used to build and select the model; their scorecards are in `scorecards/dev/` and
are used only by `dev_ablation.py`.

| Dataset | Sets in this bundle | Licence | Citation |
|---|---|---|---|
| XSTest v2 | `xstest_safe`, `xstest_unsafe` | CC-BY-4.0 (authors' release `Paul/XSTest` @ `f600c994b256f12867dfa5b3eb3d545a3e62f8b5`, and the mirror scored, `natolambert/xstest-v2-copy` @ `b71afe2a6d10e5a6254ea8bcb006c48b095a15d5`, verified identical) | Röttger, Kirk, Vidgen, Attanasio, Bianchi and Hovy, *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*, NAACL 2024 |
| OR-Bench | `orbench_hard`, `orbench_toxic` | CC-BY-4.0 | Cui, Chiang, Stoica and Hsieh, *OR-Bench: An Over-Refusal Benchmark for Large Language Models*, arXiv:2405.20947 (2024) |
| JailbreakBench (JBB-Behaviors) | `jbb_benign`, `jbb_harmful` | MIT | Chao et al., *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models*, 2024 |
| HarmBench | `harmbench_std` | MIT | Mazeika et al., *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal*, 2024 |
| SimpleSafetyTests | `sst` | CC-BY-2.0 | Vidgen et al., *SimpleSafetyTests: a Test Suite for Identifying Critical Safety Risks in Large Language Models*, arXiv:2311.08370 (2023) |
| OpenAI moderation evaluation set | `openai_mod` | MIT | Markov et al., *A Holistic Approach to Undesired Content Detection*, arXiv:2208.03274 (2022) |
| PHTest | PHTest, HarmfulQA and CatQA set (`dev_pair2`), benign side | MIT | An et al., *Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models*, COLM 2024, arXiv:2409.00598 |
| HarmfulQA | PHTest, HarmfulQA and CatQA set (`dev_pair2`), unsafe side | Apache-2.0 | Bhardwaj and Poria, *Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment*, 2023 |
| CategoricalHarmfulQA | PHTest, HarmfulQA and CatQA set (`dev_pair2`), unsafe side | Apache-2.0 | Bhardwaj, Do and Poria, *Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task Arithmetic*, 2024 |
| FalseReject | FalseReject held-back rows (`dev_holdout_p5`) | CC-BY-NC-4.0 | Zhang, Xu, Wu and Reddy, *FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning*, 2025 |
| ToxicChat | `toxicchat_test`, ToxicChat validation split (`tc_val`) | CC-BY-NC-4.0 | Lin et al., *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation*, 2023 |
| Aegis 2.0 | `aegis2_test_prompt`, Aegis 2.0 validation set (`aegis_val`) | CC-BY-4.0 | Ghosh et al., *AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails*, 2025 |
| WildGuardMix | `wildguardtest_prompt`, WildGuardMix held-back rows (`dev_holdout_p3`) | ODC-BY; gated: access on accepting the AI2 Responsible Use Guidelines | Han et al., *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs*, 2024 |
| WildJailbreak | WildJailbreak held-back rows (`dev_holdout_p4`) | ODC-BY; gated: access on accepting the AI2 Responsible Use Guidelines | Jiang et al., *WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models*, 2024 |
| WildChat-1M | training only (40,000 first user turns) | ODC-BY | Zhao, Ren, Hessel, Cardie, Choi and Deng, *WildChat: 1M ChatGPT Interaction Logs in the Wild*, ICLR 2024 |
| HarmAug generated dataset (`hbseong/HarmAug_generated_dataset`) | training only (instructions) | Apache-2.0 | Lee et al., *HarmAug*, ICLR 2025 |
| paired prompts (this project) | paired prompts (`twins_prompt`) | generated for this project; no third-party source | — |

For ToxicChat, Aegis 2.0, WildGuardMix, WildJailbreak and FalseReject, the item labels are not redistributed in this
bundle: `attach_labels.py` rebuilds them from the public datasets, under the reader's own access and acceptance of each
dataset's terms. No dataset text is redistributed for any set.

The bundle as a whole is for non-commercial research and evaluation, matching the model's CC-BY-NC-4.0 licence.

**Release 1.0.** Its scorecards (`classifier__partyhat-hf_partyhat-safety-encoder__*`) record the private source
revision they were scored at (`10ba6287…`). Release 1.0 is retired and its weights are not distributed. Its weights
file is identified by `model.safetensors` sha256
`28d9fd853a6359d2dfc3fe51c86665cd3dde052f5098e1d518193afc13a74935`.

## Names in the files

File names, run folders and scorecard fields keep the names the training and scoring code gave them. Their plain
meanings:

| Name in the files | Meaning |
|---|---|
| `v6` (in `classifier__v6_R3_s13` and similar) | the internal name for release 2.0 |
| `R3_s13`, `R3_s42`, `R3_s99` | release 2.0's three training runs (seeds 13, 42, 99) of the released recipe, base recipe + teacher labels; `R3_s13` is the published weights |
| `R1`, `R0`, `R4`, `R2` (run folders `<name>_s<seed>`) | the base recipe; base recipe without the alarming-but-benign data; base recipe without FalseReject; base recipe + bag-of-words correction |
| `A0_s13`, `A0_s13_repeat` | the release 1.0 trial checkpoints: release 1.0's recipe retrained twice with the same seed, scored on the development sets |
| `classifier__partyhat-hf_partyhat-safety-encoder` | release 1.0 |
| `tc_val`, `aegis_val` | the ToxicChat validation split; the Aegis 2.0 validation set |
| `dev_holdout_p3`, `dev_holdout_p4`, `dev_holdout_p5` | the WildGuardMix, WildJailbreak and FalseReject held-back rows (`p3`, `p4`, `p5` are the training code's numbers for those three training pools) |
| `twins_prompt`, `dev_pair2` | the paired prompts (generated in-house); the PHTest, HarmfulQA and CatQA set |
| `toxicchat_test`, `aegis2_test_prompt`, `wildguardtest_prompt`, `openai_mod`, `harmbench_std`, `sst`, `orbench_*`, `xstest_*`, `jbb_*` | the test sets listed under "Datasets" (`sst` is SimpleSafetyTests, `jbb` JailbreakBench, `openai_mod` the OpenAI moderation evaluation set) |
| `i`, `src` | the row's position in the scorecard; the row's id in the public dataset (an index, or the dataset's own id) |
| `score`, `margin`, `decision` | P(unsafe) (other models: their score, see above); the logit of unsafe minus the logit of safe; the model's decision at its default threshold (development scorecards: at 0.5) |
| `decision_gate` | release 2.0's decision at the strict threshold (the very-low-false-refusal rule, 0.957 for the published weights) |
| `meta.thresholds.T_bal`, `meta.thresholds.T_gate` | the balanced threshold and the strict threshold of that training run |
| `margin_v12`, `score_v12`, `tie_v12` | Qwen3Guard's alternative score definition (and its tie flag) |
| `cat`, `cat_flagged` | omni-moderation's 13 category scores and category flags |
| `jbb_source` | JailbreakBench's `Source` field, where the behaviour came from (empty for other sets) |
| `toxicchat_dedup_kept_i`, `toxicchat_human_i`, `wildguardtest_adversarial_i` | row indices of de-duplicated ToxicChat, ToxicChat's human-annotated rows and WildGuardTest's adversarial rows |
| `pooled_held_out_kept_rows`, `pooled_held_out_set_order` | the rows of the pooled held-out set (each prompt counted once) and the order of its sets |
| `d1_exclude.json` | held-out test rows that match prompts in ToxicChat's training split |
