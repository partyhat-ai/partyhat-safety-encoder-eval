# partyhat-safety-encoder 2.0: evaluation scripts and training code

By Cameron Feather at Partyhat. These scripts recompute the figures on the
[partyhat-safety-encoder 2.0 model card](https://huggingface.co/partyhat-hf/partyhat-safety-encoder) from the
evidence bundle in that repository's `evidence/` folder. The bundle holds a per-item scorecard for release 2.0 and
for each model the card compares it with, and the development-set scorecards that set release 2.0's thresholds and
back the card's "Where the gain came from" section: row index, label, score, margin and decision, with no prompt text.

## The scripts

- `attach_labels.py` restores the item labels the bundle leaves out. The bundle ships without labels for the ToxicChat,
  Aegis 2.0 and WildGuardTest test sets and for five development sets (the ToxicChat validation split, the Aegis 2.0
  validation set, and the WildGuardMix, WildJailbreak and FalseReject held-back rows), so that no third-party labels
  are redistributed. The script copies the bundle, rebuilds those labels from the public datasets at pinned revisions
  using your own Hugging Face access, and checks every rebuilt label vector against a sha256 in the bundle. It refuses
  to write one that differs.
- `reproduce_card.py` recomputes each figure on the card from the labelled copy: point estimates and the 95% paired
  bootstrap intervals (10,000 draws; each test set has its own numpy random stream, seeded from 13 and the set's
  name, so the intervals match the card to the printed digit, and a figure's interval depends only on the sets it
  uses: one set's labels cannot shift another figure). It also recomputes the balanced, pre-screen and
  very-low-false-refusal thresholds bit for bit from the development scorecards. For each figure it prints the card's
  wording, the card's value and the recomputed value, and it exits non-zero if any figure differs beyond the card's
  rounding. Figures that come from neither the scorecards nor the bundle's speed measurements, such as
  ToxicChat-T5-large's published results, are listed with their source. It uses numpy and scikit-learn only. The metric definitions are in its opening comment.
- `dev_ablation.py` recomputes the card's "Where the gain came from" section, a descriptive after-the-fact analysis of
  the development scorecards (no test set), and regenerates its full report, `reports/dev_ablation.md`: each recipe
  variant's results and every comparison with a 95% paired-bootstrap interval. It prints the card's wording, the card's
  value and the recomputed value, and exits non-zero on any difference. numpy and scikit-learn only.

All three scripts are byte-identical to the copies in the evidence bundle; their hashes are in the bundle's `MANIFEST.md`.

## Run it

The model repository is gated for non-commercial use: accept its access terms on its Hugging Face page and log in
(`hf auth login`) before downloading.

```bash
pip install -r requirements.txt
hf download partyhat-hf/partyhat-safety-encoder --include "evidence/*" --local-dir .
python3 attach_labels.py evidence evidence_with_labels     # accept the access terms of allenai/wildguardmix
                                                            # and allenai/wildjailbreak first
python3 reproduce_card.py evidence_with_labels
python3 dev_ablation.py evidence_with_labels
```

`attach_labels.py` should report 95 files rebuilt and verified (10 test, 85 development). `reproduce_card.py` and
`dev_ablation.py` each run in about a minute; `reproduce_card.py` needs about 2 GB of memory. `--quick` uses 1,000
bootstrap draws for a fast check of the point estimates (the intervals will then not match). The versions the scripts
were tested with are pinned in `requirements.txt`.

Dataset sources, revisions and licences are listed in the bundle's `README.md`.

## Scoring the model yourself

`score_test_sets.py` goes the other way, from the weights to the scorecards: it scores the published weights on the
card's twelve test sets, read from the public datasets at pinned revisions in the bundle's row order, and writes
scorecards in the bundle's schema. With `--compare` it checks them against the bundle's scorecards row by row.

```bash
pip install -r requirements-scoring.txt
python3 score_test_sets.py --cap 32 --device cpu --out scorecards_quick --compare evidence_with_labels   # minutes on CPU
python3 score_test_sets.py --out scorecards_full --compare evidence_with_labels                         # all sets; use a GPU
```

On a CPU check of 32 prompts per set, every row, label and decision matched the bundle and the margins matched within
1.4e-5. The same script re-runs the two open comparison guards (`--model llamaguard3`, `qwen3guard06_strict`,
`qwen3guard06_loose`; Llama Guard 3 is gated by Meta, and both want a GPU at full size).

## Re-running the selection

The recipe and thresholds were chosen by a fixed rule, `training/v6_select.py`, on development data only. To apply
that rule to the bundle's development scorecards (the labelled copy from `attach_labels.py`):

```bash
python3 training/v6_select.py --dev-scorecards evidence_with_labels/scorecards/dev --results rerun --out rerun/V6_SELECTION.md
```

It picks the teacher-labels variant, ships its first training run (`R3_s13`, the published weights), and recomputes
every run's three thresholds bit for bit; `rerun/V6_SELECTION.md` shows each step of the choice.

## How the comparison models were scored

`baselines/README.md` records each comparison model's exact configuration: weights and revisions, input format,
decoding, how the verdict is located and turned into a score and a decision, and, for the two OpenAI models, the
request, the guard prompt and the parsing. The OpenAI models are API services, so they are documented rather than
re-runnable here; their per-item results are in the evidence bundle.

## What is in `training/`

The code that built release 2.0: assembling and de-duplicating the training data, fine-tuning the DeBERTa-v3-large
candidates, scoring them on the development sets, and choosing the published checkpoint and its thresholds. It is
published from the project's private repository with local paths and references to internal documents removed, and
with internal terms renamed to the ones the model card uses; `training/README.md` describes each file and lists
exactly what changed.

## What is not here

The development-set scoring that ran inside the training pipeline (`training/v6_score.py` holds its settings), the
sealed selection record `V6_SELECTION.md` (the command above recomputes its choice and thresholds), and the scoring
runs of the other open guards the project measured but the card does not report.

## Licence

Code: MIT (see `LICENSE`). The scorecards and model weights are under their own licences (model: CC-BY-NC-4.0).
