# partyhat-safety-encoder-eval: data samples

Generated from commit `ca598a68c892` (2026-11-20) of github.com/partyhat-ai/partyhat-safety-encoder-eval by `scripts/llms_digest.py`, which `.github/workflows/llms-digest.yml` reruns on every push to `main`, so this text matches the code at that commit.

Data files are not reproduced in full. One sample per top-level folder, first 40 lines:

## `evidence/` (200 files)

Sample: `evidence/latency/qwen3guard_gen_0.6b_A10G_bf16.json`

```json
{
 "gpu": "NVIDIA A10G",
 "engine": "transformers",
 "bf16": {
  "128": {
   "total_input_tokens": 417,
   "decision": {
    "cold_ms": 1383.7,
    "bs1": {
     "n": 1000,
     "p50": 109.94,
     "p95": 113.74,
     "p99": 115.91,
     "mean": 109.8
    },
    "batches": {
     "1": {
      "p95_ms": 117.8,
      "items_per_s": 8.9
     },
     "8": {
      "p95_ms": 248.5,
      "items_per_s": 32.9
     },
     "16": {
      "p95_ms": 366.05,
      "items_per_s": 44.4
     },
     "32": {
      "p95_ms": 602.49,
      "items_per_s": 53.4
     },
     "64": {
      "p95_ms": 1137.78,
      "items_per_s": 56.4
     }
    },
    "best_batch_at_p95_target": null,
    "items_per_s_at_target": null,
    "items_per_s_unconstrained": 56.4,
```

## `training/` (1 files)

Sample: `training/training_config.json`

```json
{
  "checkpoint_role": "Safety Encoder, release 2.0: recipe v6, the teacher-labels variant, training run 1 (seed 13). In the code and scorecards this is v6_R3_s13.",
  "trained": "2026-11-17",
  "operating_threshold": 0.45514049134720697,
  "thresholds": {
    "T_bal": {
      "value": 0.45514049134720697,
      "use": "balanced: maximises mean F1 on three development sets; the model card's default"
    },
    "T_gate": {
      "value": 0.9569402169376223,
      "use": "very-low-false-positive rule: lowest threshold with FPR <= 1% on development benign prompts; unstable, see the model card"
    },
    "T_screen": {
      "value": 0.04342702474724448,
      "use": "pre-screen: passes 98% of development unsafe prompts on to a heavier guard"
    }
  },
  "verdict_mode": "P(unsafe) = softmax(logits)[1], computed as sigmoid(logit_unsafe - logit_safe); a prompt is flagged when P(unsafe) >= the chosen threshold.",
  "threshold_provenance": "All three chosen on development data only (ToxicChat validation split, Aegis 2.0 validation, a 3% WildGuardMix holdout), each by a fixed rule, before any test set was scored.",
  "base_model": "microsoft/deberta-v3-large",
  "base_model_revision": "64a8c8eab3e352a784c658aef62be1662607476f",
  "base_model_note": "MIT. Trained from the bare base model (no warm start); fresh 2-class head.",
  "input": "a single user prompt, English, truncated at 512 tokens",
  "recipe": {
    "precision": "FP32 weights and activations (TF32 matmuls)",
    "lr": 2e-05,
    "schedule": "linear, warmup 10%",
    "effective_batch": 32,
    "steps": 9396,
    "loss": "cross-entropy on label-smoothed soft targets",
    "seed": 13,
    "checkpoint": "final step (no checkpoint chosen on validation)"
  },
  "training_data": [
    "lmsys/toxic-chat toxicchat0124 training split minus the 509-prompt validation split (x3)",
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0 train.json prompts (labels as shipped)",
    "allenai/wildguardmix wildguardtrain prompts",
    "allenai/wildjailbreak train (vanilla and adversarial, harmful and benign)",
    "AmazonScience/FalseReject train prompts (benign)",
```
