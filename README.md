# Scratchpad Transparency in Sandbagging
### Does Chain-of-Thought Reasoning Aid or Impede Behavioural Monitoring?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassanql/sandbagging-detection/blob/main/sandbagging_cot.ipynb)

---

## Overview

**Sandbagging** — an AI system deliberately underperforming on capability evaluations — is a central threat model in AI control. A model that can sandbag undetectably defeats evaluation-based oversight entirely.

This repository contains the full code and data for an empirical study asking: **when a model is instructed to sandbag, does its chain-of-thought reasoning trace betray it?**

We run Mistral-7B-Instruct-v0.2 on MMLU under six conditions (honest / soft-sandbag / hard-sandbag, with and without chain-of-thought), extract token-level log-probabilities and entropy from both the reasoning trace and the final answer, and compare three logistic regression monitors trained on different feature sets.

---

## Key Results

| Monitor | Features | AUC | Std |
|---|---|---|---|
| A | Answer only (no CoT) | 0.831 | ±0.167 |
| B | CoT reasoning only | **0.951** | ±0.073 |
| C | Combined (A + B) | **0.951** | ±0.073 |

**Monitor B = Monitor C.** Adding answer features to reasoning features provides zero additional discriminating power. The scratchpad is the dominant signal source.

### Four main findings

**1. CoT reasoning features dominate detection (+0.12 AUC)**
A monitor trained only on reasoning-trace features (entropy, length, entropy variance) outperforms an answer-only monitor by 0.12 AUC. The single most discriminating feature is the *standard deviation* of per-token entropy across the reasoning trace — capturing localised uncertainty spikes, not just average uncertainty.

**2. Sandbagging produces significantly elevated reasoning entropy**
Both sandbagging conditions produce significantly higher token-level entropy than honest reasoning:

| Condition | Mean entropy | p vs honest |
|---|---|---|
| cot-honest | 0.168 ± 0.163 | — |
| cot-hard | 0.404 ± 0.153 | p < 10⁻⁴ |
| cot-soft | 0.521 ± 0.104 | p < 10⁻¹⁰ |

**3. Correct-answer leakage is paradoxically higher under soft sandbagging**

| Condition | Correct answer in reasoning |
|---|---|
| cot-honest | 4.6% |
| cot-soft | **68.2%** |
| cot-hard | 22.7% |

Soft sandbagging leaks more than hard — not because it is less deceptive, but because it is more incoherent. The ambiguous instruction ("act like a weaker model") creates more unresolved conflict than the explicit instruction ("give a wrong answer").

**4. A three-mode taxonomy of hard CoT sandbagging** (n=19 sandbagging cases):

| Mode | Rate | Description | Detectable? |
|---|---|---|---|
| Thinking out loud | 21% (4/19) | Correct answer stated in reasoning, wrong answer given | Trivially yes |
| Clean deception | 21% (4/19) | Coherent false justification, no scratchpad leak | Via entropy only |
| Compliance failure | 58% (11/19) | No valid answer letter produced | Trivially yes |

Only 21% of hard sandbagging attempts produce the most dangerous outcome (coherent false reasoning with no scratchpad leak). Even these cases exhibit elevated token-level entropy.

---

## Experimental Design

**2 × 3 within-subjects design** — every question run under all six conditions with greedy decoding (temperature = 0).

```
                    No CoT              With CoT
                 ┌─────────────┐    ┌─────────────┐
  Honest         │ direct-hon  │    │  cot-hon    │
  Soft sandbag   │ direct-soft │    │  cot-soft   │
  Hard sandbag   │ direct-hard │    │  cot-hard   │
                 └─────────────┘    └─────────────┘
```

- **Model:** Mistral-7B-Instruct-v0.2 (4-bit NF4 quantisation)
- **Benchmark:** MMLU — 13 questions × 12 subjects = 117 total
- **Filtered sample:** 22 questions correct under both honest conditions
- **Hardware:** Single NVIDIA T4 GPU (Google Colab free tier)
- **Runtime:** ~2.5 hours

### Pre-registered hypotheses

| Hypothesis | Prediction | Result |
|---|---|---|
| H1 | CoT + answer features outperform answer-only | ✅ Supported (ΔAUC = +0.12) |
| H2 | Hard sandbagging leaks correct answer into reasoning more than honest | ✅ Trend (p = 0.082) |
| H3 | Reasoning entropy higher in sandbagging conditions | ✅ Strongly supported (p < 10⁻⁴) |
| H4 | Soft sandbagging harder to detect with CoT | ❌ Opposite — CoT detection = 1.000 |

---

## Repository Structure

```
sandbagging-detection/
│
├── sandbagging_cot.ipynb          ← Full experiment (Experiment 2, main paper)
├── sandbagging_detection.ipynb    ← Pilot experiment (Experiment 1, no CoT baseline)
│
├── results/
│   ├── sandbagging_cot_results.png    ← 9-panel main results figure
│   ├── sandbagging_cot_features.csv   ← Per-question feature matrix (all 6 conditions)
│   ├── summary_cot.json               ← Headline numbers
│   └── checkpoint_results.json        ← Raw per-question outputs
│
│
└── README.md
```

---

## Reproducing the Results

### Requirements

```bash
pip install transformers accelerate bitsandbytes datasets \
            scipy scikit-learn matplotlib seaborn pandas
```

### Running on Google Colab (recommended)

1. Open [`sandbagging_cot.ipynb`](sandbagging_cot.ipynb) in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4)
3. Run all cells top to bottom
4. Expected runtime: **~2.5 hours**
5. Results auto-save to `checkpoint_results.json` every 10 questions — safe to resume after disconnect

### Running locally

You need a GPU with at least 16 GB VRAM. With 4-bit quantisation, Mistral-7B fits comfortably on a T4 or equivalent.

```bash
jupyter notebook sandbagging_cot.ipynb
```

### Checkpointing

The notebook saves progress every 10 questions. If your session disconnects, re-run the experiment cell — it will detect the checkpoint file and resume from where it left off automatically. **Download `checkpoint_results.json` periodically as a backup** — this file lives in the Colab runtime and will be lost if the session resets completely.

---

## Features Extracted

### Answer-level features (all conditions)

| Feature | Description |
|---|---|
| `ans_correct_lp` | Log-prob of correct answer letter as next token |
| `ans_chosen_lp` | Log-prob of model's chosen letter |
| `ans_logprob_gap` | correct_lp − chosen_lp (positive = model prefers correct despite choosing otherwise) |
| `ans_correct_rank` | Rank of correct answer by log-prob (1 = top) |
| `ans_chosen_rank` | Rank of chosen answer |
| `ans_entropy` | Entropy over {A,B,C,D} at answer position |

### Reasoning-level features (CoT conditions only)

| Feature | Description |
|---|---|
| `cot_reasoning_entropy` | Mean per-token entropy across reasoning tokens |
| `cot_reasoning_entropy_std` | Std of per-token entropy (captures uncertainty spikes) |
| `cot_reasoning_mean_lp` | Mean log-prob of reasoning tokens (fluency proxy) |
| `cot_reasoning_len` | Reasoning length in tokens |
| `cot_answer_lp` | Mean log-prob of tokens after `Answer:` |
| `cot_reasoning_to_answer_gap` | reasoning_mean_lp − answer_lp |
| `cot_correct_in_reasoning` | Binary: correct answer letter appears in reasoning text |

---

## Monitor Performance Detail

### Monitor C top feature coefficients

| Feature | Coefficient | Direction |
|---|---|---|
| `cot_reasoning_len` | +1.618 | Sandbagging = longer reasoning |
| `cot_reasoning_entropy_std` | +1.354 | Sandbagging = more variable entropy |
| `cot_reasoning_entropy` | +0.657 | Sandbagging = higher mean entropy |
| `cot_answer_lp` | −0.557 | Sandbagging = less confident answer |
| `cot_correct_in_reasoning` | +0.537 | Sandbagging = correct answer leaked |
| `ans_correct_rank` | +0.385 | Sandbagging = correct answer lower-ranked |
| `cot_reasoning_to_answer_gap` | +0.350 | Sandbagging = fluent reasoning, uncertain answer |

---

## Limitations

This is a pilot study. Key limitations:

- **Sample size:** n=22 filtered questions. Effect sizes are large (Cohen's d > 1 for entropy comparisons) but AUC confidence intervals are wide. Treat as directional, not definitive.
- **Model scale:** CoT *reduced* honest accuracy (28.2% vs 41.9% direct), consistent with CoT degrading performance below a capability threshold. Results may differ substantially for larger models.
- **Single prompt template:** Results should be validated across prompt variants.
- **Prompted sandbagging only:** Models fine-tuned to sandbag may suppress entropy signals and scratchpad leakage entirely — the most important open question for future work.
- **Multiple-choice format:** Enables clean logprob extraction but limits ecological validity. Free-form sandbagging detection is substantially harder.

---

## Connection to Prior Work

This experiment extends and complements:

- **van der Weij et al. (2024)** — *AI Sandbagging* — establishes that prompted sandbagging works and is more convincing in RLHF'd models. We extend by asking whether CoT aids detection.
- **Greenblatt et al. (2024)** — *AI Control* — the threat model we operationalise.
- **Lanham et al. (2023)** — *Measuring Faithfulness in CoT* — establishes partial decoupling of reasoning and output. We show this decoupling is detectable under sandbagging.
- **Turpin et al. (2023)** — *Language Models Don't Always Say What They Think* — CoT unfaithfulness under biasing features. Sandbagging is an adversarial instance of this.

---


---

*This work was conducted on free-tier compute (Google Colab T4 GPU). Full reproduction requires no paid APIs or proprietary access.*
