# Benchmarks & Answer Extraction in PruneQuest

PruneQuest enforces rigorous benchmark evaluations to guide pruning decisions. The primary benchmark for mathematical reasoning is **GSM8K** (`openai/gsm8k`).

---

## 1. GSM8K Integration

### 1.1 Dataset Partitions
PruneQuest partitions the GSM8K test set deterministically (using fixed random seed `42`):

| Partition | Size | Purpose |
| :--- | :--- | :--- |
| **`smoke`** | 16 samples | Rapid functional tests, sanity checks, CI/CD validation. |
| **`screening`**| 64 samples | Fast candidate filter during search iterations. |
| **`dev`** | 512 samples | Intermediate pruning decisions and search optimization. |
| **`full`** | 1,319 samples | Final publication-grade evaluation and Pareto frontier mapping. |
| **`calibration`**| 256 samples | Block Influence hidden state calculation (drawn from train split). |

---

## 2. Mathematical Prompting & Chat Templates

Modern reasoning models like `Qwen/Qwen2.5-Math-1.5B-Instruct` are sensitive to prompt format. PruneQuest applies the model's official chat template:

```
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

This enforces consistent Chain-of-Thought (CoT) generation and ensures final answers are formatted for parsing.

---

## 3. Answer Verification: `math-verify`

Standard LLM evaluators often rely on rigid string matching (`#### 42`), failing when the model responds with LaTeX formatting like `\boxed{42}` or equivalent fractions.

PruneQuest uses a tiered verification pipeline:
1. **Primary**: Hugging Face [`math-verify`](https://github.com/huggingface/Math-Verify) parses mathematical expressions, normalizing numbers, fractions, units, and expressions.
2. **Secondary**: Strict LaTeX `\boxed{...}` extraction with regex normalization.
3. **Tertiary**: Standard GSM8K `#### <number>` trailing pattern match.

Accuracy is reported alongside **95% Wilson Score Confidence Intervals** to provide statistically robust retention assessments.
