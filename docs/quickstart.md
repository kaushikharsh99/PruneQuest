# Quickstart Guide: PruneQuest v0.1.0

PruneQuest is an automated, benchmark-guided structured pruning framework designed for large language models. It enables systematic parameter reduction while guaranteeing task-specific benchmark retention.

---

## 1. Installation

PruneQuest requires Python 3.10+ and PyTorch with CUDA support.

```bash
# Clone the repository
git clone https://github.com/kaushikharsh99/PruneQuest.git
cd PruneQuest

# Install in editable mode
pip install -e .
```

Verify your hardware and runtime environment:

```bash
prunequest doctor
```

Output confirms GPU VRAM, CUDA compatibility, PyTorch versions, and memory headroom.

---

## 2. Inspecting a Target Model

Before pruning, inspect the model's structural parameters and layer counts:

```bash
prunequest inspect --model Qwen/Qwen2.5-Math-1.5B-Instruct
```

---

## 3. Running an Evaluation Baseline

Establish the unpruned baseline accuracy on GSM8K:

```bash
# Smoke test (16 samples)
prunequest baseline --model Qwen/Qwen2.5-Math-1.5B-Instruct --split smoke

# Full evaluation (1,319 GSM8K test samples)
prunequest baseline --model Qwen/Qwen2.5-Math-1.5B-Instruct --split full
```

---

## 4. Launching Automated Pruning Search

Execute greedy layer pruning guided by Block Influence (BI), targeting $\ge 99\%$ accuracy retention:

```bash
prunequest run \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --benchmark gsm8k \
    --retain 0.99 \
    --max-iterations 3 \
    --split smoke \
    --output-dir ./runs/qwen_run_01
```

During execution, PruneQuest:
1. Computes baseline accuracy on the chosen benchmark split.
2. Evaluates layer importance using token-level cosine distances (Block Influence).
3. Screens top candidate layers on a fast screening subset.
4. Validates full candidates and automatically rolls back if accuracy dips below 99% retention.
5. Exports the pruned checkpoint with standalone Hugging Face `safetensors` and `prunequest_metadata.json`.

---

## 5. Resuming an Interrupted Experiment

If a run is interrupted or halted:

```bash
prunequest resume ./runs/qwen_run_01
```

State is persisted automatically to an SQLite database (`experiments.db`) and JSONL run traces.

---

## 6. Python API Usage

PruneQuest can be used directly as a Python library:

```python
from prunequest import PruneQuest

# Initialize pruner session
quest = PruneQuest(
    model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    benchmark="gsm8k",
    retain=0.99,
    max_iterations=4,
    device="cuda:0",
    split="smoke",
)

# Run end-to-end pruning loop
result = quest.run()

print(f"Status: {result.status}")
print(f"Best checkpoint: {result.best_checkpoint}")
print(f"Original params: {result.original_params:,}")
print(f"Final params:    {result.final_params:,}")
print(f"Accuracy retention: {result.accuracy_retention_pct:.2f}%")
```
