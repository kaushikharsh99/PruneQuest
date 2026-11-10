# Quickstart Guide: PruneQuest Release A

**PruneQuest** is an autonomous, benchmark-guided structural pruning framework for Large Language Models. It enables systematic parameter reduction while enforcing strict directional benchmark retention constraints (e.g. $\ge 99\%$ accuracy retention) with transactional rollback and independent checkpoint reload validation.

---

## 1. Installation

PruneQuest requires **Python 3.10+** and PyTorch (supports both GPU and CPU-only environments).

```bash
# Clone the repository
git clone https://github.com/kaushikharsh99/PruneQuest.git
cd PruneQuest

# Core installation (CLI + core pruning framework)
pip install -e .

# Optional: install development testing and LM Evaluation Harness extras
pip install -e ".[dev,eval]"
```

---

## 2. Environment Diagnostics (`prunequest doctor`)

Verify system dependencies, PyTorch, CUDA, VRAM headroom, and optional packages:

```bash
# General environment diagnostics
prunequest doctor

# FLAP-specific hardware diagnostics
prunequest flap doctor
```

The diagnostics report Python version, CUDA availability, GPU device name, free/total VRAM, core libraries, optional extras, workspace write permissions, and supported model capability tiers.

---

## 3. Inspecting a Target Model

Inspect model architecture, parameter count, transformer layer structure, and capability level without downloading weights upfront:

```bash
# Universal architecture inspection
prunequest inspect --model Qwen/Qwen2.5-Math-1.5B-Instruct

# Inspect SwiGLU MLP channel dimensions directly via FLAP
prunequest flap inspect --model Qwen/Qwen2.5-Math-1.5B-Instruct
```

---

## 4. Discovering and Validating Benchmarks

Discover available benchmarks across native adapters, declarative YAML specifications, and optional LM-Eval tasks:

```bash
# List top benchmarks
prunequest benchmarks list --limit 20

# Search available benchmarks
prunequest benchmarks search math
prunequest benchmarks search gsm8k

# Inspect benchmark metadata
prunequest benchmarks info gsm8k

# Validate a custom declarative YAML benchmark specification
prunequest benchmarks validate --spec configs/default.yaml
```

---

## 5. 5-Minute Quickstart: Pipeline Smoke Test

To verify that the entire pipeline (scoring, candidate proposal, screening, checkpoint export, and subprocess reload validation) functions on your system, execute a lightweight 1-iteration smoke check:

```bash
prunequest block-based run \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --benchmark gsm8k \
    --retain 0.99 \
    --max-iterations 1 \
    --split smoke \
    --output ./runs/qwen_smoke_run
```

> **Note**: Smoke mode evaluates a 16-sample diagnostic subset to verify execution and checkpoint serialization. It is intended for engineering validation, not for reliable population-level accuracy claims.

---

## 6. Production Workflows

For production-grade compression, PruneQuest provides three distinct workflows evaluated across full development partitions:

### Workflow A: Block Influence (Structured Depth / Layer Pruning)
Greedy layer excision ranking transformer blocks by token-level cosine distances:

```bash
prunequest block-based run \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --benchmark gsm8k \
    --retain 0.99 \
    --max-iterations 3 \
    --split dev \
    --output ./runs/qwen_block_run
```

### Workflow B: FLAP (Structured Width / MLP Channel Pruning)
Fluctuation-based Adaptive Structured Pruning on SwiGLU intermediate channels with adaptive step-back halving:

```bash
# 1. Profile FLAP layer importance scores
prunequest flap profile --config configs/qwen_math_flap.yaml

# 2. Run incremental width pruning optimization
prunequest flap optimize --config configs/qwen_math_flap.yaml --batch-size 32

# 3. Independently test exported checkpoint
prunequest flap test --checkpoint runs/qwen_math_flap/checkpoints/best --split acceptance
```

### Workflow C: JIPO Full Sweep (Joint Width & Depth Optimization)
Joint Incremental Pruning Optimizer dynamically searching depth excisions and width slicing:

```bash
prunequest full-sweep --config configs/full_sweep.yaml

# Or with runtime CLI overrides:
prunequest full-pass \
    --config configs/full_sweep.yaml \
    --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --retain 0.95 \
    --max-iterations 10 \
    --device "cuda:0" \
    --output "runs/full_sweep" \
    --resume
```

---

## 7. Resuming Interrupted Runs

PruneQuest saves experiment state transactionally. If an experiment is interrupted:

```bash
# Resume Block Influence run
prunequest resume ./runs/qwen_block_run

# Resume FLAP run
prunequest flap resume --run-dir ./runs/qwen_math_flap
```

Resume validates baseline scores, model architecture, and checkpoint integrity before resuming search.

---

## 8. Checkpoint Export & Independent Validation

Export accepted checkpoints to Hugging Face `safetensors` format with out-of-process reload validation:

```bash
prunequest export runs/qwen_block_run/checkpoints/checkpoint_layers_27 --output ./exported_model --validate
```

The subprocess validator ensures `config.json`, tokenizer, finite weights (no NaNs/Infs), finite logits, and generation succeed in an isolated interpreter before confirming export.

---

## 9. Python API Usage

PruneQuest can be used directly as a Python library:

```python
from prunequest import PruneQuest

quest = PruneQuest(
    model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    benchmarks=["gsm8k"],
    retain=0.99,
    max_iterations=3,
    device="cuda:0",
    split="dev",
)

result = quest.run()

print(f"Status: {result.status}")
print(f"Best checkpoint: {result.best_checkpoint}")
print(f"Parameters pruned: {result.original_params - result.final_params:,}")
print(f"Accuracy retention: {result.accuracy_retention_pct:.2f}%")
```
