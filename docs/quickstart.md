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

# Optional: install Studio web interface, test suite, and LM Evaluation Harness extras
pip install -e ".[dev,eval,studio]"
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

## 5. PruneQuest Studio: Local-First Web Interface (`prunequest studio`)

PruneQuest Studio offers an intuitive, browser-based user experience for zero-configuration pruning:

```bash
# Launch Studio (automatically binds to 127.0.0.1 and opens your default browser)
prunequest studio

# Custom port or headless launch:
prunequest studio --port 9000 --no-browser
```

### Python API Integration
```python
import prunequest

# Launch Studio programmatically
prunequest.studio(port=8765)
```

---

## 6. One-Command Optimization (`prunequest optimize`)

The recommended way to prune any model is using `prunequest optimize`. You do not need to understand FLAP, Block Influence, JIPO, calibration, or manual YAML configurations:

```bash
# 1. Default optimization (balanced preset, auto strategy, 99% retention on GSM8K)
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct

# 2. Specify target retention and benchmark
prunequest optimize meta-llama/Llama-3.2-1B-Instruct --benchmark gsm8k --retain 0.98

# 3. Optimize on custom local datasets (JSON, JSONL, CSV, Parquet) or Hugging Face datasets
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct --dataset data/qa.jsonl --target-retain 0.98

# 4. Choose a budget preset: quick (diagnostic check), balanced (recommended), thorough (maximum search depth)
prunequest optimize Qwen/Qwen2.5-Math-1.5B-Instruct --preset quick

# 5. Dry-run preflight inspection without downloading weights
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct --dry-run

# 6. Output pure machine-readable JSON summary or stream events
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct --json
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct --json-events

# 7. Disable interactive live dashboard for plain log output
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct --no-progress
```

### Standardized Run Artifacts (Schema v2)

Every `prunequest optimize` execution produces:

```text
runs/qwen2_5_0_5b_instruct_20261111_120000/
├── best/                  # Standalone Hugging Face safetensors checkpoint (directly loadable)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── manifest.json          # Schema v2: Complete provenance, versions, parameters, and verification status
├── results.json           # Machine-readable performance metrics
├── events.jsonl           # Sequential, structured optimization event logs
├── resolved_config.yaml   # Fully resolved configuration for 100% deterministic reproduction
└── reports/
    ├── summary.md         # Executive overview and verified loading snippet
    ├── detailed_report.md # Technical audit report with candidate decisions
    ├── history.csv        # Comprehensive candidate iteration history
    └── plots/             # Accuracy vs. parameters and non-dominated Pareto frontier
```

---

## 7. Universal Dataset Onboarding & Zero-YAML Evaluation

PruneQuest allows using custom datasets without creating benchmark YAML files. Pass local files (`.json`, `.jsonl`, `.csv`, `.parquet`) or Hugging Face dataset IDs directly:

```bash
# 1. Inspect dataset schema, candidate columns, row count, and auto-inferred task
prunequest datasets inspect ./data/my_eval.jsonl

# 2. Preview formatted evaluation prompts and target completions
prunequest datasets preview ./data/my_eval.jsonl --samples 3

# 3. Validate split partitions, duplicate detection, and data leakage prevention
prunequest datasets validate ./data/my_eval.jsonl

# 4. Run autonomous optimization against your dataset
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct \
  --dataset ./data/my_eval.jsonl \
  --target-retain 0.98 \
  --preset balanced
```

### Python API Integration

```python
import prunequest

# Inspect dataset
report = prunequest.inspect_dataset("./data/my_eval.jsonl")
print(f"Detected task: {report.detected_task.value}, samples: {report.schema_info.num_records}")

# Validate data integrity
val_report = prunequest.validate_dataset("./data/my_eval.jsonl")
assert val_report.valid, f"Dataset validation failed: {val_report.errors}"

# Preview sample prompts
samples = prunequest.preview_dataset("./data/my_eval.jsonl", limit=3)
for s in samples:
    print(s["prompt"])

# Autonomous one-command optimization
result = prunequest.optimize(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dataset="./data/my_eval.jsonl",
    target_retain=0.98,
)
```

---

## 8. Experiment Management (`prunequest runs`)

PruneQuest provides commands to discover, inspect, compare, and view logs of prior optimization experiments without loading model weights:

```bash
# List experiments with filtering and sorting
prunequest runs list --sort newest
prunequest runs list --status completed --model qwen --limit 10

# Inspect complete run provenance and parameters
prunequest runs show qwen2_5_0_5b_instruct_20261111_120000

# Compare two runs and verify evaluation consistency
prunequest runs compare run_alpha run_beta

# View structured event logs
prunequest runs logs run_alpha --phase validation
prunequest runs logs run_alpha --category candidate_accepted

# Regenerate reports and Pareto plots offline
prunequest report runs/qwen2_5_0_5b_instruct_20261111_120000
```

---

## 9. 5-Minute Quickstart: Pipeline Smoke Test

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

## 10. Production Workflows

For power users requiring granular control over algorithmic parameters, PruneQuest provides direct commands:

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

## 8. Resuming Interrupted Runs

PruneQuest saves experiment state transactionally. If an experiment is interrupted, resume it with:

```bash
# One-command resume for any run directory
prunequest resume ./runs/qwen2_5_0_5b_instruct_20261111_120000

# Or strategy-specific resume
prunequest flap resume --run-dir ./runs/qwen_math_flap
```

Resume validates baseline scores, model architecture, and checkpoint integrity before resuming search.

---

## 9. Checkpoint Export & Independent Validation

Export accepted checkpoints to Hugging Face `safetensors` format with out-of-process reload validation:

```bash
prunequest export runs/qwen_block_run/checkpoints/checkpoint_layers_27 --output ./exported_model --validate
```

The subprocess validator ensures `config.json`, tokenizer, finite weights (no NaNs/Infs), finite logits, and generation succeed in an isolated interpreter before confirming export.

---

## 10. Python API Usage

### High-Level One-Command Optimization

```python
from prunequest import optimize

# Optional live event subscriber
def on_progress(event):
    print(f"[{event.phase}] {event.event_type}: {event.payload}")

# Automatically plans, executes, validates, and exports the optimized model
result = optimize(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    benchmark="gsm8k",
    retain=0.99,
    preset="balanced",
    on_progress=on_progress,
)

print(f"Status: {result.status}")
print(f"Parameters pruned: {result.params_pruned:,}")
print(f"Accuracy: {result.final_accuracy*100:.2f}% (Baseline: {result.baseline_accuracy*100:.2f}%)")
print(f"Verified Checkpoint: {result.best_checkpoint}")
```

### Framework API

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
