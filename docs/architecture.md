# PruneQuest Architecture & Design Principles

PruneQuest is designed around modularity, hardware safety on resource-constrained GPUs (such as 6 GB mobile GPUs), and physical parameter reduction.

---

## 1. High-Level Architecture Overview

```
                           +---------------------------+
                           |  CLI (Typer) / Python API |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |     SearchController      |
                           +-------------+-------------+
                                         |
        +--------------------------------+--------------------------------+
        |                                |                                |
        v                                v                                v
+------------------+           +--------------------+           +------------------+
|   ModelAdapter   |           |     BasePruner     |           |  BaseBenchmark   |
|  (e.g., Qwen2)   |           |  (e.g. LayerPruner)|           | (e.g. GSM8K)     |
+------------------+           +--------------------+           +------------------+
        |                                |                                |
        v                                v                                v
Physical Surgery & Config       Block Influence Cosine           math-verify & CoT
Index Synchronization            Importance Scoring               Deterministic Eval
        |                                |                                |
        +--------------------------------+--------------------------------+
                                         |
                                         v
                           +---------------------------+
                           |    ExperimentTracker      |
                           |   (SQLite + JSONL Traces) |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |    CheckpointExporter     |
                           |   (Safetensors + Meta)    |
                           +---------------------------+
```

---

## 2. Core Modules

### 2.1 Model Adapters (`prunequest.models`)
- **`BaseModelAdapter`**: Abstract interface defining layer discovery, physical weight surgery, forward-pass health checks, and checkpoint serialization.
- **`Qwen2Adapter`**: Handles `Qwen2ForCausalLM`. Implements:
  - Exact `nn.ModuleList` element deletion.
  - Synchronization of `model.config.num_hidden_layers` and `model.config.layer_types`.
  - Re-indexing of layer indices (`layer.self_attn.layer_idx = idx`) across remaining layers to prevent RoPE cache mismatches.
- **`ModelRegistry`**: Factory pattern decoupling model implementations from search algorithms.

### 2.2 Pruning Backends (`prunequest.pruning`)
- **`LayerPruner`**: Physical transformer layer pruner. Computes token-level cosine distances (Block Influence) across hidden state representations.
- **Candidate Generator**: Proposes candidate removals ordered by minimal influence score.

### 2.3 Evaluation & Benchmarks (`prunequest.benchmarks`)
- **`GSM8KBenchmark`**: Zero-shot Chain-of-Thought (CoT) evaluation with deterministic sample subsets (`smoke=16`, `dev=512`, `full=1319`).
- **Official Chat Template**: Formats math prompts using Qwen's chat template (`<|im_start|>system...<|im_start|>user...`).
- **Answer Verification**: Multi-stage parsing prioritizing Hugging Face `math-verify`, with regex LaTeX `\boxed{}` and numeric exact match fallbacks.

### 2.4 Search Strategies (`prunequest.core`)
- **`GreedySearchStrategy`**: Evaluates candidate layer removals sequentially.
- **Screening Fast-Path**: Screens candidates on a compact subset before full evaluation, reducing compute time by up to 70%.
- **Rollback Engine**: In-memory deep copies and disk checkpoints ensure rollback whenever accuracy falls below the retention threshold ($R = \frac{\text{Acc}_{\text{pruned}}}{\text{Acc}_{\text{baseline}}} < \tau$).

### 2.5 Experiment Tracking & Persistence (`prunequest.core.tracking`)
- **`ExperimentTracker`**: Manages run directories, logging to both SQLite (`experiments.db`) and streaming JSONL (`run_trace.jsonl`).
- Enables clean resumption after interrupts or system crashes.

### 2.6 Checkpoint Export & Validation (`prunequest.export`)
- **`CheckpointExporter`**: Produces standalone Hugging Face `safetensors` checkpoints loadable by standard `transformers.AutoModelForCausalLM.from_pretrained()`.
- Writes `prunequest_metadata.json` containing layer count, parameter count, baseline comparison, and retention metrics.
- **`SubprocessValidator`**: Launches an independent Python subprocess to verify clean reloadability, preventing corrupted checkpoints from entering production.

---

## 3. GPU Memory Management (6 GB Budget)

On laptop GPUs with 6 GB VRAM, loading two 1.5B parameter models in memory simultaneously causes immediate CUDA Out-Of-Memory errors. PruneQuest enforces:
1. **Explicit CPU Offloading**: Before child subprocess reload tests or candidate comparisons, weights are temporarily offloaded or cleared.
2. **Deterministic Garbage Collection**: `gc.collect()` followed by `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()`.
3. **MKL GNU Threading Guard**: Setting `MKL_THREADING_LAYER=GNU` and importing `numpy` before `torch` prevents deadlocks between OpenMP and Intel MKL in spawned workers.
