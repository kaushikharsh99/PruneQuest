# Legacy Migration Audit & Architecture Transition

## 1. Executive Summary

This document audits the legacy pruning implementation located at `/home/harsh/coding/prunning` and defines the formal migration strategy into **PruneQuest v0.1.0**.

The legacy prototype experimentally verified that structured layer pruning guided by ShortGPT Block Influence and fine-grained SwiGLU MLP channel pruning can compress `Qwen/Qwen2.5-Math-1.5B-Instruct` while retaining zero-shot GSM8K mathematical reasoning within a 6 GB VRAM physical budget (NVIDIA GeForce RTX 3050 Laptop GPU).

PruneQuest transitions this proof-of-concept into a modular, production-ready open-source framework equipped with clean abstractions, an extensible model and benchmark adapter registry, robust experiment tracking, transactional resumption, and a unified Typer/Rich CLI and Python API.

---

## 2. Legacy Technical Audit: 10 Critical Questions

| # | Question | Finding in Legacy Codebase | Migration Decision |
|---|:---|:---|:---|
| 1 | **Which pruning algorithms work?** | Transformer Layer Pruning (`LayerPruningBackend`) physically deletes blocks from `ModuleList`. SwiGLU MLP channel slicing (`MLPPruningBackend`) slices `gate_proj`, `up_proj`, and `down_proj`. | **Reuse & Refactor**: Formalize into `LayerPruner` (v0.1.0 focus) with `BasePruner` interface; keep MLP pruner as extensible backend. |
| 2 | **Is ShortGPT/ShortX integrated?** | Yes, token-wise cosine distance Block Influence ($BI_l = 1 - \text{mean}(\cos(H^{(l-1)}, H^{(l)}))$) is implemented in `src/pruning_lab/importance.py`. | **Reuse**: Port into `prunequest.pruning.importance`. |
| 3 | **How are layer-importance scores calculated?** | Via calibration forward passes with `output_hidden_states=True`, computing inter-block cosine distance over all tokens. | **Reuse & Optimize**: Retain bounded calibration passes (default: 16–32 samples) to keep activation VRAM low. |
| 4 | **Are pruned layers physically removed or merely disabled?** | Physically removed from `model.model.layers` ModuleList. Serialized models have genuinely fewer parameters and reduced disk size. | **Preserve**: Strict requirement: no silent weight masking. |
| 5 | **Does the evaluation system correctly support Qwen2.5-Math?** | Yes, uses official Qwen chat template (`<|im_start|>system...`), greedy decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=512`), and Hugging Face `math-verify` + LaTeX boxed extraction. | **Reuse & Refactor**: Wrap into modular `GSM8KBenchmark` adapter adhering to `BaseBenchmark`. |
| 6 | **Can the checkpoint exporter save a valid smaller model?** | Yes, saves safetensors, updates `num_hidden_layers` and `layer_types` (Transformers v5), and validates via isolated subprocess reload. | **Reuse & Formalize**: Relocate to `prunequest.export.checkpoint`. |
| 7 | **Are iteration results already stored?** | Stored in SQLite (`experiments.db`) and JSONL generation logs. | **Refactor**: Wrap into `prunequest.core.tracking.ExperimentTracker` with atomic commit support. |
| 8 | **Can existing runs resume after interruption?** | Checkpoints and DB exist, but resumption required manual script orchestration. | **Replace**: Implement full automatic resumption in `prunequest.core.search` and `prunequest.core.controller`. |
| 9 | **What code can become generic?** | Search loop, experiment tracker, checkpoint manager, metrics calculations, CLI, and reporting. | **Generalize**: Move to `prunequest.core`, `prunequest.reporting`, and `prunequest.export`. |
| 10 | **What code is coupled to one model/benchmark?** | Qwen2 layer indexing/SwiGLU naming and GSM8K question parsing/boxed regex. | **Isolate into Adapters**: `Qwen2Adapter` under `models/` and `GSM8KBenchmark` under `benchmarks/`. |

---

## 3. Migration Action Matrix

| Legacy Component | Legacy Path | Target Module in PruneQuest | Migration Action | Rationale |
|:---|:---|:---|:---:|:---|
| Mathematical metrics & Wilson CI | `src/pruning_lab/metrics.py` | `prunequest.reporting.metrics` | **Reuse** | Mathematically sound Wilson binomial score interval calculation. |
| GSM8K dataset partitions | `src/pruning_lab/datasets.py` | `prunequest.benchmarks.gsm8k` | **Refactor** | Convert to benchmark adapter with explicit split configuration. |
| Math verification & parser | `src/pruning_lab/evaluator.py` | `prunequest.benchmarks.gsm8k` | **Refactor** | Standardize into `GSM8KBenchmark.evaluate_batch` using `math-verify`. |
| Memory utilities & model load | `src/pruning_lab/model_loader.py` | `prunequest.models.base`, `prunequest.models.qwen2` | **Refactor** | Split device/memory helpers and model adapter logic. |
| ShortGPT Block Influence | `src/pruning_lab/importance.py` | `prunequest.pruning.importance` | **Reuse** | Proven calibration-based redundancy ranking. |
| Layer pruning mechanics | `src/pruning_lab/pruning/layer.py` | `prunequest.pruning.layers`, `prunequest.models.qwen2` | **Refactor** | Decouple model-specific layer slicing into `Qwen2Adapter`. |
| MLP SwiGLU pruning | `src/pruning_lab/pruning/mlp.py` | `prunequest.pruning.mlp` | **Refactor** | Keep as secondary structured pruning backend. |
| Attention analysis & blocker | `src/pruning_lab/pruning/attention.py` | `docs/architecture.md`, `prunequest.pruning.attention` | **Reuse** | Documents HF Qwen2 RoPE/head_dim architectural limitation. |
| Checkpoint manager | `src/pruning_lab/checkpoint.py` | `prunequest.export.checkpoint`, `prunequest.export.validation` | **Refactor** | Separate serialization from isolated subprocess validation. |
| SQLite & JSONL tracking | `src/pruning_lab/tracking.py` | `prunequest.core.tracking` | **Refactor** | Add atomic updates, configuration fingerprinting, and run resumption. |
| Matplotlib reporting | `src/pruning_lab/reporting.py` | `prunequest.reporting.reports` | **Refactor** | Modernize styling, add CLI summary cards via Rich. |
| Argparse CLI | `src/pruning_lab/cli.py` | `prunequest.cli` | **Replace** | Replace with modern Typer + Rich interactive CLI. |
| Master YAML config | `configs/default.yaml` | `configs/default.yaml`, `prunequest.config` | **Refactor** | Add Pydantic validation schema for type safety. |

---

## 4. Preservation Statement

The legacy repository `/home/harsh/coding/prunning` remains intact. All historical checkpoints (`checkpoint_layers_25`, `checkpoint_mlp_size_8512`), SQLite databases (`experiments.db`), JSONL generations, and benchmark traces will not be deleted or overwritten.
