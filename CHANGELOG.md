# Changelog

All notable changes to PruneQuest will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-09

### Added
- **Universal Model Architecture Discovery & Classification:**
  - `ModelInspector`: Inspects Hugging Face configs and local weights without full tensor downloads.
  - Architecture taxonomy (`ArchitectureClass`): `decoder_only_dense`, `mixture_of_experts`, `encoder_decoder`, `encoder_only`, `hybrid_ssm`, `multimodal`, and `quantized`.
  - 4-Tier Capability System:
    - Level 0: Unsupported (architecture fundamentally incompatible with decoder pruning)
    - Level 1: Evaluation (supports evaluation across all benchmarks)
    - Level 2: Experimental (experimental structural/block surgery)
    - Level 3: Verified (production-validated structural pruning, rollback, and reload)
  - Dedicated Model Adapters:
    - `DecoderStackAdapter`: Reusable dense transformer decoder surgery engine.
    - `Qwen2Adapter`: Qwen2 & Qwen2.5 (Level 3 Verified).
    - `LlamaAdapter`: Llama-2 & Llama-3 / 3.1 / 3.2 (Level 3 Verified).
    - `MistralAdapter`: Mistral & Zephyr architectures (Level 3 Verified).
    - `GemmaAdapter`: Gemma & Gemma-2 architectures (Level 2 Experimental).
    - `PhiAdapter`: Phi-2, Phi-3, and Phi-3.5 architectures (Level 2 Experimental).
    - `ModelRegistry`: Automatic adapter resolution via inspection and dynamic instantiation.
- **EleutherAI LM Evaluation Harness Integration:**
  - Runtime task discovery across 14,222+ tasks without hardcoded lists.
  - `LMEvalBenchmarkAdapter`: Wraps in-memory PyTorch models via `HFLM` to prevent weight duplication and VRAM spikes on 6GB GPUs.
  - CLI commands: `prunequest benchmarks list`, `prunequest benchmarks search <term>`, and `prunequest benchmarks info <task>`.
- **Declarative YAML BenchmarkSpec Engine:**
  - Unified specification format for custom benchmarks supporting Hugging Face datasets, JSON/JSONL, CSV, and Parquet.
  - Built-in scorers: `exact_match`, `normalized_match`, `numeric_exact_match`, `math_verify`, and `regex`.
  - `BenchmarkSpecCompiler`: Provides `validate_spec` and `preview_spec` with CLI commands `prunequest benchmarks validate` and `prunequest benchmarks preview`.
- **Directional Metric Specifications & Multi-Benchmark Constraints:**
  - `MetricSpec`: Directional metric evaluation supporting both higher-is-better and lower-is-better objectives with relative retention or degradation constraints.
  - `ConstraintSuite`: Multi-benchmark constraint validation ensuring candidates pass all configured evaluation criteria simultaneously before acceptance.
- **Universal Pruning Backend & Registry:**
  - `PrunerRegistry`: Modular registration and dynamic lookup of pruning algorithms.
  - Added `supports()` validation and `estimate_savings()` across all pruners.
- **Decoupled Search Controller:**
  - Full multi-benchmark baseline evaluation and constraint enforcement.
  - Transactional rollback on candidate retention degradation, export failure, or subprocess reload rejection.
  - Strict baseline persistence across pause/resume cycles.

## [0.1.1] - 2026-11-09

### Fixed
- **CI Linting Stability:** Resolved 232 lint issues and standardized `[tool.ruff.lint]` configuration (`select = ["E", "F", "W"]`, `ignore = ["E501"]`) to guarantee consistent GitHub Actions runs across Python 3.10, 3.11, 3.12, and 3.13.
- **Development Benchmark Enforcement:** Upgraded `SearchController.run()` to enforce a two-phase verification pipeline: fast candidate screening on candidate subsets, followed by rigorous evaluation on the target development benchmark before checkpoint acceptance.
- **Candidate Rejection Safeguard:** Candidates failing the retention threshold on the development benchmark are now automatically rejected, rolled back, and recorded in SQLite with `status="rejected_retention_failed"`.
- **Checkpoint Reload Rejection Safeguard:** When subprocess validation fails (`reload_ok is False`), the controller immediately rolls back the candidate, rejects the checkpoint, and logs `status="rejected_reload_failed"`.
- **Resume Baseline Preservation:** Updated `SearchController.resume()` to query `tracker.get_baseline()`, strictly preserving the original unpruned reference accuracy and parameter counts instead of re-establishing a baseline on an already-pruned checkpoint.
- **Offline Test Reliability:** Added comprehensive offline unit tests for `SearchController` (baseline preservation, reload failure rejection, and retention enforcement) and convenience method `BaseModelAdapter.count_parameters()`.
- **Documentation Accuracy:** Clarified sample sizes and distinguished empirical MLP channel slicing from Block Influence layer pruning in the README and documentation.

## [0.1.0] - 2026-11-09

### Added
- **Core Architecture:** Modular framework decoupling model architectures, benchmark tasks, pruning algorithms, search controllers, and reporting.
- **Model Adapter:** `Qwen2Adapter` supporting `Qwen2ForCausalLM` (`Qwen/Qwen2.5-Math-1.5B-Instruct`) with layer extraction, ablation hooks, and physical block slicing.
- **Benchmark Adapter:** `GSM8KBenchmark` supporting zero-shot Chain-of-Thought (CoT), official Qwen prompt templates, greedy generation, and Hugging Face `math-verify` answer extraction.
- **Pruning Engine:** `LayerPruner` implementing ShortGPT Block Influence (BI) layer redundancy scoring via calibration hidden-state cosine distance.
- **Search Controller:** Greedy benchmark-guided iterative search engine with accuracy retention thresholding, candidate rollback, and stopping conditions.
- **Export & Verification:** Safetensors model serialization with fresh-process standalone reload verification.
- **Experiment Tracking:** Transactional SQLite database (`experiments.db`) and JSONL generation trace logging.
- **Reporting:** Publication-grade Matplotlib plots (accuracy vs params, pruning %, throughput, VRAM, and Pareto frontier) and markdown reports.
- **Unified CLI:** Typer/Rich command suite exposing `doctor`, `inspect`, `benchmarks`, `baseline`, `run`, `resume`, `report`, and `export`.
- **Python API:** Unified `PruneQuest` class for library use.
- **Testing Suite:** Comprehensive unit tests on synthetic mini-models, mock search controllers, and benchmark parsers.
