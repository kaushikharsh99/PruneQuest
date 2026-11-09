# Changelog

All notable changes to PruneQuest will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
