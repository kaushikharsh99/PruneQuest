# PruneQuest v0.2.0 — Migration & Architecture Evolution Plan

## 1. Executive Summary

This document provides a comprehensive audit of the v0.1.0/v0.1.1 implementation and details the engineering transition to **PruneQuest v0.2.0 — Universal LLM Pruning Framework**.

The primary objective of v0.2.0 is to eliminate model-, benchmark-, and pruning-specific hardcoding, transforming PruneQuest into a modular framework capable of automated multi-benchmark structural compression across major decoder-only language model families.

---

## 2. Audit of Existing Components

| Component | File | Status in v0.1.x | Evolution in v0.2.0 |
| :--- | :--- | :--- | :--- |
| **Model Inspection** | `models/qwen2.py` | Qwen2-specific only; required downloading weights | Upgraded to fast remote/local metadata inspection (`AutoConfig`), multi-architecture classification (dense, MoE, multimodal, quantized), and 4-tier capability levels (0–3). |
| **Model Adapters** | `models/base.py`, `qwen2.py` | Working Qwen2 physical layer surgery | Added `LlamaAdapter`, `MistralAdapter`, `GemmaAdapter`, `PhiAdapter`, with common `DecoderStackAdapter` abstraction, typed `ArchitectureDescriptor`, and transactional `temporary_ablation()`. |
| **Benchmark Engine** | `benchmarks/gsm8k.py` | Hardcoded GSM8K with math-verify | Decoupled into `LMEvalBenchmarkAdapter` (harnessing 14,000+ tasks via EleutherAI LM Evaluation Harness) and native `BenchmarkSpec` (declarative YAML for custom datasets). |
| **Evaluation Splits** | `benchmarks/base.py` | Conflated smoke test and acceptance | Enforced 4-tier evaluation integrity: `calibration`, `screening`, `acceptance` (development set), and held-out `final_test`. |
| **Retention Constraints** | `core/controller.py` | Single scalar accuracy threshold | Upgraded to multi-benchmark `ConstraintSuite` supporting directional `MetricSpec` (higher-is-better, lower-is-better with degradation limits). |
| **Search Controller** | `core/controller.py` | Tied to GSM8K and Qwen2 | Fully decoupled: coordinates `BaseModelAdapter`, `EvaluationSuite`, `BasePruner`, and `ExperimentTracker`. |
| **Pruning Engine** | `pruning/layers.py` | Block Influence layer removal | Standardized `BasePruner` protocol in `PrunerRegistry` with extension points for MLP channel slicing and attention head pruning. |
| **Experiment Tracking** | `core/tracking.py` | SQLite + JSONL traces | Immutable baseline restoration on resume, transactional acceptance logging, reference-tracked disk cleanup. |
| **CLI & Packaging** | `cli.py`, `pyproject.toml` | Qwen/GSM8K CLI flags | Added `prunequest inspect`, `benchmarks list/search/info`, `benchmarks validate/preview`, multi-benchmark support in `baseline` and `run`. |

---

## 3. Reliability & Correctness Repairs Completed

1. **GitHub Actions CI & Linting**:
   - Standardized `[tool.ruff.lint]` configuration (`select = ["E", "F", "W"]`, `ignore = ["E501"]`). All 232 prior lint warnings resolved. Clean CI pass across Python 3.10, 3.11, 3.12, and 3.13.
2. **Benchmark Validity & Acceptance Criteria**:
   - Candidates are screened rapidly on candidate subsets, but can **only** be accepted after passing the retention threshold on the designated development benchmark.
3. **Resume Baseline Preservation**:
   - `SearchController.resume()` strictly queries `tracker.get_baseline()` from SQLite, preserving the original unpruned reference accuracy and parameter counts without reset.
4. **Transactional Checkpoint Acceptance**:
   - Subprocess reload validation failure (`reload_ok is False`) immediately triggers layer rollback, logs `status="rejected_reload_failed"`, and halts acceptance.
5. **Reference-Tracked Disk Retention**:
   - Active checkpoints and resumable states are protected against premature cleanup.
6. **Empirical Reporting Integrity**:
   - Documentation clearly delineates sample sizes (16-sample smoke vs 512-sample dev) and separates experimental MLP slicing findings from Block Influence layer pruning results.
