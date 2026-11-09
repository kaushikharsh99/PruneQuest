"""
Benchmark registry and loader. Supports native benchmarks, custom BenchmarkSpec files,
and dynamic LM Evaluation Harness tasks.
"""

import os
from typing import Dict, Type, Any, List
from prunequest.benchmarks.base import BaseBenchmark
from prunequest.benchmarks.gsm8k import GSM8KBenchmark
from prunequest.benchmarks.spec import BenchmarkSpec, NativeBenchmark
from prunequest.benchmarks.lm_eval import LMEvalBenchmarkAdapter, LM_EVAL_AVAILABLE
from prunequest.exceptions import BenchmarkError


class BenchmarkRegistry:
    """
    Unified registry for evaluation benchmarks.
    """

    _native_registry: Dict[str, Type[BaseBenchmark]] = {
        "gsm8k": GSM8KBenchmark,
    }

    @classmethod
    def register(cls, name: str, benchmark_cls: Type[BaseBenchmark]):
        cls._native_registry[name.lower()] = benchmark_cls

    @classmethod
    def get_benchmark(
        cls,
        name: str,
        seed: int = 42,
        max_new_tokens: int = 512,
        num_fewshot: int = 0,
        batch_size: int = 1,
    ) -> BaseBenchmark:
        lower_name = name.lower().strip()

        # 1. Custom YAML specification
        if name.endswith(".yaml") or name.endswith(".yml") or os.path.exists(name):
            spec = BenchmarkSpec.from_yaml(name)
            return NativeBenchmark(spec, seed=seed)

        # 2. Native registered benchmark (e.g., gsm8k)
        if lower_name in cls._native_registry:
            b_cls = cls._native_registry[lower_name]
            return b_cls(seed=seed, max_new_tokens=max_new_tokens)

        # 3. Dynamic LM Evaluation Harness task
        if LM_EVAL_AVAILABLE:
            try:
                adapter = LMEvalBenchmarkAdapter(
                    task_name=name,
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    seed=seed,
                )
                return adapter
            except Exception as e:
                raise BenchmarkError(f"Failed to initialize LM-Eval task '{name}': {e}") from e

        supported = ", ".join(cls._native_registry.keys())
        raise BenchmarkError(
            f"Unsupported benchmark '{name}'. Native benchmarks: {supported}. "
            "Ensure lm-evaluation-harness is installed for additional tasks."
        )

    @classmethod
    def list_benchmarks(cls) -> Dict[str, Dict[str, Any]]:
        results = {}
        # Native benchmarks
        for k, v in cls._native_registry.items():
            results[k] = {
                "source": "native",
                "metric_name": getattr(v, "metric_name", "accuracy"),
                "metric_direction": getattr(v, "metric_direction", "higher"),
                "requires_generation": getattr(v, "requires_generation", True),
            }

        # LM-Eval popular tasks
        if LM_EVAL_AVAILABLE:
            for task in ["arc_challenge", "arc_easy", "hellaswag", "mmlu", "truthfulqa_mc2", "winogrande"]:
                results[task] = {
                    "source": "lm_evaluation_harness",
                    "metric_name": "acc,none",
                    "metric_direction": "higher",
                    "requires_generation": False,
                }

        return results

    @classmethod
    def search_benchmarks(cls, query: str) -> List[str]:
        q = query.lower()
        native_matches = [k for k in cls._native_registry if q in k]
        harness_matches = []
        if LM_EVAL_AVAILABLE:
            harness_matches = LMEvalBenchmarkAdapter.list_available_tasks(search_term=q)
        return sorted(list(set(native_matches + harness_matches)))
