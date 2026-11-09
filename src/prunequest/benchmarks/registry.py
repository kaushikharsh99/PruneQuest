"""
Benchmark registry and loader.
"""

from typing import Dict, Type, Any
from prunequest.benchmarks.base import BaseBenchmark
from prunequest.benchmarks.gsm8k import GSM8KBenchmark
from prunequest.exceptions import BenchmarkError


class BenchmarkRegistry:
    """
    Registry for evaluation benchmarks.
    """

    _registry: Dict[str, Type[BaseBenchmark]] = {
        "gsm8k": GSM8KBenchmark,
    }

    @classmethod
    def register(cls, name: str, benchmark_cls: Type[BaseBenchmark]):
        cls._registry[name.lower()] = benchmark_cls

    @classmethod
    def get_benchmark(cls, name: str, seed: int = 42, max_new_tokens: int = 512) -> BaseBenchmark:
        b_cls = cls._registry.get(name.lower())
        if b_cls is None:
            supported = ", ".join(cls._registry.keys())
            raise BenchmarkError(f"Unsupported benchmark '{name}'. Supported benchmarks: {supported}.")
        return b_cls(seed=seed, max_new_tokens=max_new_tokens)

    @classmethod
    def list_benchmarks(cls) -> Dict[str, Dict[str, Any]]:
        results = {}
        for k, v in cls._registry.items():
            results[k] = {
                "metric_name": v.metric_name,
                "metric_direction": v.metric_direction,
                "requires_generation": v.requires_generation,
            }
        return results
