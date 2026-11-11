"""
BenchmarkSpec Synthesis from Onboarded Dataset Metadata.
Generates fully validated BenchmarkSpec objects for zero-YAML execution.
"""

import os
import yaml
from typing import Optional, Dict
from prunequest.benchmarks.spec import (
    BenchmarkSpec,
    DatasetConfig,
    DatasetFields,
    TaskConfig,
    GenerationConfig,
    ScoringConfig,
    OptimizationConfig,
    NativeBenchmark,
)


class BenchmarkSpecBuilder:
    """
    Builds and persists concrete BenchmarkSpec models from dataset analysis.
    """

    @classmethod
    def build(
        cls,
        name: str,
        source: str,
        path: str,
        input_col: str,
        target_col: str,
        task_type: str = "generation",
        scorer: str = "exact_match",
        higher_is_better: bool = True,
        retention: float = 0.99,
        prompt_template: str = "{{ input }}",
        use_chat_template: bool = False,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        revision: Optional[str] = "main",
        split_mapping: Optional[Dict[str, str]] = None,
    ) -> BenchmarkSpec:
        """
        Creates a validated BenchmarkSpec instance.
        """
        clean_name = name.replace("/", "_").replace(".", "_").replace("-", "_").lower()
        mapping = split_mapping or {
            "calibration": "train",
            "screening": "train",
            "dev": "validation",
            "smoke": "validation",
            "final_test": "test",
        }

        dataset_cfg = DatasetConfig(
            source=source,
            path=path,
            revision=revision or "main",
            fields=DatasetFields(input=input_col, target=target_col),
            split_mapping=mapping,
        )

        task_cfg = TaskConfig(
            type=task_type,
            prompt_template=prompt_template,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

        score_cfg = ScoringConfig(
            metric=scorer,
            higher_is_better=higher_is_better,
        )

        opt_cfg = OptimizationConfig(
            metric=scorer,
            retention=retention,
        )

        return BenchmarkSpec(
            version=1,
            name=clean_name,
            dataset=dataset_cfg,
            task=task_cfg,
            generation=gen_cfg,
            scoring=score_cfg,
            optimization=opt_cfg,
        )

    @classmethod
    def save_spec(cls, spec: BenchmarkSpec, output_path: str) -> str:
        """
        Dumps the spec to a YAML file for 100% deterministic reproduction.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec.model_dump(), f, sort_keys=False)
        return output_path

    @classmethod
    def create_benchmark(cls, spec: BenchmarkSpec, seed: int = 42) -> NativeBenchmark:
        """
        Constructs an executable NativeBenchmark from the synthesized spec.
        """
        return NativeBenchmark(spec=spec, seed=seed)
