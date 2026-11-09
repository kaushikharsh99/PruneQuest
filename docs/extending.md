# Extending PruneQuest: Custom Models, Pruners, and Benchmarks

PruneQuest is architected with clear plugin boundaries, enabling developers to register custom models, search strategies, and evaluation benchmarks.

---

## 1. Adding a New Model Adapter

Inherit from `BaseModelAdapter` and register your class with `@ModelRegistry.register`:

```python
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from prunequest.models.base import BaseModelAdapter, ModelMetadata
from prunequest.models.registry import ModelRegistry


@ModelRegistry.register("llama")
class LlamaAdapter(BaseModelAdapter):

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self.torch_dtype),
            device_map=self.device,
        )

    def get_layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def prune_layer(self, layer_idx: int) -> int:
        layers = self.get_layers()
        removed_layer = layers[layer_idx]
        params_removed = sum(p.numel() for p in removed_layer.parameters())

        del layers[layer_idx]
        self.model.config.num_hidden_layers -= 1
        
        # Re-index remaining layers
        for idx, layer in enumerate(layers):
            layer.self_attn.layer_idx = idx

        return params_removed

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def validate_forward(self) -> Tuple[bool, Optional[str]]:
        try:
            inputs = self.tokenizer("Hello, world!", return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
            return True, None
        except Exception as e:
            return False, str(e)
```

---

## 2. Adding a New Benchmark

Inherit from `BaseBenchmark` and register your class with `@BenchmarkRegistry.register`:

```python
from typing import List
from prunequest.benchmarks.base import BaseBenchmark, BenchmarkSample, EvaluationResult
from prunequest.benchmarks.registry import BenchmarkRegistry
from prunequest.models.base import BaseModelAdapter


@BenchmarkRegistry.register("custom_math")
class CustomMathBenchmark(BaseBenchmark):

    def load_samples(self, split: str = "smoke", limit: int = 16) -> List[BenchmarkSample]:
        # Return a list of BenchmarkSample(id=..., question=..., expected_answer=...)
        pass

    def evaluate(self, adapter: BaseModelAdapter, split: str = "smoke", limit: int = 16) -> EvaluationResult:
        # Run inference and evaluate accuracy
        pass
```

---

## 3. Custom Search Strategies

Custom pruning search strategies (e.g., genetic algorithms, Bayesian optimization, or multi-objective Pareto exploration) can inherit from `BaseSearchStrategy`:

```python
from prunequest.core.search import BaseSearchStrategy, SearchState
from prunequest.models.base import BaseModelAdapter
from prunequest.benchmarks.base import BaseBenchmark
from prunequest.pruning.base import BasePruner


class BayesianSearchStrategy(BaseSearchStrategy):

    def search(
        self,
        adapter: BaseModelAdapter,
        benchmark: BaseBenchmark,
        pruner: BasePruner,
        max_iterations: int,
        retention_threshold: float,
    ) -> SearchState:
        # Implement custom search loop
        pass
```
