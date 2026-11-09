"""
Integration adapter for EleutherAI LM Evaluation Harness (lm-eval).
Allows evaluating in-memory PyTorch models across thousands of standard benchmarks
without duplicating GPU memory.
"""

from typing import Dict, Any, List, Optional
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from prunequest.benchmarks.base import BaseBenchmark, BenchmarkSample, EvaluationResult
from prunequest.exceptions import BenchmarkError
from prunequest.reporting.metrics import get_gpu_vram_allocated_gb, clean_gpu_memory

try:
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False


class LMEvalBenchmarkAdapter(BaseBenchmark):
    """
    Adapter bridging EleutherAI LM Evaluation Harness to PruneQuest's BaseBenchmark interface.
    """

    def __init__(
        self,
        task_name: str,
        num_fewshot: int = 0,
        batch_size: int = 1,
        seed: int = 42,
    ):
        super().__init__(name=task_name, seed=seed)
        if not LM_EVAL_AVAILABLE:
            raise BenchmarkError(
                "lm-evaluation-harness is not installed. Install via `pip install lm-eval`."
            )
        self.task_name = task_name
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self._task_manager = None

    @property
    def task_manager(self) -> Any:
        if self._task_manager is None:
            self._task_manager = TaskManager()
        return self._task_manager

    def load_data(
        self,
        calibration_size: int = 256,
        dev_size: int = 512,
        smoke_size: int = 16,
        screening_size: int = 64,
    ) -> Dict[str, List[BenchmarkSample]]:
        # LM Eval handles its own data loading internally; we return lightweight metadata
        return {
            "calibration": [BenchmarkSample(id=f"sample_{i}", question="", gold_answer="") for i in range(smoke_size)],
            "smoke": [BenchmarkSample(id=f"smoke_{i}", question="", gold_answer="") for i in range(smoke_size)],
            "screening": [BenchmarkSample(id=f"screen_{i}", question="", gold_answer="") for i in range(screening_size)],
            "dev": [BenchmarkSample(id=f"dev_{i}", question="", gold_answer="") for i in range(dev_size)],
        }

    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        split_name: str = "smoke",
        max_samples: Optional[int] = 16,
        device: str = "cuda:0",
    ) -> EvaluationResult:
        clean_gpu_memory(device)
        start_time = time.time()

        # Wrap in-memory model in HFLM wrapper without duplicating memory
        hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            device=device if torch.cuda.is_available() else "cpu",
        )

        limit = max_samples if max_samples is not None else None

        try:
            eval_output = simple_evaluate(
                model=hflm,
                tasks=[self.task_name],
                num_fewshot=self.num_fewshot,
                limit=limit,
                task_manager=self.task_manager,
                random_seed=self.seed,
                numpy_random_seed=self.seed,
                torch_random_seed=self.seed,
            )
        except Exception as e:
            raise BenchmarkError(f"LM Evaluation Harness failed for task '{self.task_name}': {e}") from e

        total_time = time.time() - start_time
        peak_vram = get_gpu_vram_allocated_gb(device)

        task_res = eval_output.get("results", {}).get(self.task_name, {})
        if not task_res:
            # Check if aliased
            for k, v in eval_output.get("results", {}).items():
                task_res = v
                break

        # Extract primary metric
        primary_score = 0.0
        for candidate_key in [
            "acc,none", "acc_norm,none", "exact_match,none", "acc", "acc_norm", "exact_match", "f1,none", "pass@1,none",
        ]:
            if candidate_key in task_res:
                primary_score = float(task_res[candidate_key])
                break

        # Total sample count evaluated
        sample_count = limit if limit is not None else 100
        correct_count = int(round(primary_score * sample_count))

        # Format records for tracker
        records = []
        samples_dict = eval_output.get("samples", {}).get(self.task_name, [])
        for idx, sample in enumerate(samples_dict):
            records.append({
                "sample_id": str(idx),
                "prompt": str(sample.get("arguments", "")),
                "gold_answer": str(sample.get("target", "")),
                "generated_text": str(sample.get("resps", [""])[0] if sample.get("resps") else ""),
                "is_correct": bool(sample.get("acc", primary_score > 0.5)),
            })

        clean_gpu_memory(device)

        return EvaluationResult(
            total=sample_count,
            correct=correct_count,
            accuracy=primary_score,
            records=records,
            total_tokens=0,
            total_time_sec=total_time,
            tokens_per_sec=(sample_count / total_time) if total_time > 0 else 0.0,
            peak_vram_gb=peak_vram,
        )

    @classmethod
    def list_available_tasks(cls, search_term: Optional[str] = None) -> List[str]:
        """Discover available LM Evaluation Harness tasks."""
        if not LM_EVAL_AVAILABLE:
            return []
        tm = TaskManager()
        all_tasks = tm.all_tasks
        if search_term:
            s = search_term.lower()
            return [t for t in all_tasks if s in t.lower()]
        return sorted(all_tasks)

    @classmethod
    def get_task_info(cls, task_name: str) -> Dict[str, Any]:
        """Retrieve task configuration metadata from Harness."""
        if not LM_EVAL_AVAILABLE:
            return {"error": "lm-evaluation-harness not installed"}
        tm = TaskManager()
        try:
            task_dict = tm.load_task_or_group([task_name])
            if task_name in task_dict:
                task_obj = task_dict[task_name]
                cfg = getattr(task_obj, "config", {})
                if hasattr(cfg, "to_dict"):
                    cfg = cfg.to_dict()
                return {
                    "task_name": task_name,
                    "dataset_path": cfg.get("dataset_path", "unknown"),
                    "output_type": cfg.get("output_type", "unknown"),
                    "metric_list": [m.get("metric") for m in cfg.get("metric_list", []) if isinstance(m, dict)],
                    "num_fewshot": cfg.get("num_fewshot", 0),
                }
        except Exception as e:
            return {"task_name": task_name, "error": str(e)}
        return {"task_name": task_name, "status": "available"}
