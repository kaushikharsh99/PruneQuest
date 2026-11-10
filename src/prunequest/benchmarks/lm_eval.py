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
    simple_evaluate = None
    TaskManager = None
    HFLM = None


class LMEvalBenchmarkAdapter(BaseBenchmark):
    """
    Adapter bridging EleutherAI LM Evaluation Harness to PruneQuest's BaseBenchmark interface.
    Provides robust metric extraction, explicit sample accounting, and directional constraint support.
    """

    HIGHER_IS_BETTER_METRICS = {
        "acc", "acc_norm", "exact_match", "f1", "pass@1", "bleu", "rouge", "matthews_correlation", "mcc",
    }
    LOWER_IS_BETTER_METRICS = {
        "word_perplexity", "byte_perplexity", "bits_per_byte", "perplexity", "loss", "loglikelihood",
    }

    def __init__(
        self,
        task_name: str,
        primary_metric: Optional[str] = None,
        num_fewshot: int = 0,
        batch_size: int = 1,
        seed: int = 42,
    ):
        super().__init__(name=task_name, seed=seed)
        if not LM_EVAL_AVAILABLE:
            raise BenchmarkError(
                "lm-evaluation-harness is not installed. Install via `pip install 'prunequest[eval]'`."
            )
        self.task_name = task_name
        self.primary_metric = primary_metric
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self._task_manager = None
        self.metric_name = primary_metric or "accuracy"
        self.metric_direction = "higher"

    @property
    def task_manager(self) -> Any:
        if not LM_EVAL_AVAILABLE or TaskManager is None:
            raise BenchmarkError(
                "lm-evaluation-harness is not installed. Install via `pip install 'prunequest[eval]'`."
            )
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
        """
        LM Evaluation Harness is an evaluation engine and does not provide training/calibration samples.
        To calibrate pruning on an LM-Eval task, a compatible dataset must be supplied.
        """
        raise BenchmarkError(
            f"LM Evaluation Harness task '{self.task_name}' does not provide automatic calibration data. "
            "Pruning requires valid calibration prompts: please configure a separate calibration dataset "
            "(e.g. gsm8k, a YAML BenchmarkSpec, or a local jsonl/csv dataset) for importance profiling."
        )

    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        split_name: str = "smoke",
        max_samples: Optional[int] = 16,
        device: str = "cuda:0",
        progress_callback: Optional[Any] = None,
    ) -> EvaluationResult:
        if not LM_EVAL_AVAILABLE or HFLM is None or simple_evaluate is None:
            raise BenchmarkError(
                "lm-evaluation-harness is not installed. Install via `pip install 'prunequest[eval]'`."
            )

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
                log_samples=True,
            )
        except Exception as e:
            raise BenchmarkError(f"LM Evaluation Harness failed for task '{self.task_name}': {e}") from e

        total_time = time.time() - start_time
        peak_vram = get_gpu_vram_allocated_gb(device)

        task_res = eval_output.get("results", {}).get(self.task_name, {})
        if not task_res:
            # Check if aliased or nested
            for k, v in eval_output.get("results", {}).items():
                task_res = v
                break

        # 1. Resolve Primary Metric and Direction
        primary_score, resolved_metric, metric_dir = self._resolve_metric(task_res)
        self.metric_name = resolved_metric
        self.metric_direction = metric_dir

        # 2. Extract Samples and Per-Sample Records
        samples_dict = eval_output.get("samples", {}).get(self.task_name, [])
        if not samples_dict:
            for k, v in eval_output.get("samples", {}).items():
                samples_dict = v
                break

        sample_count = len(samples_dict) if samples_dict else (limit if limit is not None else 0)
        records = []
        correct_count = 0
        has_bool_evaluation = False

        for idx, sample in enumerate(samples_dict):
            prompt = str(sample.get("arguments", sample.get("prompt", "")))
            gold_target = str(sample.get("target", ""))
            resps = sample.get("resps", [])
            gen_text = str(resps[0] if (resps and isinstance(resps, list)) else "")

            # Check if sample level accuracy or exact_match is available
            is_correct = None
            if "acc" in sample:
                is_correct = bool(sample["acc"])
                has_bool_evaluation = True
                if is_correct:
                    correct_count += 1
            elif "exact_match" in sample:
                is_correct = bool(sample["exact_match"])
                has_bool_evaluation = True
                if is_correct:
                    correct_count += 1

            records.append({
                "sample_id": str(idx),
                "prompt": prompt,
                "gold_answer": gold_target,
                "generated_text": gen_text,
                "is_correct": is_correct,
            })

        # If individual boolean outcomes were not present in samples:
        if not has_bool_evaluation:
            if metric_dir == "higher" and any(k in resolved_metric for k in ("acc", "exact_match")) and sample_count > 0:
                correct_count = int(round(primary_score * sample_count))
            else:
                correct_count = 0  # Concept of correct answer count does not apply to perplexity or continuous scores

        clean_gpu_memory(device)

        return EvaluationResult(
            total=sample_count,
            correct=correct_count,
            accuracy=primary_score,
            records=records,
            total_tokens=0,
            total_time_sec=total_time,
            tokens_per_sec=0.0,
            peak_vram_gb=peak_vram,
        )

    def _resolve_metric(self, task_res: Dict[str, Any]) -> tuple[float, str, str]:
        """
        Determines the primary metric value, metric name, and direction from task results.
        """
        if self.primary_metric and self.primary_metric in task_res:
            name = self.primary_metric
            val = float(task_res[name])
            base_name = name.split(",")[0].lower()
            direction = "lower" if base_name in self.LOWER_IS_BETTER_METRICS else "higher"
            return val, name, direction

        # Priority higher-is-better metrics
        for candidate_key in [
            "acc,none", "acc_norm,none", "exact_match,none", "acc", "acc_norm", "exact_match", "f1,none", "pass@1,none",
        ]:
            if candidate_key in task_res:
                return float(task_res[candidate_key]), candidate_key, "higher"

        # Priority lower-is-better metrics (perplexity, loss)
        for candidate_key in [
            "word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none", "perplexity,none",
            "word_perplexity", "byte_perplexity", "bits_per_byte", "perplexity", "loss",
        ]:
            if candidate_key in task_res:
                return float(task_res[candidate_key]), candidate_key, "lower"

        # Fallback: find any numeric metric in results
        for k, v in task_res.items():
            if k not in ("alias", "samples", "sample_count") and isinstance(v, (int, float)):
                base_name = k.split(",")[0].lower()
                direction = "lower" if base_name in self.LOWER_IS_BETTER_METRICS else "higher"
                return float(v), k, direction

        if not task_res:
            raise BenchmarkError(f"No results returned for LM Evaluation Harness task '{self.task_name}'.")

        raise BenchmarkError(
            f"Could not determine primary metric for task '{self.task_name}' from keys: {list(task_res.keys())}. "
            "Please specify primary_metric explicitly."
        )

    @classmethod
    def list_available_tasks(cls, search_term: Optional[str] = None) -> List[str]:
        """Discover available LM Evaluation Harness tasks."""
        if not LM_EVAL_AVAILABLE or TaskManager is None:
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
        if not LM_EVAL_AVAILABLE or TaskManager is None:
            return {
                "task_name": task_name,
                "error": "lm-evaluation-harness is not installed. Install via `pip install 'prunequest[eval]'`.",
            }
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
