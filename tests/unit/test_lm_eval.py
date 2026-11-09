from unittest.mock import MagicMock, patch
from prunequest.benchmarks.lm_eval import LMEvalBenchmarkAdapter
from prunequest.benchmarks.base import EvaluationResult


def test_lm_eval_list_tasks():
    with patch("prunequest.benchmarks.lm_eval.LM_EVAL_AVAILABLE", True):
        with patch("prunequest.benchmarks.lm_eval.TaskManager") as mock_tm:
            mock_tm.return_value.all_tasks = ["gsm8k", "arc_easy", "arc_challenge", "hellaswag"]
            tasks = LMEvalBenchmarkAdapter.list_available_tasks()
            assert len(tasks) == 4
            filtered = LMEvalBenchmarkAdapter.list_available_tasks("arc")
            assert len(filtered) == 2
            assert "arc_easy" in filtered


def test_lm_eval_get_task_info():
    with patch("prunequest.benchmarks.lm_eval.LM_EVAL_AVAILABLE", True):
        with patch("prunequest.benchmarks.lm_eval.TaskManager") as mock_tm:
            mock_task = MagicMock()
            mock_task.config = {
                "dataset_path": "gsm8k",
                "output_type": "generate_until",
                "metric_list": [{"metric": "exact_match"}],
                "num_fewshot": 0,
            }
            mock_tm.return_value.load_task_or_group.return_value = {"gsm8k": mock_task}

            info = LMEvalBenchmarkAdapter.get_task_info("gsm8k")
            assert info["task_name"] == "gsm8k"
            assert info["dataset_path"] == "gsm8k"
            assert "exact_match" in info["metric_list"]


def test_lm_eval_evaluate():
    with patch("prunequest.benchmarks.lm_eval.LM_EVAL_AVAILABLE", True):
        with patch("prunequest.benchmarks.lm_eval.HFLM"):
            with patch("prunequest.benchmarks.lm_eval.simple_evaluate") as mock_eval:
                mock_eval.return_value = {
                    "results": {
                        "arc_challenge": {
                            "acc,none": 0.75,
                        }
                    },
                    "samples": {
                        "arc_challenge": [
                            {"arguments": "Q1", "target": "A", "acc": True},
                            {"arguments": "Q2", "target": "B", "acc": False},
                        ]
                    }
                }
                adapter = LMEvalBenchmarkAdapter(task_name="arc_challenge")
                dummy_model = MagicMock()
                dummy_tok = MagicMock()

                result = adapter.evaluate(dummy_model, dummy_tok, max_samples=2, device="cpu")
                assert isinstance(result, EvaluationResult)
                assert result.total == 2
                assert result.accuracy == 0.75
                assert len(result.records) == 2
