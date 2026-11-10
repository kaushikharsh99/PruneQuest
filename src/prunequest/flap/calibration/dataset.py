"""
Calibration and evaluation dataset loading with strict disjoint partitioning.
"""

from typing import Dict, List, Any, Optional
import random
import os
import json
from dataclasses import dataclass
from datasets import load_dataset
from transformers import PreTrainedTokenizer


@dataclass
class DatasetSample:
    id: str
    question: str
    gold_answer: str
    full_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetManager:
    """
    Manages loading, formatting, and partitioning of datasets for FLAP calibration
    and multi-stage evaluation.
    """

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    def __init__(
        self,
        dataset_name: str = "openai/gsm8k",
        dataset_config: str = "main",
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seed = seed
        self.splits: Dict[str, List[DatasetSample]] = {}

    def load_and_partition(
        self,
        calibration_samples: int = 128,
        screening_samples: int = 64,
        acceptance_samples: int = 512,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dict[str, List[DatasetSample]]:
        """
        Loads the dataset and deterministically partitions into disjoint sets.
        """
        # Support local files (json, jsonl, csv, parquet)
        if os.path.exists(self.dataset_name):
            return self._load_local_file(
                self.dataset_name,
                calibration_samples,
                screening_samples,
                acceptance_samples,
                tokenizer,
            )

        # Standard Hugging Face datasets (e.g. openai/gsm8k)
        ds = load_dataset(self.dataset_name, self.dataset_config)
        train_data = list(ds["train"])
        test_data = list(ds["test"])

        rng = random.Random(self.seed)
        indices = list(range(len(train_data)))
        rng.shuffle(indices)

        idx_calib = indices[:calibration_samples]
        idx_screen = indices[calibration_samples : calibration_samples + screening_samples]
        idx_accept = indices[calibration_samples + screening_samples : calibration_samples + screening_samples + acceptance_samples]

        def format_sample(item: Dict[str, Any], split: str, idx: int) -> DatasetSample:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            full_text = None
            if tokenizer is not None:
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    full_text = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
            return DatasetSample(
                id=f"{split}_{idx}",
                question=q,
                gold_answer=a,
                full_text=full_text,
                metadata={"split": split, "original_index": idx},
            )

        self.splits = {
            "calibration": [format_sample(train_data[i], "calibration", i) for i in idx_calib],
            "screening": [format_sample(train_data[i], "screening", i) for i in idx_screen],
            "acceptance": [format_sample(train_data[i], "acceptance", i) for i in idx_accept],
            "test": [format_sample(test_data[i], "test", i) for i in range(len(test_data))],
        }
        return self.splits

    def _load_local_file(
        self,
        file_path: str,
        calibration_samples: int,
        screening_samples: int,
        acceptance_samples: int,
        tokenizer: Optional[PreTrainedTokenizer],
    ) -> Dict[str, List[DatasetSample]]:
        items = []
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                items = json.load(f)

        rng = random.Random(self.seed)
        indices = list(range(len(items)))
        rng.shuffle(indices)

        idx_calib = indices[:calibration_samples]
        idx_screen = indices[calibration_samples : calibration_samples + screening_samples]
        idx_accept = indices[calibration_samples + screening_samples : calibration_samples + screening_samples + acceptance_samples]
        idx_test = indices[calibration_samples + screening_samples + acceptance_samples :]

        def to_sample(item: Dict[str, Any], split: str, idx: int) -> DatasetSample:
            q = item.get("question", item.get("prompt", "")).strip()
            a = item.get("answer", item.get("target", "")).strip()
            return DatasetSample(
                id=f"{split}_{idx}",
                question=q,
                gold_answer=a,
                metadata={"split": split, "original_index": idx},
            )

        self.splits = {
            "calibration": [to_sample(items[i], "calibration", i) for i in idx_calib],
            "screening": [to_sample(items[i], "screening", i) for i in idx_screen],
            "acceptance": [to_sample(items[i], "acceptance", i) for i in idx_accept],
            "test": [to_sample(items[i], "test", i) for i in idx_test],
        }
        return self.splits
