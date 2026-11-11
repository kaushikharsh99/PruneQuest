"""
Universal Dataset Source Resolver for PruneQuest.
Discovers and inspects dataset source types with bounded memory consumption.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from prunequest.exceptions import BenchmarkError
from prunequest.datasets.models import DatasetSourceType, ResolvedDatasetSource


class DatasetSourceResolver:
    """
    Resolves local dataset files, directories, Hugging Face repos, and registered benchmarks.
    """

    @classmethod
    def resolve(
        cls,
        source: str,
        config: Optional[str] = None,
        revision: Optional[str] = "main",
        split: Optional[str] = None,
    ) -> ResolvedDatasetSource:
        """
        Determines the source type and validates existence.
        """
        clean_source = source.strip()

        # 1. Registered benchmark name
        from prunequest.benchmarks.registry import BenchmarkRegistry
        if clean_source.lower() in BenchmarkRegistry._native_registry:
            return ResolvedDatasetSource(
                source_type=DatasetSourceType.REGISTERED_BENCHMARK,
                path_or_id=clean_source.lower(),
                is_local=False,
                split=split,
            )

        # 2. LM Evaluation Harness task prefix
        if clean_source.startswith("lm_eval:") or clean_source.startswith("lm-eval:"):
            task_name = clean_source.split(":", 1)[1]
            return ResolvedDatasetSource(
                source_type=DatasetSourceType.LM_EVAL,
                path_or_id=task_name,
                is_local=False,
                split=split,
            )

        # 3. Local file or directory
        expanded_path = os.path.expanduser(clean_source)
        if os.path.exists(expanded_path):
            if os.path.isfile(expanded_path):
                ext = os.path.splitext(expanded_path)[1].lower()
                if ext in (".yaml", ".yml"):
                    return ResolvedDatasetSource(
                        source_type=DatasetSourceType.BENCHMARK_SPEC,
                        path_or_id=expanded_path,
                        is_local=True,
                        file_format="yaml",
                        split=split,
                    )
                elif ext == ".jsonl":
                    return ResolvedDatasetSource(
                        source_type=DatasetSourceType.LOCAL_FILE,
                        path_or_id=expanded_path,
                        is_local=True,
                        file_format="jsonl",
                        split=split,
                    )
                elif ext == ".json":
                    return ResolvedDatasetSource(
                        source_type=DatasetSourceType.LOCAL_FILE,
                        path_or_id=expanded_path,
                        is_local=True,
                        file_format="json",
                        split=split,
                    )
                elif ext == ".csv":
                    return ResolvedDatasetSource(
                        source_type=DatasetSourceType.LOCAL_FILE,
                        path_or_id=expanded_path,
                        is_local=True,
                        file_format="csv",
                        split=split,
                    )
                elif ext == ".parquet":
                    return ResolvedDatasetSource(
                        source_type=DatasetSourceType.LOCAL_FILE,
                        path_or_id=expanded_path,
                        is_local=True,
                        file_format="parquet",
                        split=split,
                    )
                else:
                    raise BenchmarkError(
                        f"Unsupported file format '{ext}' for local dataset: {clean_source}. "
                        f"Supported formats: .jsonl, .json, .csv, .parquet, .yaml"
                    )
            elif os.path.isdir(expanded_path):
                # Search for supported split files in directory
                candidates = os.listdir(expanded_path)
                supported = [c for c in candidates if c.endswith((".jsonl", ".json", ".csv", ".parquet"))]
                if not supported:
                    raise BenchmarkError(
                        f"Local directory '{clean_source}' does not contain supported dataset files (.jsonl, .json, .csv, .parquet)."
                    )
                return ResolvedDatasetSource(
                    source_type=DatasetSourceType.LOCAL_FILE,
                    path_or_id=expanded_path,
                    is_local=True,
                    file_format="directory",
                    split=split,
                )

        # 4. If path looks like a non-existent local file (has slash or ends with dataset extension)
        if clean_source.startswith(("./", "../", "/")) or any(clean_source.endswith(ext) for ext in (".json", ".jsonl", ".csv", ".parquet", ".yaml", ".yml")):
            raise BenchmarkError(f"Local dataset file not found: {clean_source}")

        # 5. Hugging Face Datasets repository identifier (e.g. 'openai/gsm8k', 'glue', 'c4')
        return ResolvedDatasetSource(
            source_type=DatasetSourceType.HUGGINGFACE,
            path_or_id=clean_source,
            is_local=False,
            file_format="huggingface",
            config_name=config,
            revision=revision or "main",
            split=split,
        )

    @classmethod
    def read_preview_records(
        cls,
        resolved: ResolvedDatasetSource,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], Optional[int], List[str]]:
        """
        Reads a bounded sample of records, total sample count, and detected splits without loading full weights/datasets.
        Returns: (sample_rows, total_count_or_none, splits)
        """
        if resolved.source_type == DatasetSourceType.LOCAL_FILE:
            return cls._read_local_records(resolved.path_or_id, resolved.file_format, limit=limit)
        elif resolved.source_type == DatasetSourceType.BENCHMARK_SPEC:
            from prunequest.benchmarks.spec import BenchmarkSpec
            spec = BenchmarkSpec.from_yaml(resolved.path_or_id)
            inner_resolved = cls.resolve(spec.dataset.path)
            return cls.read_preview_records(inner_resolved, limit=limit)
        elif resolved.source_type == DatasetSourceType.HUGGINGFACE:
            return cls._read_huggingface_records(resolved, limit=limit)
        elif resolved.source_type == DatasetSourceType.REGISTERED_BENCHMARK:
            from prunequest.benchmarks.registry import BenchmarkRegistry
            bench = BenchmarkRegistry.get_benchmark(resolved.path_or_id)
            data = bench.load_data(calibration_size=10, dev_size=limit)
            dev_samples = data.get("dev", [])
            records = [{"question": s.question, "answer": s.gold_answer} for s in dev_samples[:limit]]
            return records, len(dev_samples), list(data.keys())
        elif resolved.source_type == DatasetSourceType.LM_EVAL:
            return [{"task": resolved.path_or_id}], None, ["eval"]

        return [], None, []

    @classmethod
    def _read_local_records(
        cls,
        path: str,
        file_format: Optional[str],
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], Optional[int], List[str]]:
        if file_format == "jsonl":
            records = []
            total = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s_line = line.strip()
                    if s_line:
                        total += 1
                        if len(records) < limit:
                            records.append(json.loads(s_line))
            return records, total, ["all"]
        elif file_format == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data[:limit], len(data), ["all"]
            elif isinstance(data, dict):
                # Could be split-separated: {"train": [...], "test": [...]}
                splits = list(data.keys())
                for sp in ("train", "test", "validation", "dev"):
                    if sp in data and isinstance(data[sp], list):
                        return data[sp][:limit], len(data[sp]), splits
                # Default to first list
                for k, v in data.items():
                    if isinstance(v, list):
                        return v[:limit], len(v), splits
            raise BenchmarkError(f"Unsupported JSON structure in {path}. Expected list of records.")
        elif file_format == "csv":
            df_preview = pd.read_csv(path, nrows=limit)
            # Count total rows safely
            with open(path, "rb") as f:
                total_lines = sum(1 for _ in f) - 1
            return df_preview.to_dict(orient="records"), max(total_lines, len(df_preview)), ["all"]
        elif file_format == "parquet":
            df = pd.read_parquet(path)
            return df.head(limit).to_dict(orient="records"), len(df), ["all"]
        elif file_format == "directory":
            # Pick first file
            for f in sorted(os.listdir(path)):
                if f.endswith((".jsonl", ".json", ".csv", ".parquet")):
                    sub_fmt = f.split(".")[-1]
                    return cls._read_local_records(os.path.join(path, f), sub_fmt, limit=limit)

        return [], None, []

    @classmethod
    def _read_huggingface_records(
        cls,
        resolved: ResolvedDatasetSource,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], Optional[int], List[str]]:
        from datasets import load_dataset_builder, load_dataset
        try:
            builder = load_dataset_builder(resolved.path_or_id, resolved.config_name, revision=resolved.revision)
            splits = list(builder.info.splits.keys()) if builder.info.splits else []
            # Prefer test, then validation, then train for preview
            target_split = resolved.split
            if not target_split:
                for cand in ("test", "validation", "dev", "train"):
                    if cand in splits:
                        target_split = cand
                        break
                target_split = target_split or (splits[0] if splits else "train")

            # Stream small slice without downloading whole dataset
            stream_ds = load_dataset(
                resolved.path_or_id,
                resolved.config_name,
                revision=resolved.revision,
                split=target_split,
                streaming=True,
            )
            records = []
            for item in stream_ds.take(limit):
                records.append(item)

            total_count = None
            if builder.info.splits and target_split in builder.info.splits:
                total_count = builder.info.splits[target_split].num_examples

            return records, total_count, splits
        except Exception as e:
            # Fallback to standard load if streaming not supported
            try:
                ds = load_dataset(resolved.path_or_id, resolved.config_name, revision=resolved.revision)
                splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
                chosen_split = resolved.split or ("test" if "test" in splits else splits[0])
                sub_ds = ds[chosen_split]
                sample_records = [sub_ds[i] for i in range(min(limit, len(sub_ds)))]
                return sample_records, len(sub_ds), splits
            except Exception as inner_e:
                raise BenchmarkError(f"Failed to inspect Hugging Face dataset '{resolved.path_or_id}': {inner_e}") from e

    @classmethod
    def compute_source_hash(cls, resolved: ResolvedDatasetSource) -> str:
        """
        Computes SHA256 of local file or hash of HF ID/revision.
        """
        if resolved.is_local and os.path.isfile(resolved.path_or_id):
            h = hashlib.sha256()
            with open(resolved.path_or_id, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        else:
            meta = f"{resolved.path_or_id}:{resolved.config_name}:{resolved.revision}"
            return hashlib.sha256(meta.encode("utf-8")).hexdigest()
