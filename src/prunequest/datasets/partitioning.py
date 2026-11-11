"""
Deterministic, Disjoint Dataset Partitioning & Leakage Prevention for PruneQuest.
"""

import hashlib
import random
from typing import List, Dict, Any, Tuple
from prunequest.exceptions import BenchmarkError
from prunequest.benchmarks.base import BenchmarkSample
from prunequest.datasets.models import DatasetFingerprint, ResolvedDatasetSource


class DatasetPartitioner:
    """
    Enforces strict disjoint partitions between calibration, screening, and development evaluation.
    Protects official test splits from iterative pruning contamination.
    """

    @classmethod
    def partition_records(
        cls,
        resolved_source: ResolvedDatasetSource,
        raw_records: List[Dict[str, Any]],
        input_col: str,
        target_col: str,
        task_type: str,
        scorer: str,
        calibration_size: int = 128,
        screening_size: int = 64,
        dev_size: int = 512,
        seed: int = 42,
    ) -> Tuple[Dict[str, List[BenchmarkSample]], DatasetFingerprint]:
        """
        Partitions records into disjoint sets and computes dataset fingerprint.
        """
        # Validate required columns and filter malformed records
        valid_samples: List[BenchmarkSample] = []
        seen_hashes = set()
        duplicates_count = 0

        for idx, r in enumerate(raw_records):
            if input_col not in r or target_col not in r:
                raise BenchmarkError(
                    f"Sample at index {idx} is missing required field '{input_col}' or '{target_col}'."
                )
            val_in = str(r[input_col]).strip() if r[input_col] is not None else ""
            val_tgt = str(r[target_col]).strip() if r[target_col] is not None else ""
            if not val_in or not val_tgt:
                continue

            # Duplicate check
            norm_content = f"{val_in.lower()} ||| {val_tgt.lower()}"
            content_hash = hashlib.md5(norm_content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicates_count += 1
                continue
            seen_hashes.add(content_hash)

            sample_id = str(r.get("id", idx))
            valid_samples.append(BenchmarkSample(id=sample_id, question=val_in, gold_answer=val_tgt))

        total_valid = len(valid_samples)
        needed = calibration_size + screening_size + dev_size

        if total_valid < needed:
            raise BenchmarkError(
                f"Insufficient valid unique samples in dataset ({total_valid}) for disjoint partitions. "
                f"Required: {needed} (calibration={calibration_size} + screening={screening_size} + dev={dev_size}). "
                f"Data leakage between calibration and development is strictly prevented."
            )

        # Deterministic shuffle
        rng = random.Random(seed)
        shuffled = list(valid_samples)
        rng.shuffle(shuffled)

        calib_end = calibration_size
        screen_end = calib_end + screening_size
        dev_end = screen_end + dev_size

        partitions = {
            "calibration": shuffled[:calib_end],
            "screening": shuffled[calib_end:screen_end],
            "dev": shuffled[screen_end:dev_end],
            "smoke": shuffled[screen_end : screen_end + min(len(shuffled) - screen_end, 16)],
            "final_test": shuffled[dev_end:],
            "full": shuffled,
        }

        # Compute partition ID hashes for fingerprinting
        def hash_ids(s_list: List[BenchmarkSample]) -> str:
            concat = ",".join(s.id for s in s_list)
            return hashlib.sha256(concat.encode("utf-8")).hexdigest()

        partition_hashes = {
            "calibration": hash_ids(partitions["calibration"]),
            "screening": hash_ids(partitions["screening"]),
            "dev": hash_ids(partitions["dev"]),
            "final_test": hash_ids(partitions["final_test"]),
        }

        partition_counts = {k: len(v) for k, v in partitions.items() if k != "full"}

        # Source hash
        from prunequest.datasets.resolver import DatasetSourceResolver
        source_hash = DatasetSourceResolver.compute_source_hash(resolved_source)

        fingerprint = DatasetFingerprint(
            source_hash=source_hash,
            source_path_or_id=resolved_source.path_or_id,
            total_samples=total_valid,
            partition_counts=partition_counts,
            partition_hashes=partition_hashes,
            seed=seed,
            input_column=input_col,
            target_column=target_col,
            task_type=task_type,
            scorer=scorer,
        )

        return partitions, fingerprint

    @classmethod
    def verify_fingerprint_match(
        cls,
        existing_fp: Dict[str, Any],
        current_fp: DatasetFingerprint,
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that an experiment being resumed uses the exact same dataset provenance.
        """
        mismatches = []
        if existing_fp.get("source_hash") != current_fp.source_hash:
            mismatches.append(
                f"Dataset content hash mismatch: original={existing_fp.get('source_hash')[:12]}..., current={current_fp.source_hash[:12]}..."
            )
        if existing_fp.get("input_column") != current_fp.input_column:
            mismatches.append(f"Input column mismatch: original={existing_fp.get('input_column')}, current={current_fp.input_column}")
        if existing_fp.get("target_column") != current_fp.target_column:
            mismatches.append(f"Target column mismatch: original={existing_fp.get('target_column')}, current={current_fp.target_column}")
        if existing_fp.get("seed") != current_fp.seed:
            mismatches.append(f"Partition random seed mismatch: original={existing_fp.get('seed')}, current={current_fp.seed}")

        return len(mismatches) == 0, mismatches
