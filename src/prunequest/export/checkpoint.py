"""
Checkpoint serialization and metadata management.
"""

import os
import json
from typing import Dict, Any, Optional
from prunequest.models.base import BaseModelAdapter


class CheckpointExporter:
    """
    Exports pruned checkpoints to safetensors format with complete reproducibility metadata.
    """

    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)

    def export(
        self,
        adapter: BaseModelAdapter,
        checkpoint_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        target_dir = os.path.join(self.checkpoints_dir, checkpoint_name)
        os.makedirs(target_dir, exist_ok=True)

        adapter.save_checkpoint(target_dir)

        model_meta = adapter.get_metadata()
        meta_dict = {
            "checkpoint_name": checkpoint_name,
            "architecture": model_meta.architecture,
            "total_params": model_meta.total_params,
            "num_layers": model_meta.num_layers,
            "dtype": model_meta.dtype,
            "pruning_metadata": metadata or {},
        }

        meta_file = os.path.join(target_dir, "prunequest_metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        return target_dir
