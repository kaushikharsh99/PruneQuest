"""
Safetensors model checkpoint exporter with complete provenance metadata.
"""

from typing import Dict, Any
import os
import json
import gc
from transformers import PreTrainedModel, PreTrainedTokenizer


class CheckpointExporter:
    """
    Exports physical model weights in safetensors format and records provenance.
    """

    @classmethod
    def export(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
        checkpoint_name: str,
        metadata: Dict[str, Any],
    ) -> str:
        target_dir = os.path.join(output_dir, "checkpoints", checkpoint_name)
        os.makedirs(target_dir, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(target_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(target_dir)

        # Write provenance metadata
        meta_path = os.path.join(target_dir, "provenance.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        gc.collect()
        return target_dir
