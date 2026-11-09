"""
Qwen2 model adapter for Qwen2 and Qwen2.5 causal language models.
"""

from prunequest.models.decoder import DecoderStackAdapter


class Qwen2Adapter(DecoderStackAdapter):
    """
    Adapter for Qwen2 / Qwen2.5 architecture family.
    Validates and executes structural operations on Qwen2ForCausalLM.
    """

    def __init__(self, model_id_or_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        super().__init__(
            model_id_or_path=model_id_or_path,
            device=device,
            torch_dtype=torch_dtype,
            family_name="Qwen2ForCausalLM",
        )
