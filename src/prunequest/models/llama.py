"""
Llama model adapter for Llama-2, Llama-3, CodeLlama, and Yi families.
"""

from prunequest.models.decoder import DecoderStackAdapter


class LlamaAdapter(DecoderStackAdapter):
    """
    Adapter for Llama architecture family.
    Validates and executes structural operations on LlamaForCausalLM.
    """

    def __init__(self, model_id_or_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        super().__init__(
            model_id_or_path=model_id_or_path,
            device=device,
            torch_dtype=torch_dtype,
            family_name="LlamaForCausalLM",
        )
