"""
Mistral model adapter for Mistral-7B, Zephyr, and related families.
"""

from prunequest.models.decoder import DecoderStackAdapter


class MistralAdapter(DecoderStackAdapter):
    """
    Adapter for Mistral architecture family.
    Validates and executes structural operations on MistralForCausalLM.
    """

    def __init__(self, model_id_or_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        super().__init__(
            model_id_or_path=model_id_or_path,
            device=device,
            torch_dtype=torch_dtype,
            family_name="MistralForCausalLM",
        )
