"""
Gemma model adapter for Gemma and Gemma-2 families.
"""

from prunequest.models.decoder import DecoderStackAdapter


class GemmaAdapter(DecoderStackAdapter):
    """
    Adapter for Gemma / Gemma-2 architecture family.
    """

    def __init__(self, model_id_or_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        super().__init__(
            model_id_or_path=model_id_or_path,
            device=device,
            torch_dtype=torch_dtype,
            family_name="GemmaForCausalLM",
        )
