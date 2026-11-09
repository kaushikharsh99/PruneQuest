"""
Phi model adapter for Phi-2, Phi-3, and Phi-3.5 families.
"""

from prunequest.models.decoder import DecoderStackAdapter


class PhiAdapter(DecoderStackAdapter):
    """
    Adapter for Phi / Phi-3 architecture family.
    """

    def __init__(self, model_id_or_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        super().__init__(
            model_id_or_path=model_id_or_path,
            device=device,
            torch_dtype=torch_dtype,
            family_name="PhiForCausalLM",
        )
