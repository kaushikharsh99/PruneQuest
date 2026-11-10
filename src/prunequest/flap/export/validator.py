"""
Subprocess reload validator verifying exported checkpoints in an isolated process.
"""

from typing import Tuple, Optional
import os
import sys
import subprocess


class CheckpointValidator:
    """
    Spawns an isolated Python interpreter on CPU to ensure zero-VRAM contention
    while validating serialization, configuration, tensor shapes, and text generation.
    """

    @classmethod
    def validate(
        cls,
        checkpoint_dir: str,
        timeout_sec: int = 120,
        max_new_tokens: int = 5,
        prompt: str = "The sum of 10 and 20 is",
        device: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bool, str]:
        if not os.path.exists(checkpoint_dir):
            return False, f"Directory does not exist: {checkpoint_dir}"

        verification_script = f"""
import sys
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = {repr(checkpoint_dir)}
    prompt = {repr(prompt)}
    max_tokens = {int(max_new_tokens)}
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cpu'
    )
    inputs = tokenizer(prompt, return_tensors='pt').to('cpu')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"VALIDATION_OK|{{params}}|{{text}}")
except Exception as e:
    print(f"VALIDATION_FAIL|{{str(e)}}", file=sys.stderr)
    sys.exit(1)
"""
        env = dict(os.environ)
        env["MKL_THREADING_LAYER"] = "GNU"
        env["MKL_SERVICE_FORCE_INTEL"] = "1"

        try:
            res = subprocess.run(
                [sys.executable, "-c", verification_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if res.returncode == 0 and "VALIDATION_OK" in res.stdout:
                return True, res.stdout.strip()
            return False, res.stderr.strip() or res.stdout.strip()
        except Exception as e:
            return False, f"Subprocess validator error: {e}"
