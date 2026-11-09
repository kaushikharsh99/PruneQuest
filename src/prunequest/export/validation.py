"""
Independent subprocess validation for exported checkpoints.
"""

import os
import sys
import subprocess
from typing import Tuple


class SubprocessValidator:
    """
    Validates exported checkpoints by launching an isolated Python subprocess
    to verify uncorrupted deserialization and text generation.
    """

    @classmethod
    def validate_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "cuda:0",
        timeout_sec: int = 120,
    ) -> Tuple[bool, str]:
        if not os.path.exists(checkpoint_dir):
            return False, f"Directory does not exist: {checkpoint_dir}"

        verification_script = f"""
import os, sys
try:
    import numpy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = {repr(checkpoint_dir)}
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='{device}')
    inputs = tokenizer('The sum of 10 and 20 is', return_tensors='pt').to('{device}')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
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
