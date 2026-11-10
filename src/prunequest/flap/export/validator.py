"""
Subprocess reload validator verifying exported checkpoints in an isolated process.
Ensures zero-VRAM contention while validating configuration, tokenizer, finite tensors,
and generation.
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
import os, sys, json
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    path = {repr(checkpoint_dir)}
    prompt = {repr(prompt)}
    max_tokens = {int(max_new_tokens)}

    # 1. Config Validation
    config = AutoConfig.from_pretrained(path)

    # 2. Tokenizer Validation
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 3. Model Weight Deserialization
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    model.eval()

    # 4. Check for Finite Weights (No NaN or Inf)
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise ValueError(f"Weight tensor '{{name}}' contains NaN or Inf values")

    # 5. Generation Check
    inputs = tokenizer(prompt, return_tensors='pt').to('cpu')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if not text:
        raise ValueError("Model generated empty response")

    # 6. Check Metadata if present
    meta_path = os.path.join(path, "prunequest_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            _ = json.load(f)

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
            err_msg = res.stderr.strip() or res.stdout.strip()
            if not err_msg:
                if res.returncode < 0:
                    err_msg = f"Subprocess terminated by signal {-res.returncode} (process was killed, likely out-of-memory OOM)"
                else:
                    err_msg = f"Subprocess exited with code {res.returncode}"
            return False, err_msg
        except subprocess.TimeoutExpired:
            return False, f"Subprocess validation timed out after {timeout_sec}s"
        except Exception as e:
            return False, f"Subprocess validator error: {e}"
