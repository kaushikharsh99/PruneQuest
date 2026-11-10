"""
Independent subprocess validation for exported checkpoints.
Validates uncorrupted serialization, configuration, tokenizer, finite tensors,
forward pass, and text generation in an isolated Python interpreter.
"""

import os
import sys
import subprocess
from typing import Tuple


class SubprocessValidator:
    """
    Validates exported checkpoints by launching an isolated Python subprocess
    to verify uncorrupted deserialization, weight integrity, and text generation.
    """

    @classmethod
    def _run_validation_script(cls, checkpoint_dir: str, device: str, timeout_sec: int) -> Tuple[bool, str]:
        verification_script = f"""
import os, sys, json
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    path = {repr(checkpoint_dir)}
    dev = '{device}'

    # 1. Config Validation
    config = AutoConfig.from_pretrained(path)

    # 2. Tokenizer Validation
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 3. Model Weight Deserialization
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=dev,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # 4. Check for Finite Weights (No NaN or Inf)
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise ValueError(f"Weight tensor '{{name}}' contains NaN or Inf values")

    # 5. Forward Pass Logits Check
    prompt = "The sum of 10 and 20 is"
    inputs = tokenizer(prompt, return_tensors='pt').to(dev)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if not torch.isfinite(logits).all():
            raise ValueError("Forward pass produced non-finite logits (NaN or Inf)")

    # 6. Response Generation Check
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if not text:
        raise ValueError("Model generated empty response")

    # 7. Check Optional PruneQuest Metadata
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

    @classmethod
    def validate_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "cpu",
        timeout_sec: int = 120,
    ) -> Tuple[bool, str]:
        if not os.path.exists(checkpoint_dir):
            return False, f"Directory does not exist: {checkpoint_dir}"

        ok, msg = cls._run_validation_script(checkpoint_dir, device, timeout_sec)
        # If GPU validation encountered CUDA out of memory due to process contention, fallback to CPU
        if not ok and ("out of memory" in msg.lower() or "cuda" in msg.lower()) and device != "cpu":
            fallback_ok, fallback_msg = cls._run_validation_script(checkpoint_dir, "cpu", timeout_sec)
            if fallback_ok:
                return True, f"{fallback_msg} (validated on CPU fallback due to GPU memory contention)"
            return False, fallback_msg

        return ok, msg
