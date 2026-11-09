# Compatibility & Architectural Constraints

This document details hardware, model, and upstream library compatibility constraints for PruneQuest v0.1.0.

---

## 1. Supported Models & Architectures

| Model Architecture | Tested Checkpoint | Status | Notes |
| :--- | :--- | :--- | :--- |
| `Qwen2ForCausalLM` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | **Tier 1 Supported** | Full physical layer pruning and GSM8K integration verified. |
| `LlamaForCausalLM` | Llama-2 / Llama-3 series | Planned (v0.2.0) | Requires `LlamaAdapter` implementation. |
| `MistralForCausalLM` | Mistral-7B | Planned (v0.2.0) | Requires sliding-window cache handling. |

---

## 2. Upstream Hugging Face Architectural Blockers

### Qwen2 Attention Head Coupling
In Hugging Face `transformers` implementation of `Qwen2Attention`:
```python
self.head_dim = self.hidden_size // self.num_heads
```
Notice that `head_dim` is dynamically computed from `hidden_size // num_attention_heads`.
If `num_attention_heads` is reduced from 12 to 10 while preserving `hidden_size = 1536`:
- `head_dim` changes from $1536 / 12 = 128$ to $1536 / 10 = 153.6$ (non-integer, breaking rotary embedding initialization).
- RoPE rotary position embeddings are precomputed for `dim = 128`. Changing `head_dim` causes shape mismatch in `apply_rotary_pos_emb`.

**Conclusion for v0.1.0:** Physical layer pruning and MLP intermediate dimension slicing are fully supported. Pure attention head deletion requires custom model modeling code or hidden size projection adjustments, targeted for v0.2.0.

### Strict Hugging Face Dataclass Validation
In `transformers>=5.14.0`, `config.layer_types` is enforced as a list of strings whose length must equal `config.num_hidden_layers`.
PruneQuest's `Qwen2Adapter` automatically keeps `layer_types` in sync with `num_hidden_layers` upon layer deletion.

---

## 3. Hardware Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **GPU** | NVIDIA RTX 3050 (4–6 GB VRAM) | NVIDIA RTX 4090 (24 GB) or A100 |
| **CUDA** | 12.0+ | 12.4+ |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 20 GB free disk space | 100 GB NVMe |
| **OS** | Linux (Fedora, Ubuntu, RHEL, Debian) | Ubuntu 22.04 LTS / Fedora 40 |
