# Architecture Compatibility & Capability Matrix: PruneQuest Release A

This document details model architecture compatibility, capability tiers, and hardware requirements for **PruneQuest Release A**.

---

## 1. Structured Capability Matrix

PruneQuest enforces honest capability reporting. Capability differs between architecture inspection, model loading, benchmark evaluation, Block Influence layer pruning, FLAP width pruning, joint optimization, and independent checkpoint export:

| Model Family | Architecture Class | Inspection | Baseline Eval | Block Influence (Depth) | FLAP (MLP Width) | JIPO Joint Sweep | Checkpoint Export & Reload | Overall Capability Level |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2 / 2.5** | `decoder_only_dense` | Supported | Supported | **Verified** | **Verified** | **Verified** | **Verified** | **Level 3 (Verified)** |
| **Llama 2 / 3 / 3.1 / 3.2** | `decoder_only_dense` | Supported | Supported | **Verified** | **Verified** | **Verified** | **Verified** | **Level 3 (Verified)** |
| **Mistral / Zephyr** | `decoder_only_dense` | Supported | Supported | **Verified** | **Verified** | **Verified** | **Verified** | **Level 3 (Verified)** |
| **Gemma / Gemma-2** | `decoder_only_dense` | Supported | Supported | *Experimental* | *Experimental* | *Unverified* | *Experimental* | **Level 2 (Experimental)** |
| **Phi-2 / Phi-3 / 3.5** | `decoder_only_dense` | Supported | Supported | *Experimental* | *Unsupported (Fused Proj)* | *Unverified* | *Experimental* | **Level 2 (Experimental)** |
| **Mixtral / Qwen-MoE** | `decoder_only_moe` | Supported | Supported | *Unsupported* | *Unsupported* | *Unsupported* | *Unsupported* | **Level 1 (Evaluation)** |
| **Encoder-Decoder (T5/BART)** | `encoder_decoder` | Supported | Supported | *Unsupported* | *Unsupported* | *Unsupported* | *Unsupported* | **Level 1 (Evaluation)** |
| **Quantized (AWQ/GPTQ/BitsAndBytes)** | `quantized` | Supported | Supported | *Unsupported* | *Unsupported* | *Unsupported* | *Unsupported* | **Level 1 (Evaluation)** |

### Capability Tiers Defined
- **Level 3 (Verified)**: End-to-end pruning, validation, export, reload, and benchmark evaluation pass automated regression test suites without custom loaders.
- **Level 2 (Experimental)**: Basic pruning mechanics execute, but edge cases (e.g. normalization weight scaling, sliding window caches, or rotary embedding adaptations) remain under active evaluation.
- **Level 1 (Evaluation Only)**: Architecture inspection and benchmark evaluation are supported; structural pruning operations are not supported.
- **Level 0 (Unsupported)**: Architecture cannot be parsed or executed by standard Hugging Face loaders.

---

## 2. Model Family Details & Architectural Notes

### Qwen 2 / 2.5
- **Adapters**: `prunequest.models.qwen2.Qwen2Adapter`, `prunequest.flap.models.registry`
- **Verified Operations**: Physical layer excision (`model.layers.pop()`), SwiGLU MLP width slicing (`gate_proj`, `up_proj`, `down_proj`), and JIPO joint optimization.
- **Strict HF Dataclass Synchronization**: Automatically keeps `config.layer_types` in sync with `config.num_hidden_layers` upon layer removal.

### LLaMA 2 / 3 / 3.1 / 3.2
- **Adapters**: `prunequest.models.llama.LlamaAdapter`
- **Verified Operations**: Physical layer removal, SwiGLU MLP channel slicing, standalone safetensors export.
- **RoPE / Cache Compatibility**: Position embeddings and attention cache dimensions dynamically reflect remaining layers.

### Mistral / Zephyr
- **Adapters**: `prunequest.models.mistral.MistralAdapter`
- **Verified Operations**: Physical layer removal, SwiGLU intermediate dimension reduction.
- **Sliding-Window Cache**: Preserves sliding-window attention configurations upon layer re-indexing.

### Gemma / Gemma-2
- **Adapters**: `prunequest.models.gemma.GemmaAdapter`
- **Status**: Level 2 Experimental. Layer removal is functional; post-norm layer scaling may require fine-tuning or recovery calibration.

### Phi-3 / Phi-3.5
- **Adapters**: `prunequest.models.phi.PhiAdapter`
- **Status**: Level 2 Experimental for layer excision. FLAP intermediate width slicing is currently disabled because Phi-3 fuses gate and up projections into a single `gate_up_proj` matrix.

---

## 3. Hardware & Environment Requirements

| Component | Minimum | Recommended | Notes |
| :--- | :--- | :--- | :--- |
| **GPU** | NVIDIA RTX 3050 (4–6 GB VRAM) | NVIDIA RTX 4090 (24 GB) or A100 | PruneQuest is actively tested on 6 GB VRAM laptops with automatic cache clearing and CPU validation offloading. |
| **CPU Mode** | Any 64-bit x86_64 or ARM64 CPU | Multi-core CPU | Supported for inspection, synthetic fixtures, benchmark verification, and reload validation. |
| **CUDA** | 12.0+ | 12.4+ | PyTorch 2.5.1+cu124 tested. |
| **System RAM** | 16 GB | 32 GB | High RAM is recommended when loading large models for CPU offload validation. |
| **Storage** | 20 GB free disk space | 100 GB NVMe | Checkpoints are saved atomically in Hugging Face `safetensors` format. |
| **Operating System** | Linux (Fedora, Ubuntu, RHEL, Debian) | Ubuntu 22.04 LTS / Fedora 40+ | Native Linux development environment. |
