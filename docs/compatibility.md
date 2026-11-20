# Model Compatibility

Compatibility depends on the operation. Successful configuration inspection does not establish that weights load, evaluation works, structural surgery is correct, or exported checkpoints reload. Treat each capability independently and confirm it against the current adapters and tests.

## Adapter inventory

The source currently contains adapters for Qwen2, Llama, Mistral, Gemma, and Phi under `src/prunequest/models/`, plus a separate FLAP model adapter registry under `src/prunequest/flap/models/`. Discovery and registry behavior are implemented in `models/discovery.py`, `models/registry.py`, and the corresponding FLAP modules.

| Family | Adapter present | Guidance |
| --- | --- | --- |
| Qwen2 | Yes | Check adapter, pruning, FLAP, and integration tests for the exact model/version and operation. |
| Llama | Yes | Check adapter and tests; model repository access may require authentication/acceptance. |
| Mistral | Yes | Check adapter and tests for the target configuration. |
| Gemma | Yes | Treat less-covered configurations as experimental until verified. |
| Phi | Yes | Projection layouts vary; inspect adapter support before attempting width slicing. |
| Other families | Discovery may classify some architectures | Classification is not a promise of pruning support. |

Avoid static “verified” labels unless the evidence names the tested versions, operations, and validation path. Tests in `tests/integration/` may require real model weights; unit tests generally use synthetic fixtures.

## Hardware and dependencies

`pyproject.toml` defines Python 3.10+ and core dependencies. PyTorch installation should be selected for the user's platform and accelerator; CUDA availability and usable memory depend on the installed PyTorch build and hardware. No single GPU size guarantees that a model or workflow will fit. Model weights, activations, sequence length, batch size, and candidate lifecycle all affect memory use.

Use `prunequest doctor`, inspect the workflow's options, and start with a small diagnostic budget. Keep adequate disk space for model caches and checkpoints. CPU execution may be possible for some tasks but can be slow or memory intensive.
