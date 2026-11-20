# Structural Pruning

PruneQuest contains multiple pruning paths. Check the selected implementation and model adapter for exact supported operations; shared terminology does not imply identical behavior across workflows.

## Layer pruning

The core pruning package includes importance scoring and layer-removal components (`src/prunequest/pruning/`). Model-specific surgery belongs to adapters in `src/prunequest/models/`. Correct surgery must update the module structure and any configuration or layer indices required by the model implementation.

## FLAP width pruning

The separate FLAP package implements calibration, scoring, channel slicing, evaluation, and checkpoint workflows. See `src/prunequest/flap/` and sample configurations in `configs/`. Only attempt projection layouts explicitly handled by the relevant adapter.

## Joint sweep

Joint sweep orchestration lives in `src/prunequest/sweep/` and `src/prunequest/orchestrator/`. Read the corresponding configuration schema and CLI help before changing or launching a sweep.

## Correctness principles

- Physical module/tensor surgery is required for structural parameter reduction; masking is a different method.
- Validate model forward behavior after surgery and ensure saved configurations match the resulting structure.
- Compare candidates to a stable unpruned baseline using the same evaluation setup.
- Rejected candidates must not contaminate accepted state. Preserve rollback and checkpoint lifecycle guarantees when modifying search code.
- Avoid concurrent copies of large models on constrained devices; follow existing cleanup and device-transfer patterns.

The formulas and implementation details are algorithm-specific. Use code and tests as the authoritative reference when documenting a particular scoring method.
