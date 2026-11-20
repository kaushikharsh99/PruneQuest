# Extending PruneQuest

Use the existing implementation and tests as the extension contract. APIs may differ across the core benchmark, model, FLAP, and Studio subsystems.

## Add or update a model adapter

1. Inspect `src/prunequest/models/base.py`, a nearby family adapter, and `src/prunequest/models/registry.py`.
2. Implement only behavior supported by the target architecture. Keep module/config dimensions and layer indices consistent after surgery.
3. Add CPU-only synthetic tests for discovery, structural edits, forward validation, and edge cases where practical.
4. If FLAP support is intended, inspect `src/prunequest/flap/models/` separately; core adapter support does not imply FLAP support.
5. Update the compatibility docs with the precise capability and evidence.

## Add a benchmark or scorer

Inspect `src/prunequest/benchmarks/base.py`, `registry.py`, `spec.py`, and existing implementations. Keep dataset loading, prompt construction, scoring, and split selection explicit. Add small deterministic fixtures that do not fetch remote data.

## Add an optimization strategy

Inspect `src/prunequest/core/`, `src/prunequest/orchestrator/`, and `src/prunequest/sweep/` to identify the correct layer. Preserve baseline integrity, candidate rollback, and run recording. Add focused tests for acceptance, rejection, resume, and failure paths relevant to the change.

## Update Studio

The backend is under `src/prunequest/studio/`; frontend source is under `frontend/`. Keep API schemas, frontend types, route behavior, and security middleware aligned. Build commands are defined in `frontend/package.json`.

## Documentation and verification

Update CLI help/examples whenever interfaces change. Run focused tests and `ruff check .` for Python changes when available. Real-model integration tests can download large weights; run them only when needed and with sufficient resources. See [AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
