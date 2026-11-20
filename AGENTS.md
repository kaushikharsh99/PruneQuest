# Agent Guide

This file is the repository's working guide for coding agents. Read it before making changes; follow the relevant docs linked below and inspect the implementation when behavior is unclear.

## Project map

- `src/prunequest/cli.py`: Typer CLI and top-level commands.
- `src/prunequest/api.py`: public Python API.
- `src/prunequest/models/`: model inspection, adapters, and registry.
- `src/prunequest/pruning/`, `src/prunequest/flap/`, `src/prunequest/sweep/`: structural pruning and optimization.
- `src/prunequest/benchmarks/`, `src/prunequest/datasets/`: evaluation, specs, dataset handling.
- `src/prunequest/runs/`, `src/prunequest/core/tracking.py`, `src/prunequest/reporting/`: run persistence and reports.
- `src/prunequest/studio/`: optional local web UI and API.
- `frontend/`: Studio frontend source; packaged static output is under `src/prunequest/studio/static/`.
- `tests/unit/`, `tests/integration/`, `tests/flap/`, `tests/studio/`, `tests/sweep/`: test suites. Some integration tests need external model weights; inspect markers and fixtures before running them.
- `configs/`, `examples/`, `docs/`: example configurations and user/developer documentation.

## Setup and commands

The package requires Python 3.10+. For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,eval,studio]'
```

Use the smallest relevant checks for a change:

```bash
ruff check .
pytest tests/unit
```

`pytest` is configured with `src` on `pythonpath`. Run focused tests while iterating. Real-model tests may download large checkpoints and consume substantial CPU/GPU memory; do not run those casually. Frontend commands are defined in `frontend/package.json`.

## Change guidance

- Preserve physical structural pruning semantics. Do not replace actual module or tensor surgery with masks or zeroing while claiming parameter/storage reduction.
- Treat model-family support as capability-specific. Inspection, loading, evaluation, pruning, checkpoint export, and reload validation are separate claims; document the exact demonstrated scope.
- Add or update lightweight synthetic tests for changed behavior. Avoid requiring network access or pretrained weights in unit tests.
- Keep benchmark splits and dataset handling reproducible. Do not use held-out test data to make pruning acceptance decisions.
- Be careful with device placement and model copies: LLM weights can exhaust GPU memory. Prefer existing lifecycle and cleanup patterns.
- Keep CLI help, README examples, docs, and tests consistent with actual command signatures and defaults.
- Do not commit generated run directories, downloaded weights, secrets, or large model artifacts. Review `git status` before finishing.
- For frontend edits, update source under `frontend/` and only regenerate packaged assets using the project's established build workflow.

## Documentation map

Start at [README.md](README.md), then use [docs/quickstart.md](docs/quickstart.md), [docs/architecture.md](docs/architecture.md), [docs/compatibility.md](docs/compatibility.md), [docs/benchmarks.md](docs/benchmarks.md), [docs/pruning.md](docs/pruning.md), and [docs/extending.md](docs/extending.md). `docs/migration-v0.2.md` is historical context, not a specification of current behavior.

## Before handing off

Summarize changed files and checks run. If a check was not run, say so. Report any uncertainty about hardware-dependent behavior or capability claims rather than presenting it as verified.
