# Contributing to PruneQuest

Thanks for contributing. Please open an issue for substantial changes so the intended behavior can be discussed before implementation. Small fixes and documentation corrections can go directly to a pull request.

## Development setup

```bash
git clone https://github.com/kaushikharsh99/PruneQuest.git
cd PruneQuest
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional extras: `eval` installs LM Evaluation Harness; `studio` installs web server dependencies. Install both with `python -m pip install -e '.[dev,eval,studio]'` if needed.

## Checks

Run focused checks for your change, then the relevant broader suite:

```bash
ruff check .
pytest tests/unit
```

The repository's pytest configuration adds `src` to the import path. Integration tests may require network access, model downloads, authentication, and significant memory. Inspect markers and fixtures before running them. Frontend package scripts are in `frontend/package.json`.

## Engineering principles

- Keep model, pruning, evaluation, persistence, and UI responsibilities in their existing package areas.
- Do not describe masked weights as physical structural pruning or parameter/storage reduction.
- Treat architecture inspection, evaluation, pruning, export, and reload as separate capability claims.
- Prefer deterministic, CPU-only synthetic tests for new core behavior. Avoid making unit tests depend on remote models or datasets.
- Preserve benchmark split integrity and stable baselines in optimization workflows.
- Be mindful of model memory and checkpoint size. Avoid unnecessary simultaneous model copies.
- Update examples and docs when changing user-facing commands, config formats, defaults, or supported behavior.

## Pull requests

Keep changes focused, describe user-visible behavior and compatibility implications, and include the checks you ran. Do not commit credentials, downloaded weights, run outputs, or generated model checkpoints. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
