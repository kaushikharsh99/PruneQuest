# Quickstart

PruneQuest provides Python and CLI tools for inspecting language models, evaluating benchmarks, and running structural pruning workflows. Model downloads and evaluation may require substantial disk, RAM, GPU memory, and network access.

## Install

From a checkout, install the package and optional development tools you need:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional extras are `eval` (LM Evaluation Harness) and `studio` (FastAPI/Uvicorn web interface). Install only extras used by your workflow, for example `python -m pip install -e '.[dev,eval,studio]'`.

## Inspect the environment and a model

```bash
prunequest doctor
prunequest inspect --model Qwen/Qwen2.5-0.5B-Instruct
```

Inspection can use Hugging Face configuration metadata; loading weights for evaluation or pruning requires downloading or providing the model files and accepting any model repository access terms.

## Run an optimization

Check the installed CLI's options before starting a large run:

```bash
prunequest optimize --help
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct --dry-run
```

For a real run, choose a supported model and benchmark and set an evaluation budget appropriate to your hardware. Example:

```bash
prunequest optimize Qwen/Qwen2.5-0.5B-Instruct --benchmark gsm8k --retain 0.98 --preset quick
```

Optimization evaluates model candidates and may create large checkpoints and run artifacts. Inspect the CLI help and the resolved configuration/output before committing compute or disk resources. Run artifacts are local and should generally not be committed.

## Other workflows

```bash
prunequest benchmarks list --limit 20
prunequest datasets inspect ./data/eval.jsonl
prunequest runs list
prunequest studio --help
```

The package also provides `block-based` / `block` commands, `flap` / `flap-based` commands, and `full-sweep` / `full-pass` workflows. See the command help and the sample files in `configs/` before use. Studio requires the `studio` extra. For a full API example, see [examples/python-api.py](../examples/python-api.py).

## Development setup

See [CONTRIBUTING.md](../CONTRIBUTING.md) for editable installation, linting, and test commands. The [documentation index](index.md) links to architecture, compatibility, benchmark, and pruning details.
