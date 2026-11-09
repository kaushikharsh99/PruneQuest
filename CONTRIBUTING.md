# Contributing to PruneQuest

Thank you for your interest in contributing to PruneQuest!

PruneQuest is an open-source framework dedicated to transparent, benchmark-guided structural LLM pruning. We welcome contributions ranging from model adapters and benchmark integrations to search strategies and documentation improvements.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kaushikharsh99/PruneQuest.git
   cd PruneQuest
   ```

2. **Create a virtual environment and install dependencies in editable mode:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,eval]"
   ```

3. **Verify the installation:**
   ```bash
   prunequest doctor
   pytest tests/unit
   ```

---

## Architectural Guidelines

- **Separation of Concerns:** Model adapters live in `prunequest.models`, benchmarks in `prunequest.benchmarks`, and pruning algorithms in `prunequest.pruning`.
- **Physical vs Masked Pruning:** PruneQuest focuses on physical structural removal. Never silently substitute parameter masking for actual layer or channel slicing.
- **Hardware Awareness:** Always ensure memory usage is bounded for 6 GB–16 GB GPUs. Implement clean garbage collection and avoid multiple simultaneous GPU model allocations.
- **Testing:** Every new model adapter or benchmark must include lightweight synthetic unit tests that execute on CPU without downloading large checkpoints.

---

## Submitting Pull Requests

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Run linting and unit tests:
   ```bash
   ruff check .
   pytest tests/unit
   ```
3. Commit your changes with descriptive messages: `git commit -m "feat(models): add LLaMA adapter"`
4. Push to your fork and submit a Pull Request.
