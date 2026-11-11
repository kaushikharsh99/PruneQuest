"""
Optimization presets for PruneQuest.
Centrally defines computational budgets, sample sizes, and search constraints
for quick, balanced, and thorough optimization runs.
"""

from dataclasses import dataclass
from enum import Enum


class OptimizationPreset(str, Enum):
    QUICK = "quick"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass(frozen=True)
class PresetConfig:
    name: str
    description: str
    calibration_samples: int
    screening_samples: int
    acceptance_samples: int
    max_iterations: int
    max_candidate_evaluations: int
    initial_flap_ratio: float
    minimum_flap_ratio: float
    screening_loss_factor: float
    time_budget_sec: int | None = None


PRESETS: dict[OptimizationPreset, PresetConfig] = {
    OptimizationPreset.QUICK: PresetConfig(
        name="quick",
        description="Fast 5-minute exploratory run for sanity checks and initial feasibility validation.",
        calibration_samples=32,
        screening_samples=16,
        acceptance_samples=64,
        max_iterations=2,
        max_candidate_evaluations=4,
        initial_flap_ratio=0.05,
        minimum_flap_ratio=0.01,
        screening_loss_factor=3.0,
        time_budget_sec=600,
    ),
    OptimizationPreset.BALANCED: PresetConfig(
        name="balanced",
        description="Production default: Multi-stage screening, rigorous validation, and high accuracy retention.",
        calibration_samples=128,
        screening_samples=64,
        acceptance_samples=256,
        max_iterations=6,
        max_candidate_evaluations=12,
        initial_flap_ratio=0.05,
        minimum_flap_ratio=0.005,
        screening_loss_factor=2.5,
        time_budget_sec=None,
    ),
    OptimizationPreset.THOROUGH: PresetConfig(
        name="thorough",
        description="Deep Pareto frontier search with comprehensive validation budget and granular channel steps.",
        calibration_samples=256,
        screening_samples=128,
        acceptance_samples=512,
        max_iterations=12,
        max_candidate_evaluations=24,
        initial_flap_ratio=0.03,
        minimum_flap_ratio=0.002,
        screening_loss_factor=2.0,
        time_budget_sec=None,
    ),
}


def get_preset_config(preset: OptimizationPreset | str) -> PresetConfig:
    """Retrieves standard configuration parameters for a given preset."""
    if isinstance(preset, str):
        try:
            preset = OptimizationPreset(preset.lower())
        except ValueError:
            valid = ", ".join(p.value for p in OptimizationPreset)
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {valid}.")
    return PRESETS[preset]
