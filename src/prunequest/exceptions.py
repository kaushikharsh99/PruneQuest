"""
PruneQuest custom exceptions.
"""

class PruneQuestError(Exception):
    """Base exception for all PruneQuest errors."""
    pass


class ConfigurationError(PruneQuestError):
    """Raised when user configuration or parameters are invalid."""
    pass


class ModelAdapterError(PruneQuestError):
    """Raised when a model adapter fails to load, inspect, or modify an architecture."""
    pass


class BenchmarkError(PruneQuestError):
    """Raised when a benchmark dataset, prompt, or evaluation fails."""
    pass


class PruningError(PruneQuestError):
    """Raised when a pruning operation or validation check fails."""
    pass


class ResourceExhaustionError(PruneQuestError):
    """Raised when GPU VRAM or disk limits are exceeded."""
    pass
