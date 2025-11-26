from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""

    cv_folds: int
    cost_grids: list[Any]
    discard_checkpoints: bool = False
