"""Data processing modules for various datasets.

Includes dataset definitions and factory methods to obtain appropriate data processors, that does the feature engineering for each dataset.
"""

import enum
from pathlib import Path
from typing import Any

from polars import datatypes

from .base import DataProcessor
from .corporate_credit import CorporateCreditProcessor
from .lending_club import LendingClubProcessor
from .taiwan_credit import TaiwanCreditProcessor


class Dataset(enum.Enum):
    """Datasets used."""

    CORPORATE_CREDIT_RATING = "corporate_credit_rating"
    LENDING_CLUB = "lending_club"
    TAIWAN_CREDIT = "taiwan_credit"

    def __str__(self) -> str:
        return self.value

    def get_path(self, base: Path) -> Path:
        """Returns the raw data file path for the dataset."""
        return base / f"{self.value}.csv"

    def get_size_gb(self, base: Path) -> float:
        """Returns the size of the raw data file in GB."""
        path = self.get_path(base)
        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024**3)
        return size_gb

    def get_extra_params(self) -> dict[str, Any]:
        """Returns extra parameters specific to the dataset, if any."""
        extra_params: dict[Dataset, dict[str, Any]] = {
            Dataset.LENDING_CLUB: {"schema_overrides": {"id": datatypes.Utf8}},
            Dataset.TAIWAN_CREDIT: {"infer_schema_length": None},
        }
        return extra_params.get(self, {})


__all__ = [
    "Dataset",
    "DataProcessor",
    "LendingClubProcessor",
    "CorporateCreditProcessor",
    "TaiwanCreditProcessor",
]
