"""Data processing modules for various datasets.

Includes dataset definitions and factory methods to obtain appropriate data processors, that does the feature engineering for each dataset.
"""

import enum
from pathlib import Path
from typing import Any

from polars import datatypes

from experiments.core.choices import Choice

from .base import DataProcessor
from .corporate_credit import CorporateCreditProcessor
from .lending_club import LendingClubProcessor
from .taiwan_credit import TaiwanCreditProcessor


class Dataset(enum.Enum):
    """Datasets used."""

    CORPORATE_CREDIT_RATING = Choice("corporate_credit_rating", "Corporate Credit Rating")
    LENDING_CLUB = Choice("lending_club", "Lending Club")
    TAIWAN_CREDIT = Choice("taiwan_credit", "Taiwan Credit")

    def __str__(self) -> str:
        return self.id

    @property
    def id(self) -> str:
        return self.value.id

    @property
    def display_name(self) -> str:
        return self.value.display_name

    def get_path(self, base: Path) -> Path:
        """Returns the raw data file path for the dataset."""
        return base / f"{self.id}.csv"

    def get_size_gb(self, base: Path) -> float:
        """Returns the size of the raw data file in GB."""
        path = self.get_path(base)
        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024**3)
        return size_gb

    @classmethod
    def from_id(cls, dataset_id: str) -> "Dataset":
        for member in cls:
            if member.id == dataset_id:
                return member
        raise ValueError(f"Unknown dataset id: {dataset_id}")

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        if isinstance(value, Choice):
            for member in cls:
                if member.value == value:
                    return member
        if isinstance(value, str):
            for member in cls:
                if member.id == value or member.name.lower() == value.lower():
                    return member
        return None

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
