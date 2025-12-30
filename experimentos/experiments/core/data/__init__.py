"""Data processing modules for various datasets.

This package provides a pipeline-based architecture for data processing:
- **Protocols**: Interfaces for loaders, transformers, and exporters
- **Loaders**: Load raw data from various sources (CSV, etc.)
- **Transformers**: Dataset-specific feature engineering and cleaning
- **Exporters**: Persist processed data (Parquet, etc.)
- **Pipeline**: Orchestrates load → transform → export workflow

Also includes dataset definitions and the Dataset enum.
"""

import enum
from pathlib import Path
from typing import Any

from polars import datatypes

from experiments.core.choices import Choice

from .base import BaseDataTransformer, DataProcessor
from .corporate_credit import CorporateCreditTransformer
from .exporters import ParquetDataExporter
from .lending_club import LendingClubTransformer
from .loaders import CsvRawDataLoader
from .pipeline import DataProcessingPipeline, DataProcessingPipelineFactory
from .protocols import (
    DataTransformer,
    InterimDataPathProvider,
    InterimDataUriProvider,
    ProcessedDataExporter,
    RawDataLoader,
    RawDataPathProvider,
    RawDataUriProvider,
)
from .registry import get_transformer, get_transformer_registry, register_transformer
from .taiwan_credit import TaiwanCreditTransformer

# Backward compatibility aliases
LendingClubProcessor = LendingClubTransformer
CorporateCreditProcessor = CorporateCreditTransformer
TaiwanCreditProcessor = TaiwanCreditTransformer


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
    def display_name_from_id(cls, dataset_id: str) -> str:
        dataset = cls.from_id(dataset_id)
        return dataset.display_name

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
    # Dataset enum
    "Dataset",
    # Pipeline components
    "DataProcessingPipeline",
    "DataProcessingPipelineFactory",
    # Protocols (URI-based, new names)
    "DataTransformer",
    "InterimDataUriProvider",
    "ProcessedDataExporter",
    "RawDataLoader",
    "RawDataUriProvider",
    # Loaders
    "CsvRawDataLoader",
    # Transformers
    "BaseDataTransformer",
    "CorporateCreditTransformer",
    "LendingClubTransformer",
    "TaiwanCreditTransformer",
    # Exporters
    "ParquetDataExporter",
    # Registry
    "register_transformer",
    "get_transformer",
    "get_transformer_registry",
    # Backward compatibility
    "DataProcessor",
    "LendingClubProcessor",
    "CorporateCreditProcessor",
    "TaiwanCreditProcessor",
    "InterimDataPathProvider",  # Deprecated alias
    "RawDataPathProvider",  # Deprecated alias
]
