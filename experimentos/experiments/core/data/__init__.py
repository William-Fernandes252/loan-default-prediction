"""Data processing definition for the experiment datasets.

This package provides a pipeline-based architecture for data processing:
- **Transformers**: Dataset-specific feature engineering and cleaning
- **Datasets**: Enumeration of supported datasets
- **Registry**: Auto-registration system for transformers
"""

from .corporate_credit import corporate_credit_transformer
from .datasets import Dataset
from .lending_club import lending_club_transformer
from .repository import DataRepository
from .taiwan_credit import taiwan_credit_transformer
from .transformers import (
    Transformer,
    TransformerRegistry,
    get_engine,
    get_transformer,
    get_transformer_registry,
    register_transformer,
)

__all__ = [
    "Dataset",
    "DataRepository",
    "Transformer",
    "TransformerRegistry",
    "get_transformer",
    "get_transformer_registry",
    "get_engine",
    "corporate_credit_transformer",
    "lending_club_transformer",
    "taiwan_credit_transformer",
    "register_transformer",
]
