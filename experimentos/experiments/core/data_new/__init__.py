"""Data processing definition for the experiment datasets.

This package provides a pipeline-based architecture for data processing:
- **Transformers**: Dataset-specific feature engineering and cleaning
- **Datasets**: Enumeration of supported datasets
- **Registry**: Auto-registration system for transformers
"""

from .corporate_credit import corporate_credit_transformer
from .datasets import Dataset
from .lending_club import lending_club_transformer
from .registry import get_transformer, get_transformer_registry, register_transformer
from .taiwan_credit import taiwan_credit_transformer
from .transformer import Transformer

__all__ = [
    "Dataset",
    "Transformer",
    "corporate_credit_transformer",
    "lending_club_transformer",
    "taiwan_credit_transformer",
    "register_transformer",
    "get_transformer",
    "get_transformer_registry",
]
