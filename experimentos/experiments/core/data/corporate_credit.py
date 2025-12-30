"""Data transformer for the Corporate Credit dataset."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

from experiments.core.data.base import BaseDataTransformer
from experiments.core.data.registry import register_transformer

if TYPE_CHECKING:
    pass


@register_transformer("corporate_credit_rating")
class CorporateCreditTransformer(BaseDataTransformer):
    """Transformer for the Corporate Credit Rating dataset.

    Implements the DataTransformer protocol for processing
    Corporate Credit data based on PGC methodology.
    """

    @property
    def dataset_name(self) -> str:
        return "corporate_credit"

    def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Corporate Credit specific transformations.

        Processing steps:
        1. Binarize target variable 'Rating' ('D' -> 1, others -> 0).
        2. Remove non-predictive metadata columns (Name, Symbol, Date).
        3. One-hot encode the categorical feature 'Sector'.
        4. Select all financial indicators (Float64) as features.

        Args:
            df: The raw Polars DataFrame of Corporate Credit.

        Returns:
            A clean Polars DataFrame ready for the ML pipeline.
        """

        # --- 1. Definition of Target Variable ---
        # As per Section 3.2 of chapter/ferramentas.tex
        df_processed = df.lazy().with_columns(
            pl.when(pl.col("Rating") == "D").then(pl.lit(1)).otherwise(pl.lit(0)).alias("target")
        )

        # --- 2. Feature Selection ---

        # Automatically identify all numeric (Float64) columns
        # which are the fundamental indicators.
        numeric_cols = [col for col, dtype in df.schema.items() if dtype == pl.Float64]

        # Categorical column to be encoded
        categorical_cols = ["Sector"]

        # Target column
        target_col = ["target"]

        # Select only the columns of interest
        # This discards 'Rating' (original), 'Name', 'Symbol', 'Rating Agency Name', 'Date'
        final_features_df = df_processed.select(numeric_cols + categorical_cols + target_col)

        # --- 3. One-Hot Encoding of the 'Sector' feature ---
        # Collect the result (execute the lazy plan) and apply to_dummies
        final_df_with_dummies = final_features_df.collect(engine=self._get_engine()).to_dummies(
            columns=categorical_cols,
            separator="_",
            drop_first=False,  # Keep all categories
        )

        # Clean column names for compatibility (remove special characters)
        final_df_with_dummies.columns = [
            re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
        ]

        return final_df_with_dummies
