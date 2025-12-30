"""Data transformer for the Taiwan Credit dataset."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

from experiments.core.data.base import BaseDataTransformer
from experiments.core.data.registry import register_transformer

if TYPE_CHECKING:
    pass


@register_transformer("taiwan_credit")
class TaiwanCreditTransformer(BaseDataTransformer):
    """Transformer for the Taiwan Credit Card dataset.

    Implements the DataTransformer protocol for processing
    Taiwan Credit Card data based on PGC methodology.
    """

    @property
    def dataset_name(self) -> str:
        return "taiwan_credit"

    def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Taiwan Credit specific transformations.

        Processing steps:
        1. Rename target variable 'default.payment.next.month' to 'target'.
        2. Remove the 'ID' column as it is not predictive.
        3. Clean and group categorical features 'EDUCATION' and 'MARRIAGE'.
        4. Map 'SEX' to strings ('Male', 'Female') for clarity.
        5. Normalize PAY_0 to PAY_6 columns.
        6. Convert cleaned categorical columns to one-hot encoding.

        Args:
            df: The raw Polars DataFrame of Taiwan Credit.

        Returns:
            A clean Polars DataFrame ready for the ML pipeline.
        """

        # --- 1. Definition of Mappings ---

        # Payment status columns
        pay_cols = [col for col in df.columns if re.match(r"PAY_[0-6]", col)]

        # --- 2. Start of Preprocessing (Lazy) ---
        df_processed = (
            df.lazy()
            .with_columns(
                # Rename target for consistency
                pl.col("default.payment.next.month").alias("target"),
                # Clean EDUCATION
                pl.when(pl.col("EDUCATION") <= 3)
                .then(pl.col("EDUCATION"))
                .otherwise(4)
                .cast(pl.String)  # Convert to string before to_dummies
                .alias("EDUCATION_CAT"),
                # Clean MARRIAGE
                pl.when(pl.col("MARRIAGE") <= 2)
                .then(pl.col("MARRIAGE"))
                .otherwise(3)
                .cast(pl.String)
                .alias("MARRIAGE_CAT"),
                # Map SEX
                pl.col("SEX").cast(pl.String).alias("SEX_CAT"),
            )
            .with_columns(
                [
                    # Clean PAY_* columns:
                    # Maps values <= 0 (on-time payment, early, etc.) to 0.
                    # Keeps 1-9 (months of delay)
                    pl.when(pl.col(c) <= 0).then(pl.lit(0)).otherwise(pl.col(c)).alias(c)
                    for c in pay_cols
                ]
            )
        )

        # --- 3. Final Selection and One-Hot Encoding ---

        # Categorical columns we just created
        categorical_to_encode = ["EDUCATION_CAT", "MARRIAGE_CAT", "SEX_CAT"]

        # Original columns to drop
        cols_to_drop = ["ID", "default.payment.next.month", "EDUCATION", "MARRIAGE", "SEX"]

        # Collect (execute) the lazy plan and apply to_dummies
        final_df_with_dummies = (
            df_processed.drop(cols_to_drop)
            .collect(engine=self._get_engine())
            .to_dummies(columns=categorical_to_encode, separator="_", drop_first=False)
        )

        # Clean column names for compatibility
        final_df_with_dummies.columns = [
            re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
        ]

        return final_df_with_dummies
