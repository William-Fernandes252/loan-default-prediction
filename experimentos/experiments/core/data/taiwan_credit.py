"""Data transformer for the Taiwan Credit dataset."""

import re

import polars as pl

from experiments.core.data.datasets import Dataset
from experiments.core.data.transformers import get_engine, register_transformer


@register_transformer(Dataset.TAIWAN_CREDIT)
def taiwan_credit_transformer(
    df: pl.DataFrame | pl.LazyFrame, use_gpu: bool = False
) -> pl.DataFrame:
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
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema

    # --- 1. Definition of Mappings ---

    # Columns to process
    columns = schema.names() if isinstance(df, pl.LazyFrame) else df.columns
    pay_cols = [col for col in columns if re.match(r"PAY_[0-6]", col)]
    columns_to_fix = ["LIMIT_BAL"] + [c for c in columns if "PAY" in c or "BILL" in c]

    # --- 2. Start of Preprocessing (Lazy) ---
    df_processed = (
        df.lazy()
        .with_columns(
            # Fix numeric columns stored as strings (`1e+05` for instance)
            [pl.col(c).cast(pl.Float64) for c in columns_to_fix]
        )
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
        .collect(engine=get_engine(use_gpu))
        .to_dummies(columns=categorical_to_encode, separator="_", drop_first=False)
    )

    # Clean column names for compatibility
    final_df_with_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
    ]

    return final_df_with_dummies
