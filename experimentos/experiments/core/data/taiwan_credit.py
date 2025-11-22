"""Data processor for the Taiwan Credit dataset."""

import re

import polars as pl

from experiments.core.data.base import DataProcessor


class TaiwanCreditProcessor(DataProcessor):
    dataset_name = "taiwan_credit"

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocesses the Taiwan Credit Card dataset.

        This function performs the following steps based on the dataset description
        and the PGC methodology:
        1.  Renames the target variable 'default.payment.next.month' to 'target'
            for consistency.
        2.  Removes the 'ID' column as it is not predictive.
        3.  Cleans and groups the categorical features 'EDUCATION' and 'MARRIAGE'
            to remove "unknown" or "other" values and simplify.
        4.  Maps 'SEX' to strings ('Male', 'Female') for clarity.
        5.  Normalizes the columns 'PAY_0' to 'PAY_6', treating "on-time payment"
            (values <= 0) as 0 and keeping the months of delay (1-9).
        6.  Converts the cleaned categorical columns to one-hot encoding.

        Args:
            df: The raw Polars DataFrame of Taiwan Credit.

        Returns:
            A clean Polars DataFrame ready for the ML pipeline.
        """

        # --- 1. Definition of Mappings ---

        # Payment status columns
        pay_cols = [
            col
            for col in df.columns
            if any([col.startswith("PAY_"), col.startswith("BILL_"), col.startswith("PAY_AMT_")])
        ]

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
