"""Data processor for the Lending Club dataset."""

import re

import polars as pl
from polars import datatypes

from experiments.core.data.base import DataProcessor


class LendingClubProcessor(DataProcessor):
    dataset_name = "lending_club"

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Processes the Lending Club dataset based on the methodology from
        Namvar et al. (2018).

        This includes:
        1.  Binarization of the target variable ('loan_status').
        2.  Feature engineering (calculation of 'credit_age', 'new_dti', etc.).
        3.  Cleaning and transformation of columns (e.g., 'emp_length').
        4.  Log transformation of skewed features.
        5.  One-hot encoding of categorical features.
        6.  Final feature selection for the model, removing leaky data.
        Args:
            df: The raw Polars DataFrame of the Lending Club dataset.

        Returns:
            A cleaned Polars DataFrame ready for the ML pipeline.
        """

        # --- 1. Definition of the Target Variable ---
        # Based on Section 4.1, "current" loans are removed.
        # The target is binary: 'Charged Off' (1) vs 'Fully Paid' (0).
        target_map = {"Charged Off": 1, "Fully Paid": 0}

        # --- 2. Start of Preprocessing (Lazy) ---
        df_processed = (
            df.lazy()
            .filter(pl.col("loan_status").is_in(target_map.keys()))
            .with_columns(
                # Create the target column 'target'
                pl.when(pl.col("loan_status") == "Charged Off")
                .then(1)
                .otherwise(0)
                .alias("target"),
                # Convert date strings to datetime for calculation
                pl.col("issue_d").str.to_date(format="%b-%Y", strict=False).alias("issue_d_dt"),
                pl.col("earliest_cr_line")
                .str.to_date(format="%b-%Y", strict=False)
                .alias("earliest_cr_line_dt"),
                # Convert 'emp_length' to numeric
                pl.col("emp_length")
                .str.extract(r"(\d+)", 1)
                .cast(pl.Float64)
                .alias("emp_length_num"),
            )
            .with_columns(
                # --- 3. Feature Engineering (Based on Table 1 and Section 4.1) ---
                # Calculate 'credit_age'
                # (credit age in months, from the opening of the 1st line to the loan issue)
                (
                    (pl.col("issue_d_dt") - pl.col("earliest_cr_line_dt")).dt.total_days()
                    / 30.4375
                ).alias("credit_age_months"),
                # Calculate 'monthly_inc' to use in ratios
                (pl.col("annual_inc") / 12).alias("monthly_inc"),
            )
            .with_columns(
                # Calculate Derived Ratios [cite: 1999-2007]
                # 'income_to_payment_ratio' [cite: 1999]
                (pl.col("monthly_inc") / pl.col("installment")).alias("income_to_payment_ratio"),
                # 'revolving_to_income_ratio' [cite: 1995]
                (pl.col("revol_bal") / pl.col("monthly_inc")).alias("revol_to_income_ratio"),
                # 'new_dti' (New Debt-to-Income) [cite: 2000-2007]
                # NMRA = New Monthly Repayment Amount
                (
                    ((pl.col("dti") * pl.col("monthly_inc")) + pl.col("installment"))
                    / pl.col("monthly_inc")
                ).alias("new_dti"),
            )
            .with_columns(
                # Handle divisions by zero resulting in infinity only in float columns
                pl.col(datatypes.Float32).replace([float("inf"), float("-inf")], None)
            )
            .with_columns(
                # --- 4. Log Transformations ---
                # Apply log1p (log(x+1)) to skewed features
                pl.col("annual_inc").fill_null(0).log1p(),
                # Fill nulls with 0 before log. Nulls here likely mean
                # division by 0 (e.g., no 'installment' or 'monthly_inc')
                pl.col("income_to_payment_ratio").fill_null(0).log1p(),
                pl.col("revol_to_income_ratio").fill_null(0).log1p(),
            )
        )

        # --- 5. Final Feature Selection ---
        # Based on Table 1 of the paper
        # Numeric features to keep (including the ones we created)
        numeric_cols = [
            "loan_amnt",
            "installment",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "inq_last_6mths",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "total_acc",
            "avg_cur_bal",
            "total_rev_hi_lim",
            "acc_open_past_24mths",
            "percent_bc_gt_75",
            "inq_fi",
            "emp_length_num",  # Processed variable
            "credit_age_months",  # Credit age variable
            "new_dti",  # Derived variable [cite: 2000]
            "income_to_payment_ratio",  # Derived variable [cite: 1999]
            "revol_to_income_ratio",  # Derived variable [cite: 1995]
        ]

        # Categorical features to keep (for one-hot encoding)
        categorical_cols = ["term", "home_ownership", "verification_status", "purpose"]

        target_col = ["target"]

        # Select only the columns of interest
        final_features_df = df_processed.select(numeric_cols + categorical_cols + target_col)

        # --- 6. One-Hot Encoding ---
        # Convert categorical columns to dummy (binary) columns
        final_df_with_dummies = final_features_df.collect(engine=self._get_engine()).to_dummies(
            columns=categorical_cols,
            separator="_",
            drop_first=False,  # Keep all categories
        )

        # Replace invalid characters in column names (for Scikit-learn)
        final_df_with_dummies.columns = [
            re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
        ]

        return final_df_with_dummies
