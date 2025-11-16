import re
import sys

from joblib import Parallel, cpu_count, delayed
from loguru import logger
import polars as pl
from polars import datatypes
import typer
from typing_extensions import Annotated

from experiments.config import Dataset

MODULE_NAME = "experiments.dataset"

if __name__ == "__main__":
    # When executed via `python -m experiments.dataset`, the module's __name__ becomes
    # "__main__". This breaks joblib because functions get pickled under that name
    # and child processes cannot import the corresponding module. Explicitly alias the
    # current module to its canonical name so joblib (and other tooling) can resolve it.
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _confirm_overwrite(processed_path) -> bool:
    """Ask the user if an existing processed file should be overwritten."""

    prompt = f"Processed file '{processed_path}' already exists. Overwrite it?"
    return typer.confirm(prompt, default=False)


def _filter_datasets_for_processing(
    datasets: list[Dataset],
    *,
    force: bool,
) -> list[Dataset]:
    """Return the subset of datasets approved for processing."""

    ready: list[Dataset] = []
    for dataset in datasets:
        processed_path = dataset.get_processed_data_path()
        if processed_path.exists() and not force:
            if _confirm_overwrite(processed_path):
                ready.append(dataset)
            else:
                logger.info(f"Skipping dataset {dataset} per user choice.")
        else:
            ready.append(dataset)

    return ready


def _process_single_dataset(dataset: Dataset) -> tuple[Dataset, bool, str | None]:
    """Runs the preprocessing pipeline for a single dataset."""

    try:
        logger.info(f"Processing dataset {dataset}...")

        raw_data_path = dataset.get_raw_data_path()
        processed_data_path = dataset.get_processed_data_path()

        logger.info(f"Loading raw data from {raw_data_path}...")
        raw_data = pl.read_csv(raw_data_path, **dataset.get_extra_params())
        logger.info("Raw data loaded.")

        processed_data = dataset.process_data(raw_data)

        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving processed data to {processed_data_path}...")
        processed_data.write_parquet(processed_data_path)

        logger.success(f"Processing dataset {dataset} complete.")
        return dataset, True, None
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to process dataset {dataset}: {exc}")
        return dataset, False, str(exc)


@Dataset.LENDING_CLUB.register_dataset_processor()
def preprocess_lending_club_data(df: pl.DataFrame) -> pl.DataFrame:
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
            pl.when(pl.col("loan_status") == "Charged Off").then(1).otherwise(0).alias("target"),
            # Convert date strings to datetime for calculation
            pl.col("issue_d").str.to_date(format="%b-%Y", strict=False).alias("issue_d_dt"),
            pl.col("earliest_cr_line")
            .str.to_date(format="%b-%Y", strict=False)
            .alias("earliest_cr_line_dt"),
            # Convert 'emp_length' to numeric
            pl.col("emp_length").str.extract(r"(\d+)", 1).cast(pl.Float64).alias("emp_length_num"),
        )
        .with_columns(
            # --- 3. Feature Engineering (Based on Table 1 and Section 4.1) ---
            # Calculate 'credit_age'
            # (credit age in months, from the opening of the 1st line to the loan issue)
            (
                (pl.col("issue_d_dt") - pl.col("earliest_cr_line_dt")).dt.total_days() / 30.4375
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
    final_df_with_dummies = final_features_df.collect(engine="gpu").to_dummies(
        columns=categorical_cols,
        separator="_",
        drop_first=False,  # Keep all categories
    )

    # Replace invalid characters in column names (for Scikit-learn)
    final_df_with_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
    ]

    return final_df_with_dummies


@Dataset.CORPORATE_CREDIT_RATING.register_dataset_processor()
def preprocess_corporate_credit_data(raw_data: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocesses the Corporate Credit dataset.

    This function performs the following steps based on the PGC methodology:
    1.  Binarizes the target variable 'Rating':
        - 'D' (highest risk) becomes 1 (positive/minority class).
        - All others ('AAA' to 'C') become 0 (negative/majority class).
    2.  Removes non-predictive metadata columns (e.g., Name, Symbol, Date).
    3.  One-hot encodes the categorical feature 'Sector'.
    4.  Selects all financial indicators (Float64) as features.

    Args:
        df: The raw Polars DataFrame of Corporate Credit.

    Returns:
        A clean Polars DataFrame ready for the ML pipeline.
    """

    # --- 1. Definition of Target Variable ---
    # As per Section 3.2 of chapter/ferramentas.tex
    df_processed = raw_data.lazy().with_columns(
        pl.when(pl.col("Rating") == "D").then(pl.lit(1)).otherwise(pl.lit(0)).alias("target")
    )

    # --- 2. Feature Selection ---

    # Automatically identify all numeric (Float64) columns
    # which are the fundamental indicators.
    numeric_cols = [col for col, dtype in raw_data.schema.items() if dtype == pl.Float64]

    # Categorical column to be encoded
    categorical_cols = ["Sector"]

    # Target column
    target_col = ["target"]

    # Select only the columns of interest
    # This discards 'Rating' (original), 'Name', 'Symbol', 'Rating Agency Name', 'Date'
    final_features_df = df_processed.select(numeric_cols + categorical_cols + target_col)

    # --- 3. One-Hot Encoding of the 'Sector' feature ---
    # Collect the result (execute the lazy plan) and apply to_dummies
    final_df_with_dummies = final_features_df.collect(engine="gpu").to_dummies(
        columns=categorical_cols,
        separator="_",
        drop_first=False,  # Keep all categories
    )

    # Clean column names for compatibility (remove special characters)
    final_df_with_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
    ]

    return final_df_with_dummies


@Dataset.TAIWAN_CREDIT.register_dataset_processor()
def preprocess_taiwan_credit_data(raw_data: pl.DataFrame) -> pl.DataFrame:
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
        for col in raw_data.columns
        if any([col.startswith("PAY_"), col.startswith("BILL_"), col.startswith("PAY_AMT_")])
    ]

    # --- 2. Start of Preprocessing (Lazy) ---
    df_processed = (
        raw_data.lazy()
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
        .collect(engine="gpu")
        .to_dummies(columns=categorical_to_encode, separator="_", drop_first=False)
    )

    # Clean column names for compatibility
    final_df_with_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
    ]

    return final_df_with_dummies


@app.command(name="process")
def main(
    dataset: Annotated[
        Dataset | None,
        typer.Argument(
            help=(
                "Identifier of the dataset to process. "
                "When omitted, all datasets are processed in parallel."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing processed files without prompting.",
        ),
    ] = False,
):
    """Processes one or all datasets using joblib to manage workloads."""

    datasets_to_process = [dataset] if dataset is not None else list(Dataset)
    datasets_to_process = _filter_datasets_for_processing(
        datasets_to_process,
        force=force,
    )

    if not datasets_to_process:
        logger.info("No datasets selected for processing. Exiting.")
        return

    dataset_names = ", ".join(ds.value for ds in datasets_to_process)
    logger.info(
        f"Scheduling preprocessing for {len(datasets_to_process)} dataset(s): {dataset_names}"
    )

    available_cpus = cpu_count() or 1
    n_jobs = min(len(datasets_to_process), max(1, available_cpus))
    logger.info(f"Using {n_jobs} parallel job(s) managed by joblib.")

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(ds) for ds in datasets_to_process
    )

    failed = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.value for ds in failed)
        logger.error(f"Processing failed for the following dataset(s): {failed_names}")
        raise typer.Exit(code=1)

    logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    for _func in [
        _process_single_dataset,
        preprocess_lending_club_data,
        preprocess_corporate_credit_data,
        preprocess_taiwan_credit_data,
        main,
    ]:
        _func.__module__ = MODULE_NAME

    app()
