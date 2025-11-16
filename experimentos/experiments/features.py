from pathlib import Path

import joblib
from loguru import logger
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer
from typing_extensions import Annotated

from experiments.config import Dataset

app = typer.Typer()

# --- Specific Pipeline Functions ---


def get_preprocessor(df: pl.DataFrame) -> ColumnTransformer:
    """
    Creates and returns the ColumnTransformer (preprocessing pipeline)
    based on the DataFrame columns.

    This function assumes the data has already gone through Step 2's
    preprocessing (target binarization, one-hot encoding).
    """

    # Identify all numeric columns (except the 'target')
    # At this point, all features are already numeric (original or one-hot)
    numeric_features = df.select(pl.col(pl.NUMERIC_DTYPES).exclude("target")).columns

    logger.info(f"Identified {len(numeric_features)} numeric features for imputation and scaling.")

    # Create the preprocessing pipeline
    # 1. Impute missing values (nulls) with the median (robust to outliers)
    # 2. Scale (normalize) features to a mean of 0 and a standard deviation of 1
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Create the ColumnTransformer
    # We apply the 'numeric_transformer' to all 'numeric_features'
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",  # Discards unlisted columns (none, in this case)
    )

    return preprocessor


# --- Main Command (Typer) ---


@app.command()
def main(
    dataset: Annotated[
        Dataset,
        typer.Argument(..., help="Dataset to be processed."),
    ],
):
    """
    Performs splitting and processing (Imputation/Scaling).

    Loads data from 'data/processed/', splits it into train/test,
    applies imputation and scaling ONLY to the training set, and saves the results
    back to 'data/processed/'.
    """
    logger.info(f"Starting Step 3 (Features) for dataset: {dataset.value}")

    # 1. Load processed data from Step 2
    input_path = Path("data") / "processed" / f"{dataset.value}_processed.parquet"
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        logger.error("Did you run 'dataset.py' for this dataset first?")
        raise typer.Exit(code=1)

    logger.info(f"Loading data from {input_path}")
    df = pl.read_parquet(input_path)

    # 2. Separate X (features) and y (target)
    # We convert to pandas/numpy here, as it's the format Scikit-learn expects
    X = df.drop("target").to_pandas()
    y = df.select("target").to_series().to_pandas()

    # 3. Train/Test Split (70/30, stratified)
    # [cite_start]As defined in your methodology [cite: 3094, 3095]
    logger.info("Splitting data into training (70%) and testing (30%) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,  # For reproducibility
        stratify=y,  # Essential for imbalanced data [cite: 3094]
    )

    # 4. Create and "fit" the preprocessing pipeline
    logger.info("Defining the preprocessing pipeline (Imputer + Scaler)...")
    # We pass the 'df' DataFrame (without the target) just to extract column names
    preprocessor = get_preprocessor(df.drop("target"))

    logger.info("Fitting the preprocessing pipeline ONLY on the training data...")
    # NOTE: We use .fit_transform() on the training set...
    X_train_processed = preprocessor.fit_transform(X_train)

    logger.info("Applying (transform) the pipeline to the test data...")
    # ...and ONLY .transform() on the test set. This prevents data leakage!
    X_test_processed = preprocessor.transform(X_test)

    # Retrieve column names after transformation
    feature_names = preprocessor.get_feature_names_out()

    # 5. Save the results

    # Save the "trained" pipeline (containing medians and scales)
    pipeline_path = Path("data") / "processed" / f"{dataset.value}_preprocessor.joblib"
    joblib.dump(preprocessor, pipeline_path)
    logger.success(f"Preprocessing pipeline saved to: {pipeline_path}")

    # Save the processed and split data as Polars/Parquet DataFrames
    # (Converting from numpy arrays back to Polars)
    (
        pl.DataFrame(X_train_processed, schema=feature_names.tolist()).write_parquet(
            Path("data") / "processed" / f"{dataset.value}_X_train_processed.parquet"
        )
    )
    (
        pl.DataFrame(X_test_processed, schema=feature_names.tolist()).write_parquet(
            Path("data") / "processed" / f"{dataset.value}_X_test_processed.parquet"
        )
    )
    (
        pl.DataFrame(y_train, schema={"target": pl.Int64}).write_parquet(
            Path("data") / "processed" / f"{dataset.value}_y_train.parquet"
        )
    )
    (
        pl.DataFrame(y_test, schema={"target": pl.Int64}).write_parquet(
            Path("data") / "processed" / f"{dataset.value}_y_test.parquet"
        )
    )

    logger.success("Processed train/test data saved in 'data/processed/'")


if __name__ == "__main__":
    app()
