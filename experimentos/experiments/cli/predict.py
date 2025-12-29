"""CLI for running predictions using trained models."""

from pathlib import Path
import sys
from typing import Any

from loguru import logger
import pandas as pd
import typer

from experiments.containers import container
from experiments.services.storage import StorageService

MODULE_NAME = "experiments.cli.predict"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _find_model_uri(storage: StorageService, models_base_uri: str, model_id: str) -> str:
    """Finds the model file URI by ID."""
    # The models are stored in models_base_uri / dataset / model_type / technique / {id}.joblib
    # We search recursively.
    all_files = storage.list_files(models_base_uri, pattern="**/*.joblib")
    matches = [f for f in all_files if f.endswith(f"{model_id}.joblib")]

    if not matches:
        raise FileNotFoundError(f"Model '{model_id}' not found in {models_base_uri}")

    if len(matches) > 1:
        logger.warning(
            f"Multiple models found with ID '{model_id}'. Using the first one: {matches[0]}"
        )

    return matches[0]


def _load_data(input_path: Path) -> pd.DataFrame:
    """Loads input data from CSV or Parquet."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    elif suffix == ".parquet":
        return pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Please use .csv or .parquet")


def _validate_features(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Validates that the input DataFrame has the expected features."""

    # 1. Check feature names if available
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        missing_features = set(expected_features) - set(df.columns)

        if missing_features:
            raise ValueError(
                f"Input data is missing features expected by the model: {missing_features}"
            )

        # Reorder columns to match model expectation
        return df[expected_features]

    # 2. Check number of features
    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != df.shape[1]:
            raise ValueError(
                f"Model expects {model.n_features_in_} features, but input has {df.shape[1]}."
            )

    return df


@app.command("infer")
def predict_with_model(
    model_id: str = typer.Argument(..., help="Identifier of the model to use for prediction."),
    input_path: Path = typer.Argument(
        ..., help="Path to the input data for prediction.", exists=True, dir_okay=False
    ),
    output_path: Path = typer.Argument(
        ..., help="Path to save the prediction results.", dir_okay=False
    ),
):
    """Runs predictions using a specified trained model."""
    # Resolve dependencies from container
    storage_manager = container.storage_manager()
    storage = storage_manager.storage
    settings = container.settings()

    try:
        # 1. Find and Load Model
        logger.info(f"Searching for model '{model_id}'...")
        models_base_uri = StorageService.to_uri(settings.paths.models_dir)
        model_uri = _find_model_uri(storage, models_base_uri, model_id)
        logger.info(f"Loading model from {model_uri}")
        model = storage.read_joblib(model_uri)

        # 2. Load Input Data
        logger.info(f"Loading input data from {input_path}")
        df = _load_data(input_path)

        # 3. Validate and Prepare Data
        if "target" in df.columns:
            logger.info("Dropping 'target' column from input data.")
            df = df.drop(columns=["target"])

        # Validate features
        try:
            df = _validate_features(model, df)
        except ValueError as e:
            logger.error(str(e))
            raise typer.Exit(code=1)

        # 4. Predict
        logger.info("Running prediction...")
        try:
            predictions = model.predict(df)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Try to give a hint
            if "feature_names" in str(e) or "mismatch" in str(e):
                logger.error(
                    "This might be due to feature mismatch. Ensure input columns match training data."
                )
            raise typer.Exit(code=1)

        # 5. Save Results
        logger.info(f"Saving results to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a result DataFrame
        result_df = pd.DataFrame({"prediction": predictions})

        if output_path.suffix.lower() == ".csv":
            result_df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == ".parquet":
            result_df.to_parquet(output_path, index=False)
        else:
            # Default to CSV if unknown extension
            result_df.to_csv(output_path, index=False)

        logger.success("Prediction completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
