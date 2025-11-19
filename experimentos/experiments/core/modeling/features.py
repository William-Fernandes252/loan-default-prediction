"""Feature extraction and target separation utilities."""

import numpy as np
import polars as pl


def extract_features_and_target(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits the DataFrame into features (X) and target (y).
    Ensures no infinite values exist in the features.

    Args:
        df: The processed dataframe containing features and the 'target' column.

    Returns:
        A tuple (X, y) where X is the feature dataframe and y is the target dataframe.
    """
    if "target" not in df.columns:
        # This check is pure logic, unrelated to which file it came from
        raise ValueError("Column 'target' not found in the provided DataFrame.")

    target_series = df.get_column("target")
    feature_df = df.drop("target")

    # Sanity Check and Conversion
    # Ensure that there are no infinite values that would break Scikit-Learn
    # We utilize Pandas for this specific replacement to match legacy behavior,
    # though Polars expressions could also be used.
    X_pd = feature_df.to_pandas().replace([np.inf, -np.inf], np.nan)
    y_pd = target_series.to_pandas()

    # Convert back to Polars to save efficiently in Parquet
    X_final = pl.from_pandas(X_pd)
    y_final = pl.from_pandas(y_pd.to_frame(name="target"))

    return X_final, y_final
