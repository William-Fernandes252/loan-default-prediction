"""Feature extraction and target separation utilities."""

import polars as pl
from polars import selectors as cs


def extract_features_and_target(
    transformed_data: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits the DataFrame into features (X) and target (y), and do some cleaning.
    Ensures no infinite values exist in the features.

    Args:
        transformed_data: The transformed dataset containing features and target.

    Returns:
        A tuple (X, y) where X is the feature dataframe and y is the target dataframe.
    """
    if "target" not in transformed_data.columns:
        # This check is pure logic, unrelated to which file it came from
        raise ValueError("Column 'target' not found in the provided DataFrame.")

    return transformed_data.drop("target").with_columns(
        cs.float().replace([float("inf"), -float("inf")], float("nan")),
    ), transformed_data.get_column("target").to_frame(name="target")
