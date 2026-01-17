import polars as pl
from polars import selectors as cs

from experiments.core.modeling.features import TrainingDataset


class FeatureExtractorImpl:
    """Concrete implementation of the FeatureExtractor protocol."""

    def extract_features_and_target(
        self,
        data: pl.DataFrame,
    ) -> TrainingDataset:
        """
        Splits the `DataFrame` into features (X) and target (y), and do some cleaning.

        Ensures no infinite values exist in the features.

        Args:
            data (pl.DataFrame): The transformed dataset containing features and target.

        Returns:
            TrainingDataset: A tuple (X, y) where X is the feature dataframe and y is the target dataframe.
        """
        if "target" not in data.columns:
            raise ValueError("Column for target not found in the provided data frame.")

        return TrainingDataset(
            X=data.drop("target").with_columns(
                cs.float().replace([float("inf"), -float("inf")], float("nan")),
            ),
            y=data.get_column("target").to_frame(name="target"),
        )
