"""Tests for feature_extractor service."""

import polars as pl
import pytest

from experiments.services.feature_extractor import FeatureExtractorImpl


class DescribeFeatureExtractorImpl:
    @pytest.fixture
    def extractor(self) -> FeatureExtractorImpl:
        return FeatureExtractorImpl()


class DescribeExtractFeaturesAndTarget:
    @pytest.fixture
    def extractor(self) -> FeatureExtractorImpl:
        return FeatureExtractorImpl()

    def it_extracts_target_column(self, extractor: FeatureExtractorImpl) -> None:
        data = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )

        result = extractor.extract_features_and_target(data)

        assert result.y.columns == ["target"]
        assert result.y.shape == (3, 1)

    def it_extracts_features_without_target(self, extractor: FeatureExtractorImpl) -> None:
        data = pl.DataFrame(
            {
                "feature1": [1.0, 2.0],
                "feature2": [3.0, 4.0],
                "target": [0, 1],
            }
        )

        result = extractor.extract_features_and_target(data)

        assert "target" not in result.X.columns
        assert result.X.columns == ["feature1", "feature2"]

    def it_replaces_positive_infinity_with_nan(self, extractor: FeatureExtractorImpl) -> None:
        data = pl.DataFrame(
            {
                "feature1": [1.0, float("inf"), 3.0],
                "target": [0, 1, 0],
            }
        )

        result = extractor.extract_features_and_target(data)

        # Check that inf is replaced with nan
        values = result.X["feature1"].to_list()
        assert values[0] == 1.0
        assert values[1] != values[1]  # NaN != NaN
        assert values[2] == 3.0

    def it_replaces_negative_infinity_with_nan(self, extractor: FeatureExtractorImpl) -> None:
        data = pl.DataFrame(
            {
                "feature1": [1.0, -float("inf"), 3.0],
                "target": [0, 1, 0],
            }
        )

        result = extractor.extract_features_and_target(data)

        values = result.X["feature1"].to_list()
        assert values[1] != values[1]  # NaN != NaN

    def it_preserves_non_float_columns(self, extractor: FeatureExtractorImpl) -> None:
        data = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "str_col": ["a", "b", "c"],
                "float_col": [1.0, 2.0, 3.0],
                "target": [0, 1, 0],
            }
        )

        result = extractor.extract_features_and_target(data)

        assert "int_col" in result.X.columns
        assert "str_col" in result.X.columns
        assert "float_col" in result.X.columns

    def it_raises_error_when_target_column_missing(self, extractor: FeatureExtractorImpl) -> None:
        data = pl.DataFrame(
            {
                "feature1": [1.0, 2.0],
                "feature2": [3.0, 4.0],
                # No target column
            }
        )

        with pytest.raises(ValueError) as exc_info:
            extractor.extract_features_and_target(data)

        assert "target" in str(exc_info.value).lower()

    def it_handles_multiple_float_columns_with_infinity(
        self, extractor: FeatureExtractorImpl
    ) -> None:
        data = pl.DataFrame(
            {
                "col1": [float("inf"), 1.0],
                "col2": [2.0, -float("inf")],
                "target": [0, 1],
            }
        )

        result = extractor.extract_features_and_target(data)

        # Both infinity values should be replaced
        assert result.X["col1"].to_list()[0] != result.X["col1"].to_list()[0]  # NaN
        assert result.X["col2"].to_list()[1] != result.X["col2"].to_list()[1]  # NaN

    def it_preserves_existing_nan_values(self, extractor: FeatureExtractorImpl) -> None:
        data = pl.DataFrame(
            {
                "feature": [1.0, float("nan"), 3.0],
                "target": [0, 1, 0],
            }
        )

        result = extractor.extract_features_and_target(data)

        values = result.X["feature"].to_list()
        assert values[1] != values[1]  # Original NaN preserved
