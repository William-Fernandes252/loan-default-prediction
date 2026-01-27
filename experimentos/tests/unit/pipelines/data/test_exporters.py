from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.data.datasets import Dataset
from experiments.lib.pipelines import TaskStatus
from experiments.pipelines.data.exporters import (
    export_final_features_as_parquet,
    export_processed_data_as_parquet,
)
from experiments.pipelines.data.pipeline import (
    DataProcessingPipelineContext,
    DataProcessingPipelineState,
)


class DescribeExportProcessedDataAsParquet:
    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def context(self, mock_repository: MagicMock) -> DataProcessingPipelineContext:
        return DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=mock_repository,
            use_gpu=False,
        )

    def it_exports_interim_data_to_repository(
        self, context: DataProcessingPipelineContext, mock_repository: MagicMock
    ) -> None:
        interim_data = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        state: DataProcessingPipelineState = {"interim_data": interim_data}

        result = export_processed_data_as_parquet(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_repository.save_interim_data.assert_called_once_with(
            Dataset.TAIWAN_CREDIT, interim_data
        )

    def it_returns_failure_when_no_interim_data(
        self, context: DataProcessingPipelineContext
    ) -> None:
        state: DataProcessingPipelineState = {"interim_data": None}  # type: ignore[typeddict-item]

        result = export_processed_data_as_parquet(state, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "No interim data found" in result.message

    def it_returns_success_message(self, context: DataProcessingPipelineContext) -> None:
        state: DataProcessingPipelineState = {"interim_data": pl.DataFrame({"col": [1]})}

        result = export_processed_data_as_parquet(state, context)

        assert result.message == "Processed data exported successfully."


class DescribeExportFinalFeaturesAsParquet:
    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def context(self, mock_repository: MagicMock) -> DataProcessingPipelineContext:
        return DataProcessingPipelineContext(
            dataset=Dataset.LENDING_CLUB,
            data_repository=mock_repository,
            use_gpu=False,
        )

    def it_exports_final_features_to_repository(
        self, context: DataProcessingPipelineContext, mock_repository: MagicMock
    ) -> None:
        X_final = pl.DataFrame({"feature1": [1, 2]})
        y_final = pl.DataFrame({"target": [0, 1]})
        state: DataProcessingPipelineState = {"X_final": X_final, "y_final": y_final}

        result = export_final_features_as_parquet(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_repository.save_final_features.assert_called_once_with(
            Dataset.LENDING_CLUB, X_final, y_final
        )

    def it_returns_failure_when_no_X_final(self, context: DataProcessingPipelineContext) -> None:
        state: DataProcessingPipelineState = {
            "X_final": None,  # type: ignore[typeddict-item]
            "y_final": pl.DataFrame({"target": [1]}),
        }

        result = export_final_features_as_parquet(state, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "No final features found" in result.message

    def it_returns_failure_when_no_y_final(self, context: DataProcessingPipelineContext) -> None:
        state: DataProcessingPipelineState = {
            "X_final": pl.DataFrame({"col": [1]}),
            "y_final": None,  # type: ignore[typeddict-item]
        }

        result = export_final_features_as_parquet(state, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "No final features found" in result.message

    def it_returns_success_message(self, context: DataProcessingPipelineContext) -> None:
        state: DataProcessingPipelineState = {
            "X_final": pl.DataFrame({"col": [1]}),
            "y_final": pl.DataFrame({"target": [0]}),
        }

        result = export_final_features_as_parquet(state, context)

        assert result.message == "Final features exported successfully."
