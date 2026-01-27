from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.data.datasets import Dataset
from experiments.lib.pipelines import TaskStatus
from experiments.pipelines.data.loaders import load_raw_data_from_csv
from experiments.pipelines.data.pipeline import (
    DataProcessingPipelineContext,
    DataProcessingPipelineState,
)


class DescribeLoadRawDataFromCsv:
    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        repository = MagicMock()
        repository.get_raw_data.return_value = pl.LazyFrame({"col1": [1, 2], "col2": [3, 4]})
        return repository

    @pytest.fixture
    def context(self, mock_repository: MagicMock) -> DataProcessingPipelineContext:
        return DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=mock_repository,
            use_gpu=False,
        )

    @pytest.fixture
    def state(self) -> DataProcessingPipelineState:
        return DataProcessingPipelineState()

    def it_loads_raw_data_from_repository(
        self,
        state: DataProcessingPipelineState,
        context: DataProcessingPipelineContext,
        mock_repository: MagicMock,
    ) -> None:
        result = load_raw_data_from_csv(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_repository.get_raw_data.assert_called_once_with(Dataset.TAIWAN_CREDIT)

    def it_updates_state_with_raw_data(
        self,
        state: DataProcessingPipelineState,
        context: DataProcessingPipelineContext,
    ) -> None:
        result = load_raw_data_from_csv(state, context)

        assert "raw_data" in result.state
        assert isinstance(result.state["raw_data"], pl.LazyFrame)

    def it_returns_success_message(
        self,
        state: DataProcessingPipelineState,
        context: DataProcessingPipelineContext,
    ) -> None:
        result = load_raw_data_from_csv(state, context)

        assert result.message == "Raw data loaded successfully."
