from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.features import TrainingDataset
from experiments.lib.pipelines import TaskStatus
from experiments.pipelines.data.factory import (
    DataProcessingPipelineFactory,
    DataProcessingPipelineSteps,
    check_already_processed,
    run_if_not_processed,
)
from experiments.pipelines.data.pipeline import (
    DataProcessingPipelineContext,
    DataProcessingPipelineState,
)


class DescribeCheckAlreadyProcessed:
    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        repository = MagicMock()
        repository.is_processed.return_value = False
        return repository

    @pytest.fixture
    def context(self, mock_repository: MagicMock) -> DataProcessingPipelineContext:
        return DataProcessingPipelineContext(
            dataset=Dataset.CORPORATE_CREDIT_RATING,
            data_repository=mock_repository,
            use_gpu=False,
        )

    def it_checks_if_data_is_processed(
        self, mock_repository: MagicMock, context: DataProcessingPipelineContext
    ) -> None:
        state: DataProcessingPipelineState = {}

        result = check_already_processed(state, context)

        mock_repository.is_processed.assert_called_once_with(Dataset.CORPORATE_CREDIT_RATING)
        assert result.status == TaskStatus.SUCCESS

    def it_updates_state_with_processed_flag(self, context: DataProcessingPipelineContext) -> None:
        state: DataProcessingPipelineState = {}

        result = check_already_processed(state, context)

        assert "is_processed" in result.state
        assert result.state["is_processed"] is False

    def it_returns_correct_message_when_not_processed(
        self, context: DataProcessingPipelineContext
    ) -> None:
        state: DataProcessingPipelineState = {}

        result = check_already_processed(state, context)

        assert result.message == "Data not yet processed."

    def it_returns_correct_message_when_already_processed(
        self, context: DataProcessingPipelineContext, mock_repository: MagicMock
    ) -> None:
        mock_repository.is_processed.return_value = True
        state: DataProcessingPipelineState = {}

        result = check_already_processed(state, context)

        assert result.message == "Data already processed."


class DescribeRunIfNotProcessed:
    def it_returns_true_when_not_processed(self) -> None:
        state: DataProcessingPipelineState = {"is_processed": False}
        context = DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=MagicMock(),
            use_gpu=False,
        )

        should_run, reason = run_if_not_processed(state, context)

        assert should_run is True
        assert reason is None

    def it_returns_false_when_already_processed(self) -> None:
        state: DataProcessingPipelineState = {"is_processed": True}
        context = DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=MagicMock(),
            use_gpu=False,
        )

        should_run, reason = run_if_not_processed(state, context)

        assert should_run is False
        assert reason is not None and "already processed" in reason

    def it_returns_true_when_force_overwrite_is_enabled(self) -> None:
        state: DataProcessingPipelineState = {"is_processed": True}
        context = DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=MagicMock(),
            use_gpu=False,
            force_overwrite=True,
        )

        should_run, reason = run_if_not_processed(state, context)

        assert should_run is True
        assert reason is None


class DescribeDataProcessingPipelineFactory:
    @pytest.fixture
    def mock_transformer_registry(self) -> MagicMock:
        registry = MagicMock()
        transformer = MagicMock()
        transformer.return_value = pl.DataFrame({"transformed": [1, 2]})
        registry.get.return_value = transformer
        return registry

    @pytest.fixture
    def mock_feature_extractor(self) -> MagicMock:
        extractor = MagicMock()
        extractor.extract_features_and_target.return_value = TrainingDataset(
            X=pl.DataFrame({"feature": [1, 2]}),
            y=pl.DataFrame({"target": [0, 1]}),
        )
        return extractor

    @pytest.fixture
    def factory(
        self,
        mock_feature_extractor: MagicMock,
        mock_transformer_registry: MagicMock,
    ) -> DataProcessingPipelineFactory:
        return DataProcessingPipelineFactory(
            feature_extractor=mock_feature_extractor,
            transformer_registry=mock_transformer_registry,
        )

    def it_creates_pipeline_with_all_steps(self, factory: DataProcessingPipelineFactory) -> None:
        pipeline = factory.create("test_pipeline")

        assert pipeline.name == "test_pipeline"
        step_names = [step.step.name for step in pipeline.steps]
        assert DataProcessingPipelineSteps.CHECK_ALREADY_PROCESSED.value in step_names
        assert DataProcessingPipelineSteps.LOAD_RAW_DATA.value in step_names
        assert DataProcessingPipelineSteps.TRANSFORM_DATA.value in step_names
        assert DataProcessingPipelineSteps.EXPORT_PROCESSED_DATA.value in step_names
        assert DataProcessingPipelineSteps.EXTRACT_FINAL_FEATURES.value in step_names
        assert DataProcessingPipelineSteps.EXPORT_FINAL_FEATURES.value in step_names

    def it_creates_pipeline_with_conditional_steps(
        self, factory: DataProcessingPipelineFactory
    ) -> None:
        pipeline = factory.create()

        conditional_steps = [step for step in pipeline.steps if step.condition is not None]
        assert len(conditional_steps) == 5  # All steps except CHECK_ALREADY_PROCESSED

    def it_transformer_task_gets_transformer_from_registry(
        self,
        factory: DataProcessingPipelineFactory,
        mock_transformer_registry: MagicMock,
    ) -> None:
        pipeline = factory.create()
        transform_step = next(
            step
            for step in pipeline.steps
            if step.step.name == DataProcessingPipelineSteps.TRANSFORM_DATA.value
        )

        context = DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=MagicMock(),
            use_gpu=True,
        )
        state: DataProcessingPipelineState = {"raw_data": pl.LazyFrame({"col": [1, 2]})}

        result = transform_step.step.task(state, context)

        mock_transformer_registry.get.assert_called_once_with(Dataset.TAIWAN_CREDIT)
        assert result.status == TaskStatus.SUCCESS

    def it_transformer_task_uses_gpu_flag(
        self,
        factory: DataProcessingPipelineFactory,
        mock_transformer_registry: MagicMock,
    ) -> None:
        pipeline = factory.create()
        transform_step = next(
            step
            for step in pipeline.steps
            if step.step.name == DataProcessingPipelineSteps.TRANSFORM_DATA.value
        )

        mock_transformer = mock_transformer_registry.get.return_value
        context = DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=MagicMock(),
            use_gpu=True,
        )
        state: DataProcessingPipelineState = {"raw_data": pl.LazyFrame({"col": [1, 2]})}

        transform_step.step.task(state, context)

        mock_transformer.assert_called_once()
        assert mock_transformer.call_args[0][1] is True  # use_gpu=True

    def it_transformer_task_returns_failure_when_no_transformer(
        self, mock_feature_extractor: MagicMock
    ) -> None:
        registry = MagicMock()
        registry.get.return_value = None
        factory = DataProcessingPipelineFactory(
            feature_extractor=mock_feature_extractor,
            transformer_registry=registry,
        )

        pipeline = factory.create()
        transform_step = next(
            step
            for step in pipeline.steps
            if step.step.name == DataProcessingPipelineSteps.TRANSFORM_DATA.value
        )

        context = DataProcessingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            data_repository=MagicMock(),
            use_gpu=False,
        )
        state: DataProcessingPipelineState = {"raw_data": pl.LazyFrame({"col": [1, 2]})}

        with pytest.raises(ValueError, match="No transformer registered"):
            transform_step.step.task(state, context)

    def it_feature_extractor_task_extracts_features(
        self,
        factory: DataProcessingPipelineFactory,
        mock_feature_extractor: MagicMock,
    ) -> None:
        pipeline = factory.create()
        extract_step = next(
            step
            for step in pipeline.steps
            if step.step.name == DataProcessingPipelineSteps.EXTRACT_FINAL_FEATURES.value
        )

        context = DataProcessingPipelineContext(
            dataset=Dataset.LENDING_CLUB,
            data_repository=MagicMock(),
            use_gpu=False,
        )
        interim_data = pl.DataFrame({"col": [1, 2]})
        state: DataProcessingPipelineState = {"interim_data": interim_data}

        result = extract_step.step.task(state, context)

        mock_feature_extractor.extract_features_and_target.assert_called_once_with(interim_data)
        assert result.status == TaskStatus.SUCCESS
        assert "X_final" in result.state
        assert "y_final" in result.state

    def it_feature_extractor_task_returns_failure_when_no_interim_data(
        self, factory: DataProcessingPipelineFactory
    ) -> None:
        pipeline = factory.create()
        extract_step = next(
            step
            for step in pipeline.steps
            if step.step.name == DataProcessingPipelineSteps.EXTRACT_FINAL_FEATURES.value
        )

        context = DataProcessingPipelineContext(
            dataset=Dataset.LENDING_CLUB,
            data_repository=MagicMock(),
            use_gpu=False,
        )
        state: DataProcessingPipelineState = {"interim_data": None}  # type: ignore[typeddict-item]

        result = extract_step.step.task(state, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "No interim data found" in result.message
