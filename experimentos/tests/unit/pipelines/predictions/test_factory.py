from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.training.data import TrainingData
from experiments.core.training.splitters import SplitData
from experiments.core.training.trainers import TrainedModel
from experiments.lib.pipelines import TaskStatus
from experiments.pipelines.predictions.factory import (
    PredictionsPipelineFactory,
    load_training_data,
    predict,
    retrieve_trained_model,
    split_data,
)
from experiments.pipelines.predictions.pipeline import (
    PredictionsPipelineContext,
    PredictionsPipelineState,
)


class DescribeRetrieveTrainedModel:
    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        loader = MagicMock()
        mock_model = MagicMock()
        loader.load_model.return_value = TrainedModel(
            model=mock_model,
            params={"max_depth": 10},
            seed=42,
        )
        return loader

    @pytest.fixture
    def context(self, mock_model_loader: MagicMock) -> PredictionsPipelineContext:
        return PredictionsPipelineContext(
            model_id="test-model-123",
            dataset=Dataset.TAIWAN_CREDIT,
            training_data_loader=MagicMock(),
            trained_model_loader=mock_model_loader,
            data_splitter=MagicMock(),
            seed=42,
        )

    def it_retrieves_trained_model_from_loader(
        self, context: PredictionsPipelineContext, mock_model_loader: MagicMock
    ) -> None:
        state: PredictionsPipelineState = {}

        result = retrieve_trained_model(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_model_loader.load_model.assert_called_once_with(
            Dataset.TAIWAN_CREDIT, "test-model-123"
        )

    def it_updates_state_with_trained_model(self, context: PredictionsPipelineContext) -> None:
        state: PredictionsPipelineState = {}

        result = retrieve_trained_model(state, context)

        assert "trained_model" in result.state
        assert isinstance(result.state["trained_model"], TrainedModel)

    def it_returns_success_message(self, context: PredictionsPipelineContext) -> None:
        state: PredictionsPipelineState = {}

        result = retrieve_trained_model(state, context)

        assert result.message == "Trained model retrieved successfully."


class DescribeLoadTrainingData:
    @pytest.fixture
    def mock_data_loader(self) -> MagicMock:
        loader = MagicMock()
        loader.load_training_data.return_value = TrainingData(
            X=MagicMock(),
            y=MagicMock(),
        )
        return loader

    @pytest.fixture
    def context(self, mock_data_loader: MagicMock) -> PredictionsPipelineContext:
        return PredictionsPipelineContext(
            model_id=None,
            dataset=Dataset.LENDING_CLUB,
            training_data_loader=mock_data_loader,
            trained_model_loader=MagicMock(),
            data_splitter=MagicMock(),
            seed=123,
        )

    def it_loads_training_data_from_loader(
        self, context: PredictionsPipelineContext, mock_data_loader: MagicMock
    ) -> None:
        state: PredictionsPipelineState = {}

        result = load_training_data(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_data_loader.load_training_data.assert_called_once_with(Dataset.LENDING_CLUB)

    def it_updates_state_with_training_data(self, context: PredictionsPipelineContext) -> None:
        state: PredictionsPipelineState = {}

        result = load_training_data(state, context)

        assert "training_data" in result.state
        assert isinstance(result.state["training_data"], TrainingData)

    def it_returns_success_message(self, context: PredictionsPipelineContext) -> None:
        state: PredictionsPipelineState = {}

        result = load_training_data(state, context)

        assert result.message == "Training data loaded successfully."


class DescribeSplitData:
    @pytest.fixture
    def mock_splitter(self) -> MagicMock:
        splitter = MagicMock()
        splitter.split.return_value = SplitData(
            X_train=np.array([[1, 2], [3, 4]]),
            X_test=np.array([[5, 6]]),
            y_train=np.array([0, 1]),
            y_test=np.array([1]),
        )
        return splitter

    @pytest.fixture
    def context(self, mock_splitter: MagicMock) -> PredictionsPipelineContext:
        return PredictionsPipelineContext(
            model_id="model-456",
            dataset=Dataset.CORPORATE_CREDIT_RATING,
            training_data_loader=MagicMock(),
            trained_model_loader=MagicMock(),
            data_splitter=mock_splitter,
            seed=99,
        )

    def it_splits_data_using_splitter(
        self, context: PredictionsPipelineContext, mock_splitter: MagicMock
    ) -> None:
        training_data = TrainingData(X=MagicMock(), y=MagicMock())
        state: PredictionsPipelineState = {"training_data": training_data}

        result = split_data(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_splitter.split.assert_called_once_with(training_data, seed=99)

    def it_updates_state_with_split_data(self, context: PredictionsPipelineContext) -> None:
        state: PredictionsPipelineState = {
            "training_data": TrainingData(X=MagicMock(), y=MagicMock())
        }

        result = split_data(state, context)

        assert "data_split" in result.state
        assert isinstance(result.state["data_split"], SplitData)

    def it_returns_success_message(self, context: PredictionsPipelineContext) -> None:
        state: PredictionsPipelineState = {
            "training_data": TrainingData(X=MagicMock(), y=MagicMock())
        }

        result = split_data(state, context)

        assert result.message == "Data split successfully."


class DescribePredict:
    @pytest.fixture
    def context(self) -> PredictionsPipelineContext:
        return PredictionsPipelineContext(
            model_id=None,
            dataset=Dataset.TAIWAN_CREDIT,
            training_data_loader=MagicMock(),
            trained_model_loader=MagicMock(),
            data_splitter=MagicMock(),
            seed=42,
        )

    def it_makes_predictions_using_model(self, context: PredictionsPipelineContext) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])

        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        state: PredictionsPipelineState = {
            "trained_model": TrainedModel(
                model=mock_model,
                params={},
                seed=42,
            ),
            "data_split": SplitData(
                X_train=np.array([[1, 2]]),
                X_test=X_test,
                y_train=np.array([0]),
                y_test=np.array([1, 0, 1]),
            ),
        }

        result = predict(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_model.predict.assert_called_once()
        np.testing.assert_array_equal(mock_model.predict.call_args[0][0], X_test)

    def it_updates_state_with_predictions(self, context: PredictionsPipelineContext) -> None:
        mock_model = MagicMock()
        predictions_array = np.array([1, 0])
        mock_model.predict.return_value = predictions_array

        state: PredictionsPipelineState = {
            "trained_model": TrainedModel(
                model=mock_model,
                params={},
                seed=123,
            ),
            "data_split": SplitData(
                X_train=np.array([[1]]),
                X_test=np.array([[2], [3]]),
                y_train=np.array([0]),
                y_test=np.array([1, 0]),
            ),
        }

        result = predict(state, context)

        assert "predictions" in result.state
        assert isinstance(result.state["predictions"], np.ndarray)
        np.testing.assert_array_equal(result.state["predictions"], predictions_array)

    def it_returns_success_message(self, context: PredictionsPipelineContext) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])

        state: PredictionsPipelineState = {
            "trained_model": TrainedModel(
                model=mock_model,
                params={},
                seed=42,
            ),
            "data_split": SplitData(
                X_train=np.array([[1]]),
                X_test=np.array([[2]]),
                y_train=np.array([0]),
                y_test=np.array([1]),
            ),
        }

        result = predict(state, context)

        assert result.message == "Predictions made successfully."


class DescribePredictionsPipelineFactory:
    @pytest.fixture
    def factory(self) -> PredictionsPipelineFactory:
        return PredictionsPipelineFactory()

    def it_creates_pipeline_with_correct_name(self, factory: PredictionsPipelineFactory) -> None:
        pipeline = factory.create_pipeline("TestPredictions")

        assert pipeline.name == "TestPredictions"

    def it_creates_pipeline_with_all_steps(self, factory: PredictionsPipelineFactory) -> None:
        pipeline = factory.create_pipeline()

        step_names = [step.step.name for step in pipeline.steps]
        assert "RetrieveTrainedModel" in step_names
        assert "LoadTrainingData" in step_names
        assert "SplitData" in step_names
        assert "Predict" in step_names
        assert len(step_names) == 4

    def it_creates_pipeline_with_steps_in_order(self, factory: PredictionsPipelineFactory) -> None:
        pipeline = factory.create_pipeline()

        step_names = [step.step.name for step in pipeline.steps]
        assert step_names == [
            "RetrieveTrainedModel",
            "LoadTrainingData",
            "SplitData",
            "Predict",
        ]

    def it_creates_pipeline_without_conditional_steps(
        self, factory: PredictionsPipelineFactory
    ) -> None:
        pipeline = factory.create_pipeline()

        for step in pipeline.steps:
            assert step.condition is None
