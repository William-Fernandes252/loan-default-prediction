"""Tests for model_predictions_repository service."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import RawPredictions
from experiments.services.model_predictions_repository import (
    ExecutionNotFoundError,
    ModelPredictionsStorageLayout,
    ModelPredictionsStorageRepository,
)
from experiments.storage.interface import FileInfo

# ============================================================================
# ModelPredictionsStorageLayout Tests
# ============================================================================


class DescribeModelPredictionsStorageLayoutDefaults:
    def it_has_default_predictions_key_template(
        self, predictions_layout: ModelPredictionsStorageLayout
    ) -> None:
        assert "predictions/{execution_id}" in predictions_layout.predictions_key_template

    def it_has_default_predictions_prefix(
        self, predictions_layout: ModelPredictionsStorageLayout
    ) -> None:
        assert predictions_layout.predictions_prefix == "predictions/"


class DescribeGetPredictionsKey:
    def it_formats_key_with_all_components(
        self, predictions_layout: ModelPredictionsStorageLayout
    ) -> None:
        key = predictions_layout.get_predictions_key(
            execution_id="exec-123",
            dataset="taiwan_credit",
            model_type="random_forest",
            technique="smote",
            seed=42,
        )

        expected = "predictions/exec-123/taiwan_credit/random_forest/smote/seed_42.parquet"
        assert key == expected


class DescribeParsePredictionsKey:
    def it_parses_valid_key(self, predictions_layout: ModelPredictionsStorageLayout) -> None:
        key = "predictions/exec-123/taiwan_credit/random_forest/smote/seed_42.parquet"

        result = predictions_layout.parse_predictions_key(key)

        assert result is not None
        assert result["execution_id"] == "exec-123"
        assert result["dataset"] == "taiwan_credit"
        assert result["model_type"] == "random_forest"
        assert result["technique"] == "smote"
        assert result["seed"] == 42

    def it_returns_none_for_invalid_key(
        self, predictions_layout: ModelPredictionsStorageLayout
    ) -> None:
        result = predictions_layout.parse_predictions_key("invalid/path/file.txt")

        assert result is None

    def it_returns_none_for_wrong_extension(
        self, predictions_layout: ModelPredictionsStorageLayout
    ) -> None:
        key = "predictions/exec-123/dataset/model/technique/seed_42.csv"

        result = predictions_layout.parse_predictions_key(key)

        assert result is None


# ============================================================================
# ModelPredictionsStorageRepository Tests
# ============================================================================


class DescribeModelPredictionsStorageRepositoryInit:
    def it_stores_storage_backend(self, mock_storage: MagicMock) -> None:
        repo = ModelPredictionsStorageRepository(storage=mock_storage)

        assert repo._storage is mock_storage

    def it_uses_default_layout_if_not_provided(self, mock_storage: MagicMock) -> None:
        repo = ModelPredictionsStorageRepository(storage=mock_storage)

        assert isinstance(repo._layout, ModelPredictionsStorageLayout)

    def it_uses_provided_layout(self, mock_storage: MagicMock) -> None:
        layout = ModelPredictionsStorageLayout(predictions_prefix="custom/")

        repo = ModelPredictionsStorageRepository(storage=mock_storage, layout=layout)

        assert repo._layout is layout


class DescribeGetCompletedCombinations:
    def it_returns_empty_set_when_no_files(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = []

        result = predictions_repository.get_completed_combinations(execution_id="exec-123")

        assert result == set()

    def it_returns_combinations_for_execution(
        self,
        mock_storage: MagicMock,
        predictions_repository: ModelPredictionsStorageRepository,
        make_predictions_file_info,
    ) -> None:
        mock_storage.list_files.return_value = [make_predictions_file_info()]

        result = predictions_repository.get_completed_combinations(execution_id="exec-123")

        assert len(result) == 1
        combo = next(iter(result))
        assert combo.dataset == Dataset.TAIWAN_CREDIT
        assert combo.model_type == ModelType.RANDOM_FOREST
        assert combo.technique == Technique.BASELINE
        assert combo.seed == 42

    def it_filters_by_execution_id(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/exec-other/taiwan_credit/svm/smote/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]

        result = predictions_repository.get_completed_combinations(execution_id="exec-123")

        assert len(result) == 1

    def it_skips_invalid_enum_values(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/exec-123/invalid_dataset/invalid_model/invalid/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]

        result = predictions_repository.get_completed_combinations(execution_id="exec-123")

        assert len(result) == 1


class DescribeSavePredictions:
    def it_writes_predictions_to_storage(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        predictions = RawPredictions(
            target=np.array([0, 1, 0]),
            prediction=np.array([0, 1, 1]),
        )

        predictions_repository.save_predictions(
            execution_id="exec-123",
            seed=42,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            predictions=predictions,
        )

        mock_storage.sink_parquet.assert_called_once()

    def it_uses_correct_storage_key(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        predictions = RawPredictions(
            target=np.array([0, 1]),
            prediction=np.array([0, 1]),
        )

        predictions_repository.save_predictions(
            execution_id="exec-456",
            seed=100,
            dataset=Dataset.LENDING_CLUB,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            predictions=predictions,
        )

        key = mock_storage.sink_parquet.call_args[0][1]
        assert "exec-456" in key
        assert "lending_club" in key
        assert "svm" in key
        assert "baseline" in key
        assert "seed_100" in key


class DescribeGetLatestPredictionsForExperiment:
    def it_returns_none_when_no_files(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = []

        result = predictions_repository.get_latest_predictions_for_experiment(
            dataset=Dataset.TAIWAN_CREDIT
        )

        assert result is None

    def it_returns_none_when_no_matching_dataset(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/lending_club/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]

        result = predictions_repository.get_latest_predictions_for_experiment(
            dataset=Dataset.TAIWAN_CREDIT
        )

        assert result is None

    def it_returns_predictions_for_latest_execution(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-001/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/exec-002/taiwan_credit/svm/smote/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]
        mock_storage.scan_parquet.return_value = MagicMock()

        result = predictions_repository.get_latest_predictions_for_experiment(
            dataset=Dataset.TAIWAN_CREDIT
        )
        predictions_list = list(result) if result else []

        assert len(predictions_list) == 1
        assert predictions_list[0].execution_id == "exec-002"

    def it_uses_sortable_execution_ids(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/aaa-exec/taiwan_credit/random_forest/baseline/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/zzz-exec/taiwan_credit/svm/baseline/seed_2.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]
        mock_storage.scan_parquet.return_value = MagicMock()

        result = predictions_repository.get_latest_predictions_for_experiment(
            dataset=Dataset.TAIWAN_CREDIT
        )
        predictions_list = list(result) if result else []

        assert len(predictions_list) == 1
        assert predictions_list[0].execution_id == "zzz-exec"


# ============================================================================
# ExecutionNotFoundError Tests
# ============================================================================


class DescribeExecutionNotFoundError:
    def it_stores_execution_id(self) -> None:
        error = ExecutionNotFoundError("exec-123")

        assert error.execution_id == "exec-123"

    def it_stores_dataset_when_provided(self) -> None:
        error = ExecutionNotFoundError("exec-123", dataset="taiwan_credit")

        assert error.dataset == "taiwan_credit"

    def it_has_dataset_none_by_default(self) -> None:
        error = ExecutionNotFoundError("exec-123")

        assert error.dataset is None

    def it_formats_message_without_dataset(self) -> None:
        error = ExecutionNotFoundError("exec-123")

        assert str(error) == "Execution 'exec-123' not found"

    def it_formats_message_with_dataset(self) -> None:
        error = ExecutionNotFoundError("exec-123", dataset="taiwan_credit")

        assert str(error) == "Execution 'exec-123' not found for dataset 'taiwan_credit'"


# ============================================================================
# Get Predictions For Execution Tests
# ============================================================================


class DescribeGetPredictionsForExecution:
    def it_returns_predictions_for_specific_execution(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]
        mock_storage.scan_parquet.return_value = MagicMock()

        result = predictions_repository.get_predictions_for_execution(
            dataset=Dataset.TAIWAN_CREDIT, execution_id="exec-123"
        )
        predictions_list = list(result) if result else []

        assert len(predictions_list) == 1
        assert predictions_list[0].execution_id == "exec-123"

    def it_returns_none_when_no_predictions_for_dataset(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        # First call: no files for the specific dataset
        # Second call: files exist for other datasets (execution exists)
        mock_storage.list_files.side_effect = [
            [],  # No files for taiwan_credit
            [
                FileInfo(
                    key="predictions/exec-123/lending_club/random_forest/baseline/seed_1.parquet",
                    size_bytes=1024,
                    last_modified=None,
                ),
            ],  # Execution exists with other dataset
        ]

        result = predictions_repository.get_predictions_for_execution(
            dataset=Dataset.TAIWAN_CREDIT, execution_id="exec-123"
        )

        assert result is None

    def it_raises_error_when_execution_does_not_exist(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        # Both calls return empty (execution doesn't exist at all)
        mock_storage.list_files.return_value = []

        with pytest.raises(ExecutionNotFoundError) as exc_info:
            predictions_repository.get_predictions_for_execution(
                dataset=Dataset.TAIWAN_CREDIT, execution_id="nonexistent-exec"
            )

        assert exc_info.value.execution_id == "nonexistent-exec"

    def it_returns_all_predictions_for_execution_and_dataset(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/taiwan_credit/random_forest/baseline/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/exec-123/taiwan_credit/svm/smote/seed_2.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]
        mock_storage.scan_parquet.return_value = MagicMock()

        result = predictions_repository.get_predictions_for_execution(
            dataset=Dataset.TAIWAN_CREDIT, execution_id="exec-123"
        )
        predictions_list = list(result) if result else []

        assert len(predictions_list) == 2
        assert all(p.execution_id == "exec-123" for p in predictions_list)

    def it_uses_correct_prefix_for_execution_and_dataset(
        self, mock_storage: MagicMock, predictions_repository: ModelPredictionsStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key="predictions/my-exec-id/lending_club/random_forest/baseline/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]
        mock_storage.scan_parquet.return_value = MagicMock()

        predictions_repository.get_predictions_for_execution(
            dataset=Dataset.LENDING_CLUB, execution_id="my-exec-id"
        )

        # First call should be for specific execution/dataset prefix
        first_call_args = mock_storage.list_files.call_args_list[0]
        assert first_call_args[0][0] == "predictions/my-exec-id/lending_club/"
