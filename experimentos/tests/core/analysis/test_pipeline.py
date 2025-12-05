"""Tests for experiments.core.analysis.pipeline module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiments.core.analysis.pipeline import (
    AnalysisPipeline,
    AnalysisPipelineFactory,
)
from experiments.core.data import Dataset


class MockOutputPathProvider:
    """Mock implementation of OutputPathProvider."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def get_output_dir(self, dataset_id: str, is_figure: bool = True) -> Path:
        subdir = "figures" if is_figure else ""
        return self._base_path / dataset_id / subdir


class MockResultsPathProvider:
    """Mock implementation of ResultsPathProvider."""

    def get_results_file(self, dataset_id: str) -> Path:
        return Path(f"/mock/results/{dataset_id}.parquet")


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "model": ["RF", "XGB"],
            "technique": ["baseline", "smote"],
            "f1_score": [0.85, 0.88],
            "precision": [0.82, 0.86],
        }
    )


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame for testing."""
    return pd.DataFrame()


@pytest.fixture
def mock_loader(sample_dataframe: pd.DataFrame) -> MagicMock:
    """Create a mock loader."""
    loader = MagicMock()
    loader.load.return_value = sample_dataframe
    return loader


@pytest.fixture
def mock_loader_empty(empty_dataframe: pd.DataFrame) -> MagicMock:
    """Create a mock loader that returns empty data."""
    loader = MagicMock()
    loader.load.return_value = empty_dataframe
    return loader


@pytest.fixture
def mock_transformer() -> MagicMock:
    """Create a mock transformer."""
    transformer = MagicMock()
    # Return transformed data as a dict (typical transformer output)
    transformer.transform.return_value = {"main": pd.DataFrame({"result": [1, 2]})}
    return transformer


@pytest.fixture
def mock_exporter() -> MagicMock:
    """Create a mock exporter."""
    exporter = MagicMock()
    exporter.export.return_value = [
        Path("/output/figure1.png"),
        Path("/output/figure2.png"),
    ]
    return exporter


@pytest.fixture
def mock_translate() -> MagicMock:
    """Create a mock translation function."""
    return MagicMock(side_effect=lambda x: x)


class DescribeOutputPathProvider:
    """Tests for the OutputPathProvider protocol."""

    def it_defines_get_output_dir_method(self) -> None:
        """Verify OutputPathProvider is a valid protocol."""
        provider = MockOutputPathProvider(Path("/base"))

        # Should implement the protocol
        result = provider.get_output_dir("taiwan_credit", is_figure=True)

        assert isinstance(result, Path)
        assert result == Path("/base/taiwan_credit/figures")

    def it_handles_non_figure_output(self) -> None:
        """Verify get_output_dir works for non-figure output."""
        provider = MockOutputPathProvider(Path("/base"))

        result = provider.get_output_dir("taiwan_credit", is_figure=False)

        assert result == Path("/base/taiwan_credit/")


class DescribeAnalysisPipeline:
    """Tests for AnalysisPipeline class."""

    def it_initializes_with_required_dependencies(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
    ) -> None:
        """Verify pipeline initializes correctly with all dependencies."""
        output_provider = MockOutputPathProvider(Path("/output"))

        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        assert pipeline._loader is mock_loader
        assert pipeline._transformer is mock_transformer
        assert pipeline._exporter is mock_exporter
        assert pipeline._is_figure_output is True

    def it_accepts_is_figure_output_parameter(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
    ) -> None:
        """Verify is_figure_output parameter is stored correctly."""
        output_provider = MockOutputPathProvider(Path("/output"))

        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
            is_figure_output=False,
        )

        assert pipeline._is_figure_output is False


class DescribeAnalysisPipelineRun:
    """Tests for AnalysisPipeline.run() method."""

    def it_executes_load_transform_export_sequence(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        sample_dataframe: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify run() calls loader, transformer, and exporter in sequence."""
        output_provider = MockOutputPathProvider(tmp_path)
        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        result = pipeline.run(Dataset.TAIWAN_CREDIT)

        # Verify loader was called with correct dataset
        mock_loader.load.assert_called_once_with(Dataset.TAIWAN_CREDIT)

        # Verify transformer was called with loaded data
        mock_transformer.transform.assert_called_once()
        call_args = mock_transformer.transform.call_args
        pd.testing.assert_frame_equal(call_args[0][0], sample_dataframe)
        assert call_args[0][1] == Dataset.TAIWAN_CREDIT

        # Verify exporter was called
        mock_exporter.export.assert_called_once()

        # Verify return value matches exporter output
        assert result == mock_exporter.export.return_value

    def it_returns_empty_list_when_no_data(
        self,
        mock_loader_empty: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify run() returns empty list when loader returns no data."""
        output_provider = MockOutputPathProvider(tmp_path)
        pipeline = AnalysisPipeline(
            loader=mock_loader_empty,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        result = pipeline.run(Dataset.TAIWAN_CREDIT)

        assert result == []
        # Transformer and exporter should not be called
        mock_transformer.transform.assert_not_called()
        mock_exporter.export.assert_not_called()

    def it_creates_output_directory_if_not_exists(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify run() creates output directory when it doesn't exist."""
        output_provider = MockOutputPathProvider(tmp_path)
        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        # Output directory should not exist initially
        expected_dir = tmp_path / "taiwan_credit" / "figures"
        assert not expected_dir.exists()

        pipeline.run(Dataset.TAIWAN_CREDIT)

        # Output directory should be created
        assert expected_dir.exists()

    def it_passes_correct_output_directory_to_exporter(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify run() passes correct output directory to exporter."""
        output_provider = MockOutputPathProvider(tmp_path)
        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        pipeline.run(Dataset.TAIWAN_CREDIT)

        # Verify exporter received correct output directory
        call_args = mock_exporter.export.call_args
        output_dir = call_args[0][1]
        assert output_dir == tmp_path / "taiwan_credit" / "figures"


class DescribeAnalysisPipelineRunAll:
    """Tests for AnalysisPipeline.run_all() method."""

    def it_runs_for_all_datasets_when_none_specified(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify run_all() processes all datasets when no list provided."""
        output_provider = MockOutputPathProvider(tmp_path)
        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        result = pipeline.run_all()

        # Should have called loader for each dataset
        assert mock_loader.load.call_count == len(list(Dataset))

        # Result should contain all dataset IDs
        for dataset in Dataset:
            assert dataset.id in result

    def it_runs_only_for_specified_datasets(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify run_all() processes only specified datasets."""
        output_provider = MockOutputPathProvider(tmp_path)
        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        datasets = [Dataset.TAIWAN_CREDIT]
        result = pipeline.run_all(datasets)

        # Should only call loader once
        assert mock_loader.load.call_count == 1
        mock_loader.load.assert_called_with(Dataset.TAIWAN_CREDIT)

        # Result should only contain specified dataset
        assert len(result) == 1
        assert Dataset.TAIWAN_CREDIT.id in result

    def it_returns_dictionary_mapping_dataset_ids_to_paths(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify run_all() returns correct dictionary structure."""
        output_provider = MockOutputPathProvider(tmp_path)
        pipeline = AnalysisPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
            output_path_provider=output_provider,
        )

        datasets = [Dataset.TAIWAN_CREDIT]
        result = pipeline.run_all(datasets)

        assert isinstance(result, dict)
        assert result[Dataset.TAIWAN_CREDIT.id] == mock_exporter.export.return_value


class DescribeAnalysisPipelineFactory:
    """Tests for AnalysisPipelineFactory class."""

    def it_initializes_with_providers_and_translate(
        self,
        mock_translate: MagicMock,
    ) -> None:
        """Verify factory initializes correctly with dependencies."""
        path_provider = MockResultsPathProvider()
        output_provider = MockOutputPathProvider(Path("/output"))

        factory = AnalysisPipelineFactory(
            path_provider=path_provider,  # type: ignore[arg-type]
            output_path_provider=output_provider,
            translate=mock_translate,
        )

        assert factory._path_provider is path_provider
        assert factory._output_path_provider is output_provider
        assert factory._translate is mock_translate


class DescribeAnalysisPipelineFactoryCreateMethods:
    """Tests for AnalysisPipelineFactory create_* methods."""

    @pytest.fixture
    def factory(self, mock_translate: MagicMock) -> AnalysisPipelineFactory:
        """Create a factory for testing."""
        path_provider = MockResultsPathProvider()
        output_provider = MockOutputPathProvider(Path("/output"))
        return AnalysisPipelineFactory(
            path_provider=path_provider,  # type: ignore[arg-type]
            output_path_provider=output_provider,
            translate=mock_translate,
        )

    def it_creates_stability_pipeline(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create_stability_pipeline returns correct pipeline."""
        with (
            patch(
                "experiments.core.analysis.transformers.StabilityTransformer"
            ) as mock_transformer_class,
            patch(
                "experiments.core.analysis.exporters.StabilityFigureExporter"
            ) as mock_exporter_class,
        ):
            pipeline = factory.create_stability_pipeline()

            assert isinstance(pipeline, AnalysisPipeline)
            mock_transformer_class.assert_called_once()
            mock_exporter_class.assert_called_once()

    def it_creates_risk_tradeoff_pipeline(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create_risk_tradeoff_pipeline returns correct pipeline."""
        with (
            patch(
                "experiments.core.analysis.transformers.RiskTradeoffTransformer"
            ) as mock_transformer_class,
            patch(
                "experiments.core.analysis.exporters.RiskTradeoffFigureExporter"
            ) as mock_exporter_class,
        ):
            pipeline = factory.create_risk_tradeoff_pipeline()

            assert isinstance(pipeline, AnalysisPipeline)
            mock_transformer_class.assert_called_once()
            mock_exporter_class.assert_called_once()

    def it_creates_imbalance_impact_pipeline(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create_imbalance_impact_pipeline returns correct pipeline."""
        with (
            patch(
                "experiments.core.analysis.transformers.ImbalanceImpactTransformer"
            ) as mock_transformer_class,
            patch(
                "experiments.core.analysis.exporters.ImbalanceImpactFigureExporter"
            ) as mock_exporter_class,
        ):
            pipeline = factory.create_imbalance_impact_pipeline()

            assert isinstance(pipeline, AnalysisPipeline)
            mock_transformer_class.assert_called_once()
            mock_exporter_class.assert_called_once()

    def it_creates_cost_sensitive_vs_resampling_pipeline(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create_cost_sensitive_vs_resampling_pipeline returns correct pipeline."""
        with (
            patch(
                "experiments.core.analysis.transformers.CostSensitiveVsResamplingTransformer"
            ) as mock_transformer_class,
            patch(
                "experiments.core.analysis.exporters.CostSensitiveVsResamplingFigureExporter"
            ) as mock_exporter_class,
        ):
            pipeline = factory.create_cost_sensitive_vs_resampling_pipeline()

            assert isinstance(pipeline, AnalysisPipeline)
            mock_transformer_class.assert_called_once()
            mock_exporter_class.assert_called_once()

    def it_creates_hyperparameter_pipeline(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create_hyperparameter_pipeline returns correct pipeline."""
        with (
            patch(
                "experiments.core.analysis.transformers.HyperparameterTransformer"
            ) as mock_transformer_class,
            patch(
                "experiments.core.analysis.exporters.HyperparameterFigureExporter"
            ) as mock_exporter_class,
        ):
            pipeline = factory.create_hyperparameter_pipeline()

            assert isinstance(pipeline, AnalysisPipeline)
            mock_transformer_class.assert_called_once()
            mock_exporter_class.assert_called_once()

    def it_creates_experiment_summary_pipeline_with_composite_exporter(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create_experiment_summary_pipeline uses CompositeExporter."""
        with (
            patch(
                "experiments.core.analysis.transformers.ExperimentSummaryTransformer"
            ) as mock_transformer_class,
            patch("experiments.core.analysis.exporters.CompositeExporter") as mock_composite_class,
            patch("experiments.core.analysis.exporters.ExperimentSummaryFigureExporter"),
            patch("experiments.core.analysis.exporters.CsvExporter"),
            patch("experiments.core.analysis.exporters.LatexExporter"),
        ):
            pipeline = factory.create_experiment_summary_pipeline()

            assert isinstance(pipeline, AnalysisPipeline)
            mock_transformer_class.assert_called_once()
            mock_composite_class.assert_called_once()
            # Verify is_figure_output is False for summary pipeline
            assert pipeline._is_figure_output is False

    def it_creates_loader_using_path_provider(
        self,
        mock_translate: MagicMock,
    ) -> None:
        """Verify _create_loader uses the path provider."""
        path_provider = MockResultsPathProvider()
        output_provider = MockOutputPathProvider(Path("/output"))
        factory = AnalysisPipelineFactory(
            path_provider=path_provider,  # type: ignore[arg-type]
            output_path_provider=output_provider,
            translate=mock_translate,
        )

        with patch("experiments.core.analysis.pipeline.ParquetResultsLoader") as mock_loader_class:
            factory._create_loader()

            mock_loader_class.assert_called_once_with(path_provider, mock_translate)


class DescribeAnalysisPipelineFactoryRegistryPattern:
    """Tests for the registry pattern implementation in AnalysisPipelineFactory."""

    @pytest.fixture
    def factory(self, mock_translate: MagicMock) -> AnalysisPipelineFactory:
        """Create a factory for testing."""
        from experiments.core.analysis.pipeline import AnalysisPipelineFactory

        path_provider = MockResultsPathProvider()
        output_provider = MockOutputPathProvider(Path("/output"))
        return AnalysisPipelineFactory(
            path_provider=path_provider,  # type: ignore[arg-type]
            output_path_provider=output_provider,
            translate=mock_translate,
        )

    def it_creates_pipeline_using_analysis_type_enum(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create() method works with AnalysisType enum."""
        from experiments.core.analysis.pipeline import AnalysisType

        # Test creating each type
        for analysis_type in AnalysisType:
            pipeline = factory.create(analysis_type)
            assert isinstance(pipeline, AnalysisPipeline)

    def it_raises_value_error_for_unknown_analysis_type(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify create() raises ValueError for unknown analysis type."""
        from experiments.core.analysis.pipeline import AnalysisType

        # Create a mock analysis type that's not in the registry
        # We'll just use a string instead of an Enum
        with pytest.raises(ValueError, match="Unknown analysis type"):
            factory.create("invalid_type")  # type: ignore[arg-type]

    def it_registers_custom_pipeline_factory(
        self,
        factory: AnalysisPipelineFactory,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
    ) -> None:
        """Verify register() method allows adding custom pipeline factories."""
        from experiments.core.analysis.pipeline import AnalysisType

        output_provider = MockOutputPathProvider(Path("/output"))

        # Create a custom factory function
        def custom_factory() -> AnalysisPipeline:
            return AnalysisPipeline(
                loader=mock_loader,
                transformer=mock_transformer,
                exporter=mock_exporter,
                output_path_provider=output_provider,
            )

        # Register the custom factory
        factory.register(AnalysisType.STABILITY, custom_factory)

        # Verify it's used when creating the pipeline
        pipeline = factory.create(AnalysisType.STABILITY)

        assert isinstance(pipeline, AnalysisPipeline)
        assert pipeline._loader is mock_loader
        assert pipeline._transformer is mock_transformer
        assert pipeline._exporter is mock_exporter

    def it_creates_same_pipeline_as_legacy_methods(
        self,
        factory: AnalysisPipelineFactory,
    ) -> None:
        """Verify new create() method produces same result as legacy methods."""
        from experiments.core.analysis.pipeline import AnalysisType

        # Test each analysis type
        legacy_pipeline = factory.create_stability_pipeline()
        new_pipeline = factory.create(AnalysisType.STABILITY)

        # Both should be AnalysisPipeline instances
        assert isinstance(legacy_pipeline, AnalysisPipeline)
        assert isinstance(new_pipeline, AnalysisPipeline)

        # Both should use the same transformer and exporter types
        assert type(legacy_pipeline._transformer) == type(new_pipeline._transformer)
        assert type(legacy_pipeline._exporter) == type(new_pipeline._exporter)
        assert legacy_pipeline._is_figure_output == new_pipeline._is_figure_output
