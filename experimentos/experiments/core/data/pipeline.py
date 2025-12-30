"""Data processing pipeline orchestrator.

This module provides the DataProcessingPipeline class that coordinates
data loading, transformation, and export in a clean, composable way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

# Import transformers to ensure they are registered
# These imports trigger the @register_transformer decorators
from experiments.core.data import corporate_credit, lending_club, taiwan_credit
from experiments.core.data.base import BaseDataTransformer
from experiments.core.data.exporters import ParquetDataExporter
from experiments.core.data.loaders import CsvRawDataLoader
from experiments.core.data.protocols import ProcessedDataExporter, RawDataLoader
from experiments.core.data.registry import get_transformer_registry

# Silence unused import warnings
_ = (corporate_credit, lending_club, taiwan_credit)

if TYPE_CHECKING:
    from experiments.core.data import Dataset
    from experiments.services.storage import StorageService


class DataProcessingPipeline:
    """Orchestrates the data processing pipeline: load → transform → export.

    This class coordinates the three stages of data processing using
    dependency injection for maximum flexibility and testability.

    Example:
        ```python
        loader = CsvRawDataLoader(storage, raw_data_uri)
        transformer = LendingClubTransformer(use_gpu=True)
        exporter = ParquetDataExporter(storage, interim_uri)

        pipeline = DataProcessingPipeline(
            loader=loader,
            transformer=transformer,
            exporter=exporter,
        )

        output_uri = pipeline.run(Dataset.LENDING_CLUB)
        ```
    """

    def __init__(
        self,
        loader: RawDataLoader,
        transformer: BaseDataTransformer,
        exporter: ProcessedDataExporter,
    ) -> None:
        """Initialize the pipeline.

        Args:
            loader: Raw data loader instance.
            transformer: Data transformer instance.
            exporter: Processed data exporter instance.
        """
        self._loader = loader
        self._transformer = transformer
        self._exporter = exporter

    def run(self, dataset: Dataset) -> str:
        """Execute the pipeline for a single dataset.

        Args:
            dataset: The dataset to process.

        Returns:
            URI to the exported processed data file.
        """
        logger.info(f"Processing dataset {dataset.display_name}...")

        # Load raw data
        logger.info("Loading raw data...")
        raw_df = self._loader.load(dataset)
        logger.info(f"Loaded {len(raw_df):,} rows")

        # Transform
        logger.info("Applying transformations...")
        processed_df = self._transformer.transform(raw_df, dataset)
        logger.info(
            f"Transformation complete: {len(processed_df):,} rows, {len(processed_df.columns)} columns"
        )

        # Export
        logger.info("Exporting processed data...")
        output_uri = self._exporter.export(processed_df, dataset)

        logger.success(f"Processing complete: {output_uri}")
        return output_uri


class DataProcessingPipelineFactory:
    """Factory for creating configured DataProcessingPipeline instances.

    This factory creates pipelines with the appropriate transformer
    for each dataset, handling the wiring of loaders and exporters
    using the storage layer.

    Example:
        ```python
        factory = DataProcessingPipelineFactory(
            storage=storage,
            raw_data_uri="file:///data/raw",
            interim_data_uri="file:///data/interim",
            use_gpu=True,
        )
        pipeline = factory.create(Dataset.TAIWAN_CREDIT)
        pipeline.run(Dataset.TAIWAN_CREDIT)
        ```
    """

    def __init__(
        self,
        storage: StorageService,
        raw_data_uri: str,
        interim_data_uri: str,
        *,
        use_gpu: bool = False,
        transformer_registry: dict[str, type[BaseDataTransformer]] | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            storage: Storage service for file operations.
            raw_data_uri: Base URI for raw data files.
            interim_data_uri: Base URI for interim data files.
            use_gpu: Whether to enable GPU acceleration for transformations.
            transformer_registry: Optional custom transformer registry.
                If None, uses the global registry from the registration system.
        """
        self._storage = storage
        self._raw_data_uri = raw_data_uri
        self._interim_data_uri = interim_data_uri
        self._use_gpu = use_gpu

        # Use provided registry or get from global registration system
        self._transformer_registry = (
            transformer_registry
            if transformer_registry is not None
            else get_transformer_registry()
        )

    def create(self, dataset: Dataset) -> DataProcessingPipeline:
        """Create a pipeline configured for the specified dataset.

        Args:
            dataset: The dataset to create a pipeline for.

        Returns:
            A configured DataProcessingPipeline instance.

        Raises:
            ValueError: If no transformer is registered for the dataset.
        """
        transformer_cls = self._transformer_registry.get(dataset.id)
        if transformer_cls is None:
            raise ValueError(f"No transformer registered for dataset: {dataset.id}")

        loader = CsvRawDataLoader(self._storage, self._raw_data_uri)
        transformer = transformer_cls(use_gpu=self._use_gpu)
        exporter = ParquetDataExporter(self._storage, self._interim_data_uri)

        return DataProcessingPipeline(
            loader=loader,
            transformer=transformer,
            exporter=exporter,
        )
