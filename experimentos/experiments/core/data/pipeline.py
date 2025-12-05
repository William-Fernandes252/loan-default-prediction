"""Data processing pipeline orchestrator.

This module provides the DataProcessingPipeline class that coordinates
data loading, transformation, and export in a clean, composable way.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from experiments.core.data.base import BaseDataTransformer
from experiments.core.data.corporate_credit import CorporateCreditTransformer
from experiments.core.data.exporters import ParquetDataExporter
from experiments.core.data.lending_club import LendingClubTransformer
from experiments.core.data.loaders import CsvRawDataLoader
from experiments.core.data.protocols import (
    InterimDataPathProvider,
    ProcessedDataExporter,
    RawDataLoader,
    RawDataPathProvider,
)
from experiments.core.data.taiwan_credit import TaiwanCreditTransformer

if TYPE_CHECKING:
    from experiments.core.data import Dataset


class DataPathProvider(RawDataPathProvider, InterimDataPathProvider, Protocol):
    """Combined protocol for providing both raw and interim data paths.

    This protocol combines RawDataPathProvider and InterimDataPathProvider
    for use by the DataProcessingPipelineFactory.
    """

    ...


class DataProcessingPipeline:
    """Orchestrates the data processing pipeline: load → transform → export.

    This class coordinates the three stages of data processing using
    dependency injection for maximum flexibility and testability.

    Example:
        ```python
        loader = CsvRawDataLoader(context)
        transformer = LendingClubTransformer(use_gpu=True)
        exporter = ParquetDataExporter(context)

        pipeline = DataProcessingPipeline(
            loader=loader,
            transformer=transformer,
            exporter=exporter,
        )

        output_path = pipeline.run(Dataset.LENDING_CLUB)
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

    def run(self, dataset: Dataset) -> Path:
        """Execute the pipeline for a single dataset.

        Args:
            dataset: The dataset to process.

        Returns:
            Path to the exported processed data file.
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
        output_path = self._exporter.export(processed_df, dataset)

        logger.success(f"Processing complete: {output_path}")
        return output_path


class DataProcessingPipelineFactory:
    """Factory for creating configured DataProcessingPipeline instances.

    This factory creates pipelines with the appropriate transformer
    for each dataset, handling the wiring of loaders and exporters.

    Example:
        ```python
        factory = DataProcessingPipelineFactory(context, use_gpu=True)
        pipeline = factory.create(Dataset.TAIWAN_CREDIT)
        pipeline.run(Dataset.TAIWAN_CREDIT)
        ```
    """

    def __init__(
        self,
        path_provider: DataPathProvider,
        *,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the factory.

        Args:
            path_provider: Provider for both raw and interim data paths.
                Typically this is the application Context.
            use_gpu: Whether to enable GPU acceleration for transformations.
        """
        self._path_provider = path_provider
        self._use_gpu = use_gpu

        # Registry mapping datasets to their transformer classes
        self._transformer_registry: dict[str, type[BaseDataTransformer]] = {
            "lending_club": LendingClubTransformer,
            "corporate_credit_rating": CorporateCreditTransformer,
            "taiwan_credit": TaiwanCreditTransformer,
        }

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

        loader = CsvRawDataLoader(self._path_provider)
        transformer = transformer_cls(use_gpu=self._use_gpu)
        exporter = ParquetDataExporter(self._path_provider)

        return DataProcessingPipeline(
            loader=loader,
            transformer=transformer,
            exporter=exporter,
        )
