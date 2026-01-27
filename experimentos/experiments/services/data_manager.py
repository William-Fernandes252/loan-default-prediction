"""Service for managing experiment data, including pre-processing, feature engineering, and storage."""

from typing import Iterable, Sequence

from experiments.config.settings import ResourceSettings
from experiments.core.data import DataRepository, Dataset
from experiments.lib.pipelines import PipelineExecutionResult, PipelineExecutor
from experiments.pipelines.data.factory import DataProcessingPipelineFactory
from experiments.pipelines.data.pipeline import (
    DataProcessingPipeline,
    DataProcessingPipelineContext,
    DataProcessingPipelineState,
)
from experiments.services.resource_calculator import ResourceCalculator


class DataManager:
    """Service for managing experiment data."""

    def __init__(
        self,
        data_pipeline_factory: DataProcessingPipelineFactory,
        data_repository: DataRepository,
        pipeline_executor: PipelineExecutor,
        resource_calculator: ResourceCalculator,
        resource_settings: ResourceSettings,
    ) -> None:
        self._data_pipeline_factory = data_pipeline_factory
        self._data_repository = data_repository
        self._pipeline_executor = pipeline_executor
        self._resource_calculator = resource_calculator
        self._resource_settings = resource_settings

    def process_datasets(
        self,
        datasets: Iterable[Dataset] | None = None,
        force_overwrite: bool = False,
        use_gpu: bool = False,
        workers: int | None = None,
    ) -> list[tuple[Dataset, Exception]]:
        """Processes the specified datasets.

        Args:
            datasets (Iterator[Dataset] | None): Iterator of datasets to process. If not provided, all datasets are processed.
            force_overwrite: Whether to force re-processing of datasets.
            use_gpu: Whether to utilize GPU acceleration.
            workers: Number of parallel workers to use.
        """
        datasets = datasets if datasets is not None else iter(Dataset)

        for ds in datasets:
            self._schedule_dataset_processing(
                dataset=ds,
                use_gpu=self._get_effective_use_gpu(use_gpu),
                force_overwrite=force_overwrite,
            )

        self._pipeline_executor.start(max_workers=self._get_safe_num_workers(datasets, workers))

        pipeline_results = self._pipeline_executor.wait()
        return self._get_errors_from_results(pipeline_results)

    def _get_effective_use_gpu(self, use_gpu: bool) -> bool:
        """Determines the effective GPU usage setting.

        Args:
            use_gpu: The requested GPU usage setting.

        Returns:
            The effective GPU usage setting.
        """
        return use_gpu or self._resource_settings.use_gpu

    def _get_pipeline_name(self, dataset: Dataset) -> str:
        """Generates a name for the data processing pipeline.

        Args:
            dataset: The dataset being processed.

        Returns:
            A string representing the pipeline name.
        """
        return f"DataProcessingPipeline[dataset={dataset}]"

    def _get_safe_num_workers(
        self, datasets: Iterable[Dataset], provided_workers: int | None
    ) -> int:
        """Calculates the number of parallel workers to use.

        Args:
            datasets (Iterator[Dataset]): An iterator of datasets to be processed.
            provided_workers (int | None): The number of workers provided by the user.

        Returns:
            The number of parallel workers.
        """
        total_size_gb = sum(self._data_repository.get_size_in_bytes(ds) / 1e9 for ds in datasets)
        return provided_workers or self._resource_calculator.compute_safe_jobs(
            dataset_size_gb=total_size_gb,
        )

    def _schedule_dataset_processing(
        self,
        dataset: Dataset,
        force_overwrite: bool,
        use_gpu: bool,
    ) -> None:
        """Schedules processing for a single dataset.

        Args:
            dataset: The dataset to process.
            force_overwrite: Whether to force re-processing of the dataset.
            use_gpu: Whether to utilize GPU acceleration.
        """
        self._pipeline_executor.schedule(
            self._create_pipeline(dataset),
            DataProcessingPipelineState(is_processed=False),
            self._create_context_for_dataset(
                dataset,
                force_overwrite,
                use_gpu,
            ),
        )

    def _create_context_for_dataset(
        self,
        dataset: Dataset,
        force_overwrite: bool,
        use_gpu: bool,
    ) -> DataProcessingPipelineContext:
        """Creates a data processing pipeline context.

        Args:
            dataset: The dataset being processed.
            force_overwrite: Whether to force re-processing of the dataset.
            use_gpu: Whether to utilize GPU acceleration.

        Returns:
            The created DataProcessingPipelineContext.
        """
        return DataProcessingPipelineContext(
            dataset=dataset,
            data_repository=self._data_repository,
            use_gpu=use_gpu,
            force_overwrite=force_overwrite,
        )

    def _create_pipeline(self, dataset: Dataset) -> DataProcessingPipeline:
        """Creates a data processing pipeline for the specified dataset.

        Args:
            dataset (Dataset): The dataset to create the pipeline for.

        Returns:
            DataProcessingPipeline: The created DataProcessingPipeline.
        """
        return self._data_pipeline_factory.create(
            name=self._get_pipeline_name(dataset),
        )

    def _get_errors_from_results(
        self,
        results: Sequence[PipelineExecutionResult],
    ) -> list[tuple[Dataset, Exception]]:
        """Extracts errors from pipeline execution results.

        Args:
            results (Sequence[PipelineExecutionResult]): A sequence of pipeline execution results.

        Returns:
            A list of tuples containing datasets and their corresponding exceptions.
        """
        errors = []
        for result in results:
            dataset: Dataset = getattr(result.context, "dataset")
            if (error := result.last_error()) is not None:
                errors.append((dataset, error))
        return errors
