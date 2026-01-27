"""Implementation of the analysis artifacts repository using a storage backend."""

from dataclasses import dataclass

from experiments.core.data.datasets import Dataset
from experiments.storage import Storage


@dataclass(frozen=True, slots=True)
class AnalysisArtifactsStorageLayout:
    """Layout for storing analysis artifacts in the storage backend."""

    artifacts_key_template: str = "reports/{dataset}/{analysis_name}"
    artifacts_prefix: str = "reports/"

    def get_artifact_key(
        self,
        dataset: Dataset,
        analysis_name: str,
    ) -> str:
        """Generate the storage key for an analysis artifact.

        Args:
            dataset: The dataset associated with the analysis.
            analysis_name: The name of the analysis artifact (e.g., 'summary_table.tex').

        Returns:
            str: The storage key for the analysis artifact.
        """
        return self.artifacts_key_template.format(
            dataset=dataset.value,
            analysis_name=analysis_name,
        )


class AnalysisArtifactsRepository:
    """Analysis artifacts repository implementation using a storage backend.

    This repository implements the `AnalysisArtifactRepository` protocol defined
    in `experiments.pipelines.analysis.pipeline` and provides methods to save
    and check for the existence of analysis artifacts.

    Attributes:
        _storage: The storage backend for persisting artifacts.
        _layout: The layout configuration for artifact storage keys.
    """

    def __init__(
        self,
        storage: Storage,
        layout: AnalysisArtifactsStorageLayout | None = None,
    ) -> None:
        """Initialize the analysis artifacts repository.

        Args:
            storage: The storage backend to use for persisting artifacts.
            layout: Optional layout configuration. If not provided, uses defaults.
        """
        self._storage = storage
        self._layout = layout or AnalysisArtifactsStorageLayout()

    def save_analysis_artifact(
        self,
        dataset: Dataset,
        analysis_name: str,
        artifact_data: bytes,
    ) -> None:
        """Saves an analysis artifact for a given experiment.

        Delegates to the storage backend's `write_bytes` method to persist the
        artifact data at the appropriate key.

        Args:
            dataset: The dataset associated with the experiment.
            analysis_name: The name of the analysis artifact to be saved.
            artifact_data: The binary data of the artifact.

        Raises:
            StorageError: If there is an error saving the artifact.
        """
        key = self._layout.get_artifact_key(dataset, analysis_name)
        self._storage.write_bytes(artifact_data, key)

    def artifact_exists(
        self,
        dataset: Dataset,
        analysis_name: str,
    ) -> bool:
        """Checks if an analysis artifact exists in the repository.

        Uses the storage backend's `exists` method to check for the artifact.

        Args:
            dataset: The dataset associated with the experiment.
            analysis_name: The name of the analysis artifact to check.

        Returns:
            bool: True if the artifact exists, False otherwise.
        """
        key = self._layout.get_artifact_key(dataset, analysis_name)
        return self._storage.exists(key)
