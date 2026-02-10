"""Service for resolving experiment parameters with auto-resume logic.

This module provides the ExperimentParamsResolver service, which handles all parameter
resolution for experiment execution. It encapsulates complex auto-resume decision logic,
validation of mutually exclusive options, and execution ID management.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel, Field

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType
from experiments.core.predictions.repository import ModelPredictionsRepository
from experiments.services.experiment_executor import ExperimentConfig, ExperimentParams

if TYPE_CHECKING:
    from experiments.services.experiment_executor import ExperimentExecutor


class ResolverOptions(BaseModel):
    """Options for parameter resolution from CLI inputs.

    These are the raw CLI inputs that need to be resolved into final ExperimentParams.
    This allows the execution_id to be optional and includes the skip_resume flag.
    """

    datasets: list[Dataset] = Field(
        default_factory=lambda: list(Dataset),
        json_schema_extra={"description": "Datasets to process"},
    )
    excluded_models: list[ModelType] = Field(
        default_factory=list,
        json_schema_extra={"description": "Models to exclude"},
    )
    execution_id: str | None = Field(
        default=None,
        json_schema_extra={"description": "Explicit execution ID for continuation"},
    )
    skip_resume: bool = Field(
        default=False,
        json_schema_extra={"description": "Skip auto-resume and start a new execution"},
    )


class ResolutionStatus(str, Enum):
    """Status describing how parameters were resolved."""

    NEW_EXECUTION = "new_execution"
    """Starting a fresh execution (no prior work)."""

    RESUMED_INCOMPLETE = "resumed_incomplete"
    """Resuming an incomplete execution."""

    ALREADY_COMPLETE = "already_complete"
    """Latest execution is already complete (idempotent)."""

    EXPLICIT_ID_NEW = "explicit_id_new"
    """Using explicit execution ID with no prior work."""

    EXPLICIT_ID_CONTINUED = "explicit_id_continued"
    """Continuing explicit execution ID with existing work."""

    SKIP_RESUME = "skip_resume"
    """Skipped auto-resume to start fresh."""


class ResolutionContext(TypedDict, total=False):
    """Contextual information about how parameters were resolved.

    This metadata helps with logging, debugging, and understanding how the
    final parameters were determined from the input options.
    """

    status: ResolutionStatus
    completed_count: int
    execution_id: str
    datasets: list[Dataset]


@dataclass(frozen=True, slots=True)
class ResolutionError:
    """Error that occurred during parameter resolution.

    Attributes:
        code (str): A short error code identifier
        message (str): A human-readable error message
        details (dict[str, str] | None): Optional dictionary with additional error details for debugging
    """

    code: str
    message: str
    details: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ResolutionSuccess:
    """Result of parameter resolution.

    Attributes:
        params (ExperimentParams): The resolved experiment parameters to use for execution
        context (ResolutionContext): Metadata about how the parameters were resolved
    """

    params: ExperimentParams
    context: ResolutionContext

    @property
    def should_exit_early(self) -> bool:
        """Check if execution should exit without running (e.g., already complete)."""
        return (
            self.context is not None
            and self.context.get("status") == ResolutionStatus.ALREADY_COMPLETE
        )


class ExperimentParamsResolver:
    """Resolves experiment parameters with auto-resume logic.

    This service handles the complete parameter resolution flow including:
    - Dataset filtering based on CLI options
    - Auto-resume logic for incomplete executions
    - Validation of execution IDs
    - Mutually exclusive option checking
    - Idempotent exits when execution is already complete

    The resolver implements three resolution paths:
    1. Explicit execution ID: User provides --execution-id flag
    2. Skip resume: User provides --skip-resume flag (force new execution)
    3. Auto-resume: Default behavior that resumes latest incomplete execution

    All resolution logic preserves the exact behavior from the original CLI
    implementation to ensure backward compatibility.
    """

    def __init__(
        self,
        predictions_repository: ModelPredictionsRepository,
        experiment_executor: "ExperimentExecutor",
    ) -> None:
        """Initialize the resolver.

        Args:
            predictions_repository: Repository for querying execution history
            experiment_executor: Executor for checking completion status
        """
        self._predictions_repository = predictions_repository
        self._experiment_executor = experiment_executor

    def resolve_params(
        self,
        options: ResolverOptions,
        config: ExperimentConfig,
    ) -> ResolutionSuccess | ResolutionError:
        """Resolve experiment parameters from options.

        Implements the complete auto-resume logic while remaining decoupled
        from any specific UI framework. Returns a result object that the
        caller can interpret and handle appropriately.

        Args:
            options: Raw options to resolve
            config: Experiment configuration (needed for completion checking)

        Returns:
            ResolutionSuccess | ResolutionError: containing either resolved parameters or an error
        """
        # Validate mutually exclusive options
        error = self._validate_mutually_exclusive_options(
            options.execution_id, options.skip_resume
        )
        if error is not None:
            return error

        # Case 1: User provided explicit execution ID → validate and use it
        if options.execution_id is not None:
            return self._resolve_with_explicit_id(options)

        # Case 2: User wants to skip auto-resume → force new execution
        if options.skip_resume:
            return self._resolve_with_skip_resume(options)

        # Case 3: No execution ID and no skip flag → auto-resume latest execution
        return self._resolve_with_auto_resume(options, config)

    def _validate_mutually_exclusive_options(
        self, execution_id: str | None, skip_resume: bool
    ) -> ResolutionError | None:
        """Validate that execution_id and skip_resume are not both set.

        Args:
            execution_id: The execution ID provided (or None)
            skip_resume: Whether skip_resume flag is set

        Returns:
            ResolutionError if validation fails, None otherwise
        """
        if execution_id is not None and skip_resume:
            return ResolutionError(
                code="mutually_exclusive_options",
                message="Cannot use both execution_id and skip_resume options together",
                details={
                    "execution_id": execution_id,
                    "skip_resume": str(skip_resume),
                },
            )
        return None

    def _resolve_with_explicit_id(self, options: ResolverOptions) -> ResolutionSuccess:
        """Resolve parameters when explicit execution ID is provided.

        Checks if the execution ID has existing work, but allows starting
        fresh with a specific ID if no work exists yet.

        Args:
            options: Resolver options with execution_id set

        Returns:
            ResolutionSuccess: with parameters and context
        """
        completed_count = self._experiment_executor.get_completed_count(
            options.execution_id  # type: ignore
        )

        params = ExperimentParams(
            datasets=options.datasets,
            excluded_models=options.excluded_models,
            execution_id=options.execution_id,  # type: ignore
        )

        status = (
            ResolutionStatus.EXPLICIT_ID_CONTINUED
            if completed_count > 0
            else ResolutionStatus.EXPLICIT_ID_NEW
        )

        context: ResolutionContext = {
            "status": status,
            "completed_count": completed_count,
            "execution_id": options.execution_id,  # type: ignore
            "datasets": options.datasets,
        }

        return ResolutionSuccess(params=params, context=context)

    def _resolve_with_skip_resume(self, options: ResolverOptions) -> ResolutionSuccess:
        """Resolve parameters when skip_resume flag is set.

        Creates a new execution even if incomplete executions exist.

        Args:
            options: Resolver options with skip_resume=True

        Returns:
            ResolutionSuccess: with new execution parameters
        """
        params = ExperimentParams(
            datasets=options.datasets,
            excluded_models=options.excluded_models,
            # No execution_id = new ID will be generated by default_factory
        )

        context: ResolutionContext = {
            "status": ResolutionStatus.SKIP_RESUME,
            "execution_id": params.execution_id,
            "datasets": options.datasets,
        }

        return ResolutionSuccess(params=params, context=context)

    def _resolve_with_auto_resume(
        self, options: ResolverOptions, config: ExperimentConfig
    ) -> ResolutionSuccess:
        """Resolve parameters with auto-resume logic.

        This method implements the complete auto-resume decision tree:
        - Query for latest execution with matching datasets
        - If no prior execution found → start new execution
        - If prior execution found and complete → return already_complete status
        - If prior execution found and incomplete → resume it

        CRITICAL: This preserves exact behavior from lines 128-177 of the
        original experiment.py CLI implementation.

        Args:
            options: Resolver options (no execution_id, skip_resume=False)
            config: Experiment configuration for completion checking

        Returns:
            ResolutionSuccess: with parameters and status
        """
        # Query latest execution for these datasets
        latest_exec_id = self._predictions_repository.get_latest_execution_id(options.datasets)

        # No prior execution found
        if latest_exec_id is None:
            params = ExperimentParams(
                datasets=options.datasets,
                excluded_models=options.excluded_models,
            )
            context: ResolutionContext = {
                "status": ResolutionStatus.NEW_EXECUTION,
                "execution_id": params.execution_id,
                "datasets": options.datasets,
            }
            return ResolutionSuccess(params=params, context=context)

        # Prior execution found - check if complete
        temp_params = ExperimentParams(
            datasets=options.datasets,
            excluded_models=options.excluded_models,
            execution_id=latest_exec_id,
        )

        is_complete = self._experiment_executor.is_execution_complete(
            latest_exec_id,
            temp_params,
            config,
        )

        if is_complete:
            # All work already done - return special status
            # Caller should exit without error (idempotent)
            params = temp_params
            context: ResolutionContext = {
                "status": ResolutionStatus.ALREADY_COMPLETE,
                "execution_id": latest_exec_id,
                "datasets": options.datasets,
            }
            return ResolutionSuccess(params=params, context=context)

        # Incomplete execution - resume it
        completed_count = self._experiment_executor.get_completed_count(latest_exec_id)

        params = ExperimentParams(
            datasets=options.datasets,
            excluded_models=options.excluded_models,
            execution_id=latest_exec_id,
        )

        context: ResolutionContext = {
            "status": ResolutionStatus.RESUMED_INCOMPLETE,
            "completed_count": completed_count,
            "execution_id": latest_exec_id,
            "datasets": options.datasets,
        }

        return ResolutionSuccess(params=params, context=context)
