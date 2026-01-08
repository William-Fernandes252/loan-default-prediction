"""Configuration and setup for logging in experiments."""

from typing import Any

from loguru import logger

from experiments.lib.pipelines import ErrorAction, Pipeline, PipelineExecutionResult, TaskResult
from experiments.lib.pipelines.lifecycle import IgnoreActionsPipelineObserver
from experiments.lib.pipelines.tasks import TaskStatus


class LoggingObserver(IgnoreActionsPipelineObserver):
    """Observer to logs pipeline events."""

    def on_step_start(
        self, pipeline: Pipeline[Any, Any], step_name: str, current_state: Any
    ) -> None:
        with logger.contextualize(pipeline_name=pipeline.name, step_name=step_name):
            logger.info(
                "Starting step '{step_name}' in pipeline '{pipeline_name}'",
                step_name=step_name,
                pipeline_name=pipeline.name,
            )

    def on_step_finish(
        self,
        pipeline: Pipeline[Any, Any],
        step_name: str,
        result: TaskResult[Any],
    ) -> None:
        with logger.contextualize(pipeline_name=pipeline.name, step_name=step_name):
            if result.status == TaskStatus.SUCCESS:
                logger.info(
                    "Finished step '{step_name}' in pipeline '{pipeline_name}' successfully (duration: {duration:.2f}s)",
                    step_name=step_name,
                    pipeline_name=pipeline.name,
                )
            elif result.status == TaskStatus.FAILURE:
                logger.warning(
                    "Step '{step_name}' in pipeline '{pipeline_name}' failed",
                    step_name=step_name,
                    pipeline_name=pipeline.name,
                )
            elif result.status == TaskStatus.SKIPPED:
                logger.info(
                    "Step '{step_name}' in pipeline '{pipeline_name}' was skipped",
                    step_name=step_name,
                    pipeline_name=pipeline.name,
                )

    def on_error(
        self, pipeline: Pipeline[Any, Any], step_name: str, error: Exception
    ) -> ErrorAction:
        with logger.contextualize(pipeline_name=pipeline.name, step_name=step_name):
            logger.error(
                "Error in step '{step_name}' of pipeline '{pipeline_name}': {error}",
                error=error,
                step_name=step_name,
                pipeline_name=pipeline.name,
            )
        return ErrorAction.ABORT

    def on_pipeline_start(self, pipeline: Pipeline[Any, Any]) -> None:
        with logger.contextualize(pipeline_name=pipeline.name):
            logger.info("Starting pipeline '{pipeline_name}'", pipeline_name=pipeline.name)

    def on_pipeline_finish(
        self,
        pipeline: Pipeline[Any, Any],
        result: "PipelineExecutionResult[Any, Any]",
    ) -> None:
        with logger.contextualize(pipeline_name=pipeline.name):
            if result.succeeded():
                logger.success(
                    "Pipeline '{pipeline_name}' finished successfully", pipeline_name=pipeline.name
                )
            else:
                logger.warning(
                    "Pipeline '{pipeline_name}' finished with errors or failures",
                    pipeline_name=pipeline.name,
                )
