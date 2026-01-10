"""Configuration and setup for logging in experiments."""

from typing import Any

from loguru import logger

from experiments.lib.pipelines import Action, Pipeline, PipelineExecutionResult, TaskResult
from experiments.lib.pipelines.lifecycle import IgnoreAllObserver
from experiments.lib.pipelines.tasks import TaskStatus


class LoggingObserver(IgnoreAllObserver[Any, Any]):
    """Observer that logs pipeline events.

    Extends IgnoreAllObserver to add logging while returning PROCEED
    for all events except errors, which return ABORT.
    """

    def _format_pipeline_and_step(self, pipeline: Pipeline[Any, Any], step_name: str) -> str:
        return f"{pipeline.name}.{step_name}"

    def on_step_start(
        self, pipeline: Pipeline[Any, Any], step_name: str, current_state: Any
    ) -> Action:
        with logger.contextualize(pipeline_name=pipeline.name, step_name=step_name):
            logger.info(f"{self._format_pipeline_and_step(pipeline, step_name)}: Starting step")
        return Action.PROCEED

    def on_step_finish(
        self,
        pipeline: Pipeline[Any, Any],
        step_name: str,
        result: TaskResult[Any],
    ) -> Action:
        with logger.contextualize(pipeline_name=pipeline.name, step_name=step_name):
            if result.status == TaskStatus.SUCCESS:
                logger.info(
                    f"{self._format_pipeline_and_step(pipeline, step_name)}: Step completed successfully"
                )
            elif result.status == TaskStatus.FAILURE:
                logger.warning(
                    f"{self._format_pipeline_and_step(pipeline, step_name)}: Step failed"
                )
            elif result.status == TaskStatus.SKIPPED:
                logger.info(
                    f"{self._format_pipeline_and_step(pipeline, step_name)}: Step was skipped"
                )
        return Action.PROCEED

    def on_error(self, pipeline: Pipeline[Any, Any], step_name: str, error: Exception) -> Action:
        with logger.contextualize(pipeline_name=pipeline.name, step_name=step_name):
            logger.error(f"{self._format_pipeline_and_step(pipeline, step_name)}: {error}")
        return Action.ABORT

    def on_pipeline_start(self, pipeline: Pipeline[Any, Any]) -> Action:
        with logger.contextualize(pipeline_name=pipeline.name):
            logger.info("Starting pipeline '{pipeline_name}'", pipeline_name=pipeline.name)
        return Action.PROCEED

    def on_pipeline_finish(
        self,
        pipeline: Pipeline[Any, Any],
        result: "PipelineExecutionResult[Any, Any]",
    ) -> Action:
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
        return Action.PROCEED
