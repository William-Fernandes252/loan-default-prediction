from typing import Generic

from experiments.lib.pipelines.context import Context


class PipelineException(Exception):
    """Base exception for pipeline errors."""


class PipelineInterruption(Generic[Context], Exception):
    """Exception to signal pipeline interruption and request user action.

    It carries the context at the point of interruption, allowing handlers to make informed decisions.
    """

    def __init__(self, message: str, context: Context) -> None:
        self.context = context
        super().__init__(message)
