import time
from typing import Any

import pytest

from experiments.lib.pipelines import Pipeline, TaskResult, TaskStatus
from experiments.lib.pipelines.execution import (
    PipelineExecutionResult,
    PipelineExecutor,
    PipelineStatus,
    StepTrace,
)
from experiments.lib.pipelines.lifecycle import Action, IgnoreAllObserver


class DescribeStepTrace:
    def it_stores_step_execution_information(self) -> None:
        error = ValueError("test error")
        trace = StepTrace(
            name="test_step",
            duration=1.5,
            status=TaskStatus.SUCCESS,
            message="Step completed",
            error=error,
        )

        assert trace.name == "test_step"
        assert trace.duration == 1.5
        assert trace.status == TaskStatus.SUCCESS
        assert trace.message == "Step completed"
        assert trace.error is error

    def it_allows_none_for_optional_fields(self) -> None:
        trace = StepTrace(
            name="test_step",
            duration=0.5,
            status=TaskStatus.SUCCESS,
            message=None,
            error=None,
        )

        assert trace.message is None
        assert trace.error is None


class DescribePipelineExecutionResult:
    @pytest.fixture
    def successful_result(self) -> PipelineExecutionResult[int, dict[str, str]]:
        traces = [
            StepTrace("step1", 1.0, TaskStatus.SUCCESS, "Done", None),
            StepTrace("step2", 2.0, TaskStatus.SUCCESS, "Done", None),
        ]
        return PipelineExecutionResult(
            pipeline_name="test_pipeline",
            status=PipelineStatus.COMPLETED,
            step_traces=traces,
            context={"key": "value"},
            final_state=10,
        )

    @pytest.fixture
    def failed_result(self) -> PipelineExecutionResult[int, dict[str, Any]]:
        error1 = ValueError("error1")
        error2 = RuntimeError("error2")
        traces = [
            StepTrace("step1", 1.0, TaskStatus.ERROR, "Failed", error1),
            StepTrace("step2", 2.0, TaskStatus.ERROR, "Failed", error2),
        ]
        return PipelineExecutionResult(
            pipeline_name="test_pipeline",
            status=PipelineStatus.ABORTED,
            step_traces=traces,
            context={},
            final_state=0,
        )

    @pytest.fixture
    def empty_result(self) -> PipelineExecutionResult[int, dict[str, Any]]:
        return PipelineExecutionResult(
            pipeline_name="empty_pipeline",
            status=PipelineStatus.COMPLETED,
            step_traces=[],
            context={},
            final_state=0,
        )

    def it_returns_last_executed_step_name(
        self, successful_result: PipelineExecutionResult[int, dict[str, str]]
    ) -> None:
        assert successful_result.last_executed_step == "step2"

    def it_returns_none_when_no_steps_executed(
        self, empty_result: PipelineExecutionResult[int, dict[str, Any]]
    ) -> None:
        assert empty_result.last_executed_step is None

    def it_collects_errors_from_steps(
        self, failed_result: PipelineExecutionResult[int, dict[str, Any]]
    ) -> None:
        errors = failed_result.errors
        assert len(errors) == 2
        assert "step1" in errors
        assert "step2" in errors
        assert isinstance(errors["step1"], ValueError)
        assert isinstance(errors["step2"], RuntimeError)

    def it_returns_empty_dict_when_no_errors(
        self, successful_result: PipelineExecutionResult[int, dict[str, str]]
    ) -> None:
        assert successful_result.errors == {}

    def it_returns_step_durations(
        self, successful_result: PipelineExecutionResult[int, dict[str, str]]
    ) -> None:
        durations = successful_result.durations
        assert durations["step1"] == 1.0
        assert durations["step2"] == 2.0

    def it_calculates_total_duration(
        self, successful_result: PipelineExecutionResult[int, dict[str, str]]
    ) -> None:
        assert successful_result.total_duration() == 3.0

    def it_returns_zero_duration_for_empty_pipeline(
        self, empty_result: PipelineExecutionResult[int, dict[str, Any]]
    ) -> None:
        assert empty_result.total_duration() == 0.0

    def it_identifies_successful_execution(
        self, successful_result: PipelineExecutionResult[int, dict[str, str]]
    ) -> None:
        assert successful_result.succeeded() is True

    def it_identifies_failed_execution(
        self, failed_result: PipelineExecutionResult[int, dict[str, Any]]
    ) -> None:
        assert failed_result.succeeded() is False

    def it_identifies_incomplete_as_not_successful(self) -> None:
        result = PipelineExecutionResult(
            pipeline_name="test",
            status=PipelineStatus.PENDING,
            step_traces=[],
            context={},
            final_state=0,
        )
        assert result.succeeded() is False

    def it_returns_last_error_if_exists(
        self, failed_result: PipelineExecutionResult[int, dict[str, Any]]
    ) -> None:
        last_error = failed_result.last_error()
        assert last_error is not None
        assert isinstance(last_error, RuntimeError)
        assert str(last_error) == "error2"

    def it_returns_none_when_no_errors_exist(
        self, successful_result: PipelineExecutionResult[int, dict[str, str]]
    ) -> None:
        assert successful_result.last_error() is None

    def it_stores_pipeline_metadata(
        self, successful_result: PipelineExecutionResult[int, dict[str, str]]
    ) -> None:
        assert successful_result.pipeline_name == "test_pipeline"
        assert successful_result.status == PipelineStatus.COMPLETED
        assert successful_result.context == {"key": "value"}
        assert successful_result.final_state == 10


class DescribePipelineExecutor:
    @pytest.fixture
    def simple_pipeline(self) -> Pipeline[int, dict[str, Any]]:
        def task1(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + 1, TaskStatus.SUCCESS, "Step 1 done")

        def task2(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state * 2, TaskStatus.SUCCESS, "Step 2 done")

        pipeline = Pipeline[int, dict]("simple_pipeline")
        pipeline.add_step("step1", task1)
        pipeline.add_step("step2", task2)
        return pipeline

    @pytest.fixture
    def executor(self) -> PipelineExecutor:
        return PipelineExecutor(max_workers=2)

    def it_executes_pipeline_successfully(
        self, executor: PipelineExecutor, simple_pipeline: Pipeline[int, dict[str, Any]]
    ) -> None:
        result = executor.execute(
            pipeline=simple_pipeline,
            initial_state=5,
            context={},
        )

        assert result.succeeded()
        assert result.final_state == 12  # (5 + 1) * 2
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.step_traces) == 2
        assert result.step_traces[0].name == "step1"
        assert result.step_traces[1].name == "step2"

    def it_records_step_execution_traces(
        self, executor: PipelineExecutor, simple_pipeline: Pipeline[int, dict[str, Any]]
    ) -> None:
        result = executor.execute(
            pipeline=simple_pipeline,
            initial_state=0,
            context={},
        )

        assert len(result.step_traces) == 2
        for trace in result.step_traces:
            assert trace.duration > 0
            assert trace.status == TaskStatus.SUCCESS
            assert trace.error is None

    def it_handles_task_errors(self, executor: PipelineExecutor) -> None:
        def failing_task(state: int, context: dict) -> TaskResult[int]:
            raise ValueError("Task failed")

        pipeline = Pipeline[int, dict]("failing_pipeline")
        pipeline.add_step("failing_step", failing_task)

        result = executor.execute(
            pipeline=pipeline,
            initial_state=0,
            context={},
        )

        assert not result.succeeded()
        assert result.status == PipelineStatus.ABORTED
        assert len(result.errors) == 1
        assert "failing_step" in result.errors

    def it_skips_conditional_steps_when_condition_fails(self, executor: PipelineExecutor) -> None:
        def task(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + 1, TaskStatus.SUCCESS)

        def condition(state: int, context: dict):
            if state > 10:
                return (True, None)
            return (False, "State not greater than 10")

        pipeline = Pipeline[int, dict]("conditional_pipeline")
        pipeline.add_conditional_step("conditional_step", task, condition)

        result = executor.execute(
            pipeline=pipeline,
            initial_state=5,
            context={},
        )

        assert result.succeeded()
        assert result.final_state == 5  # State unchanged

    def it_executes_conditional_steps_when_condition_passes(
        self, executor: PipelineExecutor
    ) -> None:
        def task(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + 10, TaskStatus.SUCCESS)

        def condition(state: int, context: dict):
            if state > 10:
                return (True, None)
            return (False, "State not greater than 10")

        pipeline = Pipeline[int, dict]("conditional_pipeline")
        pipeline.add_conditional_step("conditional_step", task, condition)

        result = executor.execute(
            pipeline=pipeline,
            initial_state=15,
            context={},
        )

        assert result.succeeded()
        assert result.final_state == 25
        assert result.step_traces[0].status == TaskStatus.SUCCESS

    def it_executes_multiple_pipelines_in_parallel(self, executor: PipelineExecutor) -> None:
        def slow_task(state: int, context: dict) -> TaskResult[int]:
            time.sleep(0.1)
            return TaskResult(state + 1, TaskStatus.SUCCESS)

        pipeline1 = Pipeline[int, dict]("pipeline1")
        pipeline1.add_step("step1", slow_task)

        pipeline2 = Pipeline[int, dict]("pipeline2")
        pipeline2.add_step("step1", slow_task)

        start_time = time.time()
        results = executor.execute_all(
            pipelines=[(pipeline1, 0), (pipeline2, 0)],
            context={},
        )
        end_time = time.time()

        # Should complete in roughly 0.1s (parallel) not 0.2s (sequential)
        assert end_time - start_time < 0.15
        assert len(results) == 2
        assert all(r.succeeded() for r in results)

    def it_passes_context_through_pipeline(self, executor: PipelineExecutor) -> None:
        def task_with_context(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + context.get("increment", 0), TaskStatus.SUCCESS)

        pipeline = Pipeline[int, dict]("context_pipeline")
        pipeline.add_step("step1", task_with_context)

        result = executor.execute(
            pipeline=pipeline,
            initial_state=10,
            context={"increment": 5},
        )

        assert result.succeeded()
        assert result.final_state == 15
        assert result.context == {"increment": 5}

    def it_allows_custom_max_workers(self) -> None:
        executor = PipelineExecutor(max_workers=8)
        # Executor should be created without error
        assert executor is not None

    def it_maintains_step_order_in_traces(self, executor: PipelineExecutor) -> None:
        def task(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + 1, TaskStatus.SUCCESS)

        pipeline = Pipeline[int, dict]("ordered_pipeline")
        for i in range(5):
            pipeline.add_step(f"step{i}", task)

        result = executor.execute(
            pipeline=pipeline,
            initial_state=0,
            context={},
        )

        assert len(result.step_traces) == 5
        for i, trace in enumerate(result.step_traces):
            assert trace.name == f"step{i}"

    def it_handles_empty_pipelines(self, executor: PipelineExecutor) -> None:
        pipeline = Pipeline[int, dict]("empty_pipeline")

        result = executor.execute(
            pipeline=pipeline,
            initial_state=42,
            context={},
        )

        assert result.succeeded()
        assert result.final_state == 42
        assert len(result.step_traces) == 0


class DescribePipelineExecutorWithObservers:
    @pytest.fixture
    def simple_pipeline(self) -> Pipeline[int, dict[str, Any]]:
        def task(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + 1, TaskStatus.SUCCESS)

        pipeline = Pipeline[int, dict]("test_pipeline")
        pipeline.add_step("step1", task)
        return pipeline

    @pytest.fixture
    def executor(self) -> PipelineExecutor:
        return PipelineExecutor(max_workers=2)

    def it_notifies_observers_on_pipeline_start(
        self, executor: PipelineExecutor, simple_pipeline: Pipeline[int, dict[str, Any]]
    ) -> None:
        started = []

        class TestObserver(IgnoreAllObserver[int, dict[str, Any]]):
            def on_pipeline_start(self, pipeline: Pipeline[int, dict]) -> Action:
                started.append(pipeline.name)
                return Action.PROCEED

        observer = TestObserver()
        result = executor.execute(
            pipeline=simple_pipeline,
            initial_state=0,
            context={},
            observers={observer},
        )

        assert "test_pipeline" in started
        assert result.succeeded()

    def it_notifies_observers_on_pipeline_finish(
        self, executor: PipelineExecutor, simple_pipeline: Pipeline[int, dict[str, Any]]
    ) -> None:
        finished = []

        class TestObserver(IgnoreAllObserver[int, dict[str, Any]]):
            def on_pipeline_finish(
                self,
                pipeline: Pipeline[int, dict],
                result: PipelineExecutionResult[int, dict],
            ) -> Action:
                finished.append(result.pipeline_name)
                return Action.PROCEED

        observer = TestObserver()
        result = executor.execute(
            pipeline=simple_pipeline,
            initial_state=0,
            context={},
            observers={observer},
        )

        assert "test_pipeline" in finished
        assert result.succeeded()

    def it_notifies_observers_on_step_start(
        self, executor: PipelineExecutor, simple_pipeline: Pipeline[int, dict[str, Any]]
    ) -> None:
        steps_started = []

        class TestObserver(IgnoreAllObserver[int, dict[str, Any]]):
            def on_step_start(
                self, pipeline: Pipeline[int, dict], step_name: str, current_state: int
            ) -> Action:
                steps_started.append(step_name)
                return Action.PROCEED

        observer = TestObserver()
        result = executor.execute(
            pipeline=simple_pipeline,
            initial_state=0,
            context={},
            observers={observer},
        )

        assert "step1" in steps_started
        assert result.succeeded()

    def it_notifies_observers_on_step_finish(
        self, executor: PipelineExecutor, simple_pipeline: Pipeline[int, dict[str, Any]]
    ) -> None:
        steps_finished = []

        class TestObserver(IgnoreAllObserver[int, dict[str, Any]]):
            def on_step_finish(
                self,
                pipeline: Pipeline[int, dict],
                step_name: str,
                result: TaskResult[int],
            ) -> Action:
                steps_finished.append(step_name)
                return Action.PROCEED

        observer = TestObserver()
        result = executor.execute(
            pipeline=simple_pipeline,
            initial_state=0,
            context={},
            observers={observer},
        )

        assert "step1" in steps_finished
        assert result.succeeded()

    def it_notifies_observers_on_error(self, executor: PipelineExecutor) -> None:
        def failing_task(state: int, context: dict) -> TaskResult[int]:
            raise ValueError("Test error")

        pipeline = Pipeline[int, dict]("failing_pipeline")
        pipeline.add_step("failing_step", failing_task)

        errors_caught = []

        class TestObserver(IgnoreAllObserver[int, dict[str, Any]]):
            def on_error(
                self, pipeline: Pipeline[int, dict], step_name: str, error: Exception
            ) -> Action:
                errors_caught.append((step_name, error))
                return Action.ABORT

        observer = TestObserver()
        result = executor.execute(
            pipeline=pipeline,
            initial_state=0,
            context={},
            observers={observer},
        )

        assert len(errors_caught) == 1
        assert errors_caught[0][0] == "failing_step"
        assert isinstance(errors_caught[0][1], ValueError)
        assert not result.succeeded()

    def it_aborts_pipeline_when_observer_returns_abort(self, executor: PipelineExecutor) -> None:
        def task(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + 1, TaskStatus.SUCCESS)

        pipeline = Pipeline[int, dict]("abort_pipeline")
        pipeline.add_step("step1", task)
        pipeline.add_step("step2", task)

        class TestObserver(IgnoreAllObserver[int, dict[str, Any]]):
            def on_step_finish(
                self,
                pipeline: Pipeline[int, dict],
                step_name: str,
                result: TaskResult[int],
            ) -> Action:
                if step_name == "step1":
                    return Action.ABORT
                return Action.PROCEED

        observer = TestObserver()
        result = executor.execute(
            pipeline=pipeline,
            initial_state=0,
            context={},
            observers={observer},
        )

        assert result.status == PipelineStatus.ABORTED
        assert len(result.step_traces) == 1
        assert result.step_traces[0].name == "step1"


class DescribePipelineStatus:
    def it_has_pending_status(self) -> None:
        assert PipelineStatus.PENDING.value == "pending"

    def it_has_running_status(self) -> None:
        assert PipelineStatus.RUNNING.value == "running"

    def it_has_completed_status(self) -> None:
        assert PipelineStatus.COMPLETED.value == "completed"

    def it_has_aborted_status(self) -> None:
        assert PipelineStatus.ABORTED.value == "aborted"

    def it_has_panicked_status(self) -> None:
        assert PipelineStatus.PANICKED.value == "panicked"
