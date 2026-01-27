import pytest

from experiments.lib.pipelines import Pipeline, Task, TaskResult, TaskStatus


class DescribePipeline:
    @pytest.fixture
    def task(self):
        def sample_task(state: int, context: dict) -> TaskResult[int]:
            return TaskResult(state + context.get("increment", 1), TaskStatus.SUCCESS)

        return sample_task

    @pytest.fixture
    def pipeline(self):
        pipeline = Pipeline[int, dict]("sample_pipeline")
        return pipeline

    def it_allows_adding_steps(self, pipeline: Pipeline, task: Task):
        pipeline.add_step("step1", task)
        assert any(step.step.name == "step1" for step in pipeline.steps)

    def it_allows_adding_conditional_steps(self, pipeline: Pipeline, task: Task):
        def condition(state, _):
            return state < 5

        pipeline.add_conditional_step("conditional_step", task, condition)
        assert any(step.condition == condition for step in pipeline.steps)
