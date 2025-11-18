from unittest.mock import Mock

import pytest
import typer

from experiments import features as features_module
from experiments.config import Dataset
from experiments.utils import overwrites as overwrites_module


@pytest.fixture(autouse=True)
def _setup_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(features_module, "_BASE_PATH", tmp_path)

    def fake_get_processed_path(dataset):
        return tmp_path / f"{dataset.value}_input.parquet"

    monkeypatch.setattr(features_module, "get_processed_path", fake_get_processed_path)
    return tmp_path


@pytest.fixture
def parallel_spy(monkeypatch):
    class ParallelStub:
        def __init__(self):
            self.args = None
            self.kwargs = None
            self.submitted = []
            self.call_count = 0

        def __call__(self, tasks):
            self.call_count += 1
            submissions = list(tasks)
            self.submitted.extend(submissions)
            return [task() for task in submissions]

    stub = ParallelStub()

    def fake_parallel(*args, **kwargs):
        stub.args = args
        stub.kwargs = kwargs
        stub.submitted = []
        stub.call_count = 0
        return stub

    parallel_ctor = Mock(side_effect=fake_parallel)
    monkeypatch.setattr(features_module, "Parallel", parallel_ctor)

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(features_module, "delayed", fake_delayed)
    stub.constructor = parallel_ctor
    return stub


def test_process_single_dataset_missing_input_returns_error(monkeypatch, tmp_path):
    dataset = Dataset.LENDING_CLUB
    input_path = tmp_path / "missing.parquet"
    monkeypatch.setattr(features_module, "get_processed_path", lambda ds: input_path)

    result = features_module._process_single_dataset(dataset)

    assert result[0] is dataset
    assert result[1] is False
    assert "File not found" in result[2]


def test_main_processes_specific_dataset_in_parallel(parallel_spy, monkeypatch):
    calls: list[Dataset] = []

    def fake_process(ds: Dataset):
        calls.append(ds)
        return (ds, True, None)

    monkeypatch.setattr(features_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(features_module, "cpu_count", lambda: 4)

    features_module.main(Dataset.LENDING_CLUB)

    assert calls == [Dataset.LENDING_CLUB]
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 1
    assert len(parallel_spy.submitted) == 1


def test_main_processes_all_datasets_when_argument_missing(parallel_spy, monkeypatch):
    calls: list[Dataset] = []

    def fake_process(ds: Dataset):
        calls.append(ds)
        return (ds, True, None)

    monkeypatch.setattr(features_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(features_module, "cpu_count", lambda: 2)

    features_module.main(dataset=None)

    assert calls == list(Dataset)
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 2
    assert len(parallel_spy.submitted) == len(Dataset)


def test_main_skips_dataset_when_user_declines_overwrite(parallel_spy, monkeypatch, tmp_path):
    artifact = tmp_path / f"{Dataset.LENDING_CLUB.value}_X.parquet"
    artifact.touch()

    monkeypatch.setattr(features_module, "_BASE_PATH", tmp_path)
    monkeypatch.setattr(overwrites_module.typer, "confirm", lambda *args, **kwargs: False)

    features_module.main(Dataset.LENDING_CLUB)

    assert parallel_spy.constructor.call_count == 0


def test_main_force_overwrites_without_prompt(parallel_spy, monkeypatch, tmp_path):
    for suffix in [
        "_X.parquet",
        "_y.parquet",
    ]:
        (tmp_path / f"{Dataset.CORPORATE_CREDIT_RATING.value}{suffix}").touch()

    monkeypatch.setattr(features_module, "_BASE_PATH", tmp_path)

    confirm_mock = Mock(return_value=False)
    monkeypatch.setattr(overwrites_module.typer, "confirm", confirm_mock)

    calls: list[Dataset] = []

    def fake_process(ds: Dataset):
        calls.append(ds)
        return (ds, True, None)

    monkeypatch.setattr(features_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(features_module, "cpu_count", lambda: 2)

    features_module.main(dataset=None, force=True)

    assert confirm_mock.call_count == 0
    assert Dataset.CORPORATE_CREDIT_RATING in calls


def test_main_raises_exit_when_any_dataset_fails(monkeypatch):
    def fake_parallel(*args, **kwargs):  # noqa: ARG001
        def runner(tasks):
            list(tasks)
            return [
                (Dataset.LENDING_CLUB, True, None),
                (Dataset.CORPORATE_CREDIT_RATING, False, "boom"),
            ]

        return runner

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(features_module, "Parallel", fake_parallel)
    monkeypatch.setattr(features_module, "delayed", fake_delayed)
    monkeypatch.setattr(features_module, "cpu_count", lambda: 1)
    monkeypatch.setattr(features_module, "_process_single_dataset", lambda ds: (ds, True, None))

    with pytest.raises(typer.Exit) as excinfo:
        features_module.main(dataset=None)

    assert excinfo.value.exit_code == 1


def test_main_respects_jobs_override(parallel_spy, monkeypatch):
    calls: list[Dataset] = []

    def fake_process(ds: Dataset):
        calls.append(ds)
        return (ds, True, None)

    monkeypatch.setattr(features_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(features_module, "cpu_count", lambda: 8)

    features_module.main(dataset=None, jobs=1)

    assert calls == list(Dataset)
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 1
