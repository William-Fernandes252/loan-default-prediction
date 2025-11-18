from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest
import typer

from experiments.config import Dataset
from experiments.modeling import train as train_module


@pytest.fixture(autouse=True)
def _setup_dirs(monkeypatch, tmp_path):
    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"
    processed_dir.mkdir()
    results_dir.mkdir()

    monkeypatch.setattr(train_module, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(train_module, "RESULTS_DIR", results_dir)
    return processed_dir, results_dir


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
    monkeypatch.setattr(train_module, "Parallel", parallel_ctor)

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(train_module, "delayed", fake_delayed)
    stub.constructor = parallel_ctor
    return stub


def test_load_training_artifacts_missing_files():
    with pytest.raises(FileNotFoundError):
        train_module._load_training_artifacts(Dataset.LENDING_CLUB)


def test_load_training_artifacts_reads_parquet():
    dataset = Dataset.LENDING_CLUB
    x_path, y_path = train_module._get_training_artifact_paths(dataset)

    pl.DataFrame({"f1": [1, 2], "f2": [3, 4]}).write_parquet(x_path)
    pl.DataFrame({"target": [0, 1]}).write_parquet(y_path)

    X, y = train_module._load_training_artifacts(dataset)

    assert list(X.columns) == ["f1", "f2"]
    assert y.tolist() == [0, 1]


def test_train_single_dataset_success(monkeypatch):
    dataset = Dataset.LENDING_CLUB
    X = pd.DataFrame({"feat": [1, 2]})
    y = pd.Series([0, 1], name="target")

    monkeypatch.setattr(train_module, "_load_training_artifacts", lambda ds: (X, y))
    run_mock = Mock()
    monkeypatch.setattr(train_module, "run_experiment", run_mock)

    result = train_module._train_single_dataset(dataset, "output.parquet")

    expected_path = train_module.RESULTS_DIR / f"{dataset.value}_output.parquet"
    assert result == (dataset, True, str(expected_path))
    run_mock.assert_called_once()
    args, kwargs = run_mock.call_args
    assert args == (dataset, X, y, expected_path)
    assert kwargs == {}


def test_train_single_dataset_failure_returns_error(monkeypatch):
    dataset = Dataset.CORPORATE_CREDIT_RATING

    def failing_loader(ds):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(train_module, "_load_training_artifacts", failing_loader)

    result = train_module._train_single_dataset(dataset, "out.parquet")

    assert result[0] is dataset
    assert result[1] is False
    assert "boom" in result[2]


def test_main_processes_specific_dataset_in_parallel(parallel_spy, monkeypatch):
    called = []

    def fake_train(ds, output):  # noqa: ARG001
        called.append(ds)
        return (ds, True, None)

    monkeypatch.setattr(train_module, "_train_single_dataset", fake_train)
    monkeypatch.setattr(train_module, "_artifacts_exist", lambda ds: True)
    monkeypatch.setattr(train_module, "cpu_count", lambda: 4)

    train_module.main(Dataset.LENDING_CLUB)

    assert called == [Dataset.LENDING_CLUB]
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 1
    assert len(parallel_spy.submitted) == 1


def test_main_processes_all_datasets_when_argument_missing(parallel_spy, monkeypatch):
    called = []

    def fake_train(ds, output):  # noqa: ARG001
        called.append(ds)
        return (ds, True, None)

    monkeypatch.setattr(train_module, "_train_single_dataset", fake_train)
    monkeypatch.setattr(train_module, "_artifacts_exist", lambda ds: True)
    monkeypatch.setattr(train_module, "cpu_count", lambda: 2)

    train_module.main(dataset=None)

    assert called == list(Dataset)
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 2
    assert len(parallel_spy.submitted) == len(Dataset)


def test_main_raises_exit_when_no_dataset_available(monkeypatch):
    monkeypatch.setattr(train_module, "_artifacts_exist", lambda ds: False)

    with pytest.raises(typer.Exit) as excinfo:
        train_module.main(dataset=None)

    assert excinfo.value.exit_code == 1


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

    monkeypatch.setattr(train_module, "Parallel", fake_parallel)
    monkeypatch.setattr(train_module, "delayed", fake_delayed)
    monkeypatch.setattr(train_module, "_artifacts_exist", lambda ds: True)

    with pytest.raises(typer.Exit) as excinfo:
        train_module.main(dataset=None)

    assert excinfo.value.exit_code == 1


def test_main_respects_jobs_override(parallel_spy, monkeypatch):
    called = []

    def fake_train(ds, output):  # noqa: ARG001
        called.append(ds)
        return (ds, True, None)

    monkeypatch.setattr(train_module, "_train_single_dataset", fake_train)
    monkeypatch.setattr(train_module, "_artifacts_exist", lambda ds: True)
    monkeypatch.setattr(train_module, "cpu_count", lambda: 8)

    train_module.main(dataset=None, jobs=1)

    assert called == list(Dataset)
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 1
