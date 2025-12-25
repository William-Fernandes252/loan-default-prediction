import numpy as np
import polars as pl
import pytest

from experiments.core.modeling.features import extract_features_and_target


def test_extract_features_and_target_splits_and_sanitizes() -> None:
    df = pl.DataFrame({"a": [1.0, np.inf], "target": [0, 1]})

    X, y = extract_features_and_target(df)

    assert "target" not in X.columns
    assert y.to_pandas()["target"].tolist() == [0, 1]
    assert not np.isinf(X.to_pandas()["a"]).any()


def test_extract_features_and_target_requires_target_column() -> None:
    with pytest.raises(ValueError):
        extract_features_and_target(pl.DataFrame({"a": [1]}))
