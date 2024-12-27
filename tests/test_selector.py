from typing import Any, Tuple
from unittest.mock import MagicMock

import numpy as np
import optuna
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from feature_selector import FeatureSelectionResult, FeatureSelector


@pytest.fixture
def sample_data() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = X - X.mean(axis=0)
    return X, y


@pytest.fixture
def feature_selector(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]]
) -> FeatureSelector:
    X, y = sample_data
    model = LogisticRegression(random_state=42)
    return FeatureSelector(model=model, X=X, y=y, scoring="accuracy", random_state=42)


def test_initialization(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]]
) -> None:
    X, y = sample_data
    model = LogisticRegression()

    selector = FeatureSelector(model, X, y, "accuracy")
    assert len(selector.feature_names) == X.shape[1]

    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    selector = FeatureSelector(model, X_df, y, "accuracy")
    assert selector.feature_names == X_df.columns.tolist()


def test_input_validation(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]]
) -> None:
    X, y = sample_data
    model = LogisticRegression()

    with pytest.raises(ValueError):
        FeatureSelector(model, X, y, "accuracy", cv=1)

    with pytest.raises(ValueError):
        FeatureSelector(model, X, y[:-1], "accuracy")

    X_with_nan = X.copy()
    X_with_nan[0, 0] = np.nan
    X_df_with_nan = pd.DataFrame(X_with_nan)
    with pytest.raises(ValueError):
        FeatureSelector(model, X_df_with_nan, y, "accuracy")


def test_objective_function(feature_selector: FeatureSelector) -> None:
    trial = optuna.trial.Trial(study=MagicMock(), trial_id=0)
    trial.suggest_categorical = MagicMock(return_value="keep")

    score = feature_selector.objective(trial)
    assert isinstance(score, float)
    assert score <= 0


def test_optimize(feature_selector: FeatureSelector) -> None:
    result = feature_selector.optimize(n_trials=5, show_progress_bar=False)

    assert isinstance(result, FeatureSelectionResult)
    assert isinstance(result.selected_features, list)
    assert isinstance(result.zero_out_features, list)
    assert isinstance(result.best_score, float)
    assert isinstance(result.best_trial_params, dict)
    assert isinstance(result.study, optuna.study.Study)


def test_transform(
    feature_selector: FeatureSelector,
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, _ = sample_data

    with pytest.raises(RuntimeError):
        feature_selector.transform(X)

    feature_selector.optimize(n_trials=5, show_progress_bar=False)
    X_transformed = feature_selector.transform(X)

    assert X_transformed.shape == X.shape
    assert not np.array_equal(X_transformed, X)


def test_feature_importance(feature_selector: FeatureSelector) -> None:
    with pytest.raises(RuntimeError):
        feature_selector.get_feature_importance()

    feature_selector.optimize(n_trials=5, show_progress_bar=False)
    importance_df = feature_selector.get_feature_importance()

    assert isinstance(importance_df, pd.DataFrame)
    assert len(importance_df) == len(feature_selector.feature_names)
    assert all(0 <= freq <= 1 for freq in importance_df["selection_frequency"])


def test_reproducibility(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]]
) -> None:
    X, y = sample_data
    model = LogisticRegression(random_state=42)

    selector1 = FeatureSelector(model, X, y, "accuracy", random_state=42)
    selector2 = FeatureSelector(model, X, y, "accuracy", random_state=42)

    result1 = selector1.optimize(n_trials=5, show_progress_bar=False)
    result2 = selector2.optimize(n_trials=5, show_progress_bar=False)

    assert result1.best_score == result2.best_score
    assert result1.selected_features == result2.selected_features


def test_pandas_integration(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]]
) -> None:
    X, y = sample_data
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    model = LogisticRegression(random_state=42)

    selector = FeatureSelector(model, X_df, y, "accuracy")
    result = selector.optimize(n_trials=5, show_progress_bar=False)

    X_transformed = selector.transform(X_df)
    assert isinstance(X_transformed, pd.DataFrame)
    assert list(X_transformed.columns) == list(X_df.columns)


if __name__ == "__main__":
    pytest.main([__file__])
