import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from unittest.mock import MagicMock


@pytest.fixture
def feature_selector():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = X - X.mean(axis=0)
    model = LogisticRegression()
    scoring = "accuracy"
    return FeatureSelector(model=model, X=X, y=y, scoring=scoring)


def test_objective(feature_selector):
    trial = MagicMock()
    trial.suggest_categorical.return_value = "zero_out"
    score = feature_selector.objective(trial)
    assert score < 0


def test_optimize(feature_selector):
    feature_selector.study = MagicMock()
    feature_selector.optimize(n_trials=10)
    feature_selector.study.optimize.assert_called_once_with(
        feature_selector.objective, n_trials=10
    )


def test_get_best_zero_out_features(feature_selector):
    feature_selector.study.best_trial = MagicMock()
    feature_selector.study.best_trial.params = {
        "feature1": "zero_out",
        "feature2": "keep",
        "feature3": "zero_out",
    }
    zero_out_features = feature_selector.get_best_zero_out_features()
    assert zero_out_features == ["feature1", "feature3"]


if __name__ == "__main__":
    pytest.main()
