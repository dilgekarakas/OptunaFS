import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

logging.getLogger("optuna").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

T = TypeVar("T")


@dataclass
class FeatureSelectionResult:
    selected_features: List[str] = field(default_factory=list)
    zero_out_features: List[str] = field(default_factory=list)
    best_score: float = 0.0
    best_trial_params: Dict[str, Any] = field(default_factory=dict)
    study: Optional[optuna.study.Study] = None
    execution_time: float = 0.0
    feature_importance: Optional[pd.DataFrame] = None
    cv_scores: List[float] = field(default_factory=list)
    feature_groups: Optional[Dict[str, List[str]]] = None
    feature_names: List[str] = field(default_factory=list)
    optimization_direction: str = "maximize"

    def _create_feature_mask(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create a feature mask dictionary with threshold and weights"""
        threshold = trial.suggest_float("threshold", 0.1, 0.9)

        weights = {}
        if self.feature_groups:
            for group_name, features in self.feature_groups.items():
                group_weight = trial.suggest_float(f"group_{group_name}", 0.0, 1.0)
                for feature in features:
                    weights[feature] = group_weight
        else:
            for feature in self.feature_names:
                weights[feature] = trial.suggest_float(f"weight_{feature}", 0.0, 1.0)

        return {"weights": weights, "threshold": threshold}

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using feature weights and threshold"""
        if feature_params is None:
            if not hasattr(self, "study") or self.study is None:
                raise RuntimeError(
                    "Must run optimize() before transform() without feature_params"
                )

            best_trial = self.study.best_trial
            threshold = best_trial.params["threshold"]
            weights = {}

            if self.feature_groups:
                for group_name, features in self.feature_groups.items():
                    group_weight = best_trial.params[f"group_{group_name}"]
                    for feature in features:
                        weights[feature] = group_weight
            else:
                for feature in self.feature_names:
                    weights[feature] = best_trial.params[f"weight_{feature}"]

            feature_params = {"weights": weights, "threshold": threshold}

        X_new = X.copy()
        weights = feature_params["weights"]
        threshold = feature_params["threshold"]

        if isinstance(X, pd.DataFrame):
            for feature, weight in weights.items():
                if weight <= threshold:
                    X_new[feature] = 0
        else:
            weight_array = np.array([weights[f] for f in self.feature_names])
            mask = weight_array <= threshold
            X_new[:, mask] = 0

        return X_new

    def get_feature_importance(self) -> pd.DataFrame:
        """Enhanced feature importance calculation with proper type handling"""
        if self.study is None:
            raise RuntimeError("Must run optimize() before getting feature importance")

        feature_stats: Dict[str, Dict[str, Union[int, List[float]]]] = {
            name: {"keep_count": 0, "scores": []} for name in self.feature_names
        }

        for trial in self.study.trials:
            if (
                trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
            ):
                score = float(trial.value)
                if self.optimization_direction != "maximize":
                    score = -score

                for name, param in zip(self.feature_names, trial.params.values()):
                    if param == "keep":
                        stats = feature_stats[name]
                        keep_count = cast(int, stats["keep_count"])
                        scores = cast(List[float], stats["scores"])
                        stats["keep_count"] = keep_count + 1
                        scores.append(score)

        importance_data = []
        for name, stats in feature_stats.items():
            scores_list = cast(List[float], stats["scores"])
            scores_array = np.array(scores_list, dtype=np.float64)
            keep_count = cast(int, stats["keep_count"])
            n_trials = len(self.study.trials)

            importance_data.append(
                {
                    "feature": name,
                    "selection_frequency": keep_count / n_trials,
                    "mean_score_when_selected": float(np.mean(scores_array))
                    if len(scores_array)
                    else 0.0,
                    "std_score_when_selected": float(np.std(scores_array))
                    if len(scores_array)
                    else 0.0,
                    "times_selected": keep_count,
                }
            )

        importance_df = pd.DataFrame(importance_data)
        return importance_df.sort_values(
            "selection_frequency", ascending=False
        ).reset_index(drop=True)


class BaseFeatureSelector(ABC):
    """Abstract base class for feature selection"""

    @abstractmethod
    def optimize(self) -> FeatureSelectionResult:
        pass

    @abstractmethod
    def transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        pass


class FeatureSelector(BaseFeatureSelector):
    """
    Feature selector using Optuna for optimization.
    Implements feature selection by trying different combinations of zeroing out
    features and evaluating model performance.
    """

    def __init__(
        self,
        model: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: str,
        cv: int = 4,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        optimization_direction: str = "maximize",
        early_stopping_rounds: Optional[int] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the FeatureSelector.

        Args:
            model: Scikit-learn model or compatible estimator
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric for cross validation
            cv: Number of cross-validation folds
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            optimization_direction: Direction of optimization ("maximize" or "minimize")
            early_stopping_rounds: Number of rounds for early stopping
            feature_groups: Dictionary of feature groups for group-wise selection
        """
        self.model = model
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.optimization_direction = optimization_direction
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_groups = feature_groups
        self.logger = self._setup_logger()

        self.feature_names = (
            X.columns.tolist()
            if isinstance(X, pd.DataFrame)
            else [f"feature{i}" for i in range(X.shape[1])]
        )

        self._validate_inputs()
        self._setup_study()

    def _setup_study(self) -> None:
        """Initialize Optuna study"""
        self.study = optuna.create_study(
            direction="maximize"
            if self.optimization_direction == "maximize"
            else "minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner()
            if self.early_stopping_rounds
            else None,
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration with more detailed formatting"""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _validate_inputs(self) -> None:
        """Validate input parameters with detailed error messages"""
        if not isinstance(self.cv, int) or self.cv < 2:
            raise ValueError("cv must be an integer greater than 1")

        if len(self.y) != len(self.X):
            raise ValueError(
                f"X and y must have the same number of samples. Got X: {len(self.X)}, y: {len(self.y)}"
            )

        if isinstance(self.X, pd.DataFrame):
            if self.X.isna().any().any():
                cols_with_na = self.X.columns[self.X.isna().any()].tolist()
                raise ValueError(
                    f"Input X contains missing values in columns: {cols_with_na}"
                )

        if self.optimization_direction not in ["maximize", "minimize"]:
            raise ValueError(
                "optimization_direction must be either 'maximize' or 'minimize'"
            )

    def _create_feature_mask(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create a feature mask dictionary with threshold and weights"""
        threshold = trial.suggest_float("threshold", 0.1, 0.9)

        weights = {}
        if self.feature_groups is not None:
            for group_name, features in self.feature_groups.items():
                group_weight = trial.suggest_float(f"group_{group_name}", 0.0, 1.0)
                for feature in features:
                    weights[feature] = group_weight
        else:
            for feature in self.feature_names:
                weights[feature] = trial.suggest_float(f"weight_{feature}", 0.0, 1.0)

        return {"weights": weights, "threshold": threshold}

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        try:
            feature_params = self._create_feature_mask(trial)
            weights = feature_params["weights"]
            threshold = feature_params["threshold"]

            if not any(w > threshold for w in weights.values()):
                return (
                    -float("inf")
                    if self.optimization_direction == "maximize"
                    else float("inf")
                )

            X_tmp = self.transform(self.X, feature_params)

            cv_scores_array = cross_val_score(
                self.model,
                X_tmp,
                self.y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )

            mean_score = float(np.mean(cv_scores_array))
            std_score = float(np.std(cv_scores_array))

            n_features_above_threshold = sum(w > threshold for w in weights.values())
            sparsity_penalty = 0.01 * (
                n_features_above_threshold / len(self.feature_names)
            )

            final_score = mean_score - sparsity_penalty

            self.logger.debug(
                f"Trial {trial.number}: Score = {mean_score:.4f} ± {std_score:.4f}, "
                f"Features above threshold = {n_features_above_threshold}/{len(weights)}, "
            )

            trial.set_user_attr("cv_scores", cv_scores_array.tolist())
            trial.set_user_attr("std_score", std_score)
            trial.set_user_attr("n_features", n_features_above_threshold)

            return float(
                final_score
                if self.optimization_direction == "maximize"
                else -final_score
            )

        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> FeatureSelectionResult:
        """Optimization with continuous feature weights"""
        start_time = datetime.now()
        self.logger.info(f"Starting optimization with {n_trials} trials...")

        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
            callbacks=[self._optimization_callback]
            if self.early_stopping_rounds
            else None,
        )

        best_trial = self.study.best_trial
        threshold = best_trial.params["threshold"]

        # Get feature weights from best trial
        weights = {}
        for feature in self.feature_names:
            param_name = f"weight_{feature}"
            if param_name in best_trial.params:
                weights[feature] = best_trial.params[param_name]
            elif self.feature_groups:
                for group_name, features in self.feature_groups.items():
                    if feature in features:
                        weights[feature] = best_trial.params[f"group_{group_name}"]
                        break

        # Determine selected and zeroed features based on threshold
        selected_features = [
            feature for feature, weight in weights.items() if weight > threshold
        ]

        zero_out_features = [
            feature for feature, weight in weights.items() if weight <= threshold
        ]

        execution_time = (datetime.now() - start_time).total_seconds()

        result = FeatureSelectionResult(
            selected_features=selected_features,
            zero_out_features=zero_out_features,
            best_score=best_trial.value
            if self.optimization_direction == "maximize"
            else -best_trial.value,
            best_trial_params=best_trial.params,
            study=self.study,
            execution_time=execution_time,
            cv_scores=best_trial.user_attrs["cv_scores"],
        )

        self.logger.info(
            f"Optimization finished:\n"
            f"Best score: {result.best_score:.4f}\n"
            f"Selected features: {len(selected_features)}/{len(self.feature_names)}\n"
            f"Execution time: {execution_time:.2f}s"
        )

        return result

    def _optimization_callback(
        self, study: optuna.study.Study, trial: optuna.trial.Trial
    ) -> None:
        """Callback for early stopping"""
        if (
            self.early_stopping_rounds
            and len(study.trials) >= self.early_stopping_rounds
        ):
            recent_scores = [
                t.value for t in study.trials[-self.early_stopping_rounds :]
            ]
            if len(set(recent_scores)) == 1:
                study.stop()

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using feature weights and threshold"""
        if feature_params is None:
            if not hasattr(self, "study") or self.study is None:
                raise RuntimeError(
                    "Must run optimize() before transform() without feature_params"
                )

            best_trial = self.study.best_trial
            threshold = best_trial.params["threshold"]
            weights = {}

            if self.feature_groups is not None:
                for group_name, features in self.feature_groups.items():
                    group_weight = best_trial.params[f"group_{group_name}"]
                    for feature in features:
                        weights[feature] = group_weight
            else:
                for feature in self.feature_names:
                    weights[feature] = best_trial.params[f"weight_{feature}"]

            feature_params = {"weights": weights, "threshold": threshold}

        X_new = X.copy()
        weights = feature_params["weights"]
        threshold = feature_params["threshold"]

        if isinstance(X, pd.DataFrame):
            for feature, weight in weights.items():
                if weight <= threshold:
                    X_new[feature] = 0
        else:
            weight_array = np.array([weights[f] for f in self.feature_names])
            mask = weight_array <= threshold
            X_new[:, mask] = 0

        return X_new

    def get_feature_importance(self) -> pd.DataFrame:
        """Enhanced feature importance calculation with strict type handling"""
        if not hasattr(self, "study") or self.study is None:
            raise RuntimeError("Must run optimize() before getting feature importance")

        feature_stats: Dict[str, Dict[str, Union[int, List[float]]]] = {
            name: {"keep_count": 0, "scores": []} for name in self.feature_names
        }

        for trial in self.study.trials:
            if (
                trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
            ):
                score = float(trial.value)
                if self.optimization_direction != "maximize":
                    score = -score

                for name, param in zip(self.feature_names, trial.params.values()):
                    if param == "keep":
                        stats = feature_stats[name]
                        keep_count: int = cast(int, stats.get("keep_count", 0))
                        scores: List[float] = cast(List[float], stats.get("scores", []))
                        stats["keep_count"] = keep_count + 1
                        scores.append(score)
                        stats["scores"] = scores

        importance_data = []
        for name, stats in feature_stats.items():
            keep_count = cast(int, stats["keep_count"])
            scores = cast(List[float], stats["scores"])
            scores_array = (
                np.array(scores, dtype=np.float64) if scores else np.array([0.0])
            )

            importance_data.append(
                {
                    "feature": name,
                    "selection_frequency": float(keep_count) / len(self.study.trials),
                    "mean_score_when_selected": float(np.mean(scores_array)),
                    "std_score_when_selected": float(np.std(scores_array)),
                    "times_selected": keep_count,
                }
            )

        return (
            pd.DataFrame(importance_data)
            .sort_values("selection_frequency", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk"""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureSelector":
        """Load model from disk with type checking"""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return cast(FeatureSelector, loaded)
