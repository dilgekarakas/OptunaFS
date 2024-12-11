import optuna
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from dataclasses import dataclass
from sklearn.base import BaseEstimator
import logging


@dataclass
class FeatureSelectionResult:
    selected_features: List[str]
    zero_out_features: List[str]
    best_score: float
    best_trial_params: Dict
    study: optuna.study.Study


class FeatureSelector:
    """
    Feature selector using Optuna for optimization.
    Implements feature selection by trying different combinations of zeroing out features
    and evaluating model performance.
    """

    def __init__(
        self,
        model: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: str,
        cv: int = 4,
        random_state: Optional[int] = None,
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
        """
        self.model = model
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.logger = self._setup_logger()

        # Set feature names
        self.feature_names = (
            X.columns.tolist()
            if isinstance(X, pd.DataFrame)
            else [f"feature{i}" for i in range(X.shape[1])]
        )

        self._validate_inputs()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _validate_inputs(self):
        """Validate input parameters"""
        if not isinstance(self.cv, int) or self.cv < 2:
            raise ValueError("cv must be an integer greater than 1")

        if len(self.y) != len(self.X):
            raise ValueError("X and y must have the same number of samples")

        if isinstance(self.X, pd.DataFrame):
            if self.X.isna().any().any():
                raise ValueError("Input X contains missing values")

    def _create_feature_mask(self, trial: optuna.Trial) -> List[bool]:
        """Create a boolean mask for feature selection"""
        return [
            trial.suggest_categorical(f"feature_{i}", ["keep", "zero_out"]) == "keep"
            for i, _ in enumerate(self.feature_names)
        ]

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            float: Negative mean cross-validation score (to minimize)
        """
        # Create feature mask
        feature_mask = self._create_feature_mask(trial)

        # Create temporary dataset with zeroed out features
        X_tmp = self.X.copy()
        if isinstance(X_tmp, pd.DataFrame):
            X_tmp.iloc[:, ~np.array(feature_mask)] = 0
        else:
            X_tmp[:, ~np.array(feature_mask)] = 0

        # Calculate cross-validation score
        try:
            cv_scores = cross_val_score(
                self.model, X_tmp, self.y, cv=self.cv, scoring=self.scoring
            )
            mean_score = np.mean(cv_scores)

            # Log progress
            self.logger.debug(
                f"Trial {trial.number}: Score = {mean_score:.4f}, "
                f"Features kept = {sum(feature_mask)}"
            )

            return -mean_score  # Negative because Optuna minimizes

        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> FeatureSelectionResult:
        """
        Run the optimization process.

        Args:
            n_trials: Number of trials for optimization
            timeout: Timeout in seconds
            show_progress_bar: Whether to show progress bar

        Returns:
            FeatureSelectionResult object containing selection results
        """
        # Create study
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )

        # Get results
        best_trial = self.study.best_trial
        selected_features = [
            name
            for name, param in zip(self.feature_names, best_trial.params.values())
            if param == "keep"
        ]
        zero_out_features = [
            name
            for name, param in zip(self.feature_names, best_trial.params.values())
            if param == "zero_out"
        ]

        # Log results
        self.logger.info(
            f"Optimization finished. Best score: {-best_trial.value:.4f}, "
            f"Selected {len(selected_features)} features"
        )

        return FeatureSelectionResult(
            selected_features=selected_features,
            zero_out_features=zero_out_features,
            best_score=-best_trial.value,
            best_trial_params=best_trial.params,
            study=self.study,
        )

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data by zeroing out unselected features.

        Args:
            X: Input features to transform

        Returns:
            Transformed features with unselected features zeroed out
        """
        if not hasattr(self, "study"):
            raise RuntimeError("Must run optimize() before transform()")

        X_new = X.copy()
        zero_out_features = self.get_zero_out_features()

        if isinstance(X, pd.DataFrame):
            X_new[zero_out_features] = 0
        else:
            indices = [
                i
                for i, name in enumerate(self.feature_names)
                if name in zero_out_features
            ]
            X_new[:, indices] = 0

        return X_new

    def get_zero_out_features(self) -> List[str]:
        """Get list of features that should be zeroed out"""
        if not hasattr(self, "study"):
            raise RuntimeError("Must run optimize() before getting features")

        return [
            name
            for name, param in zip(
                self.feature_names, self.study.best_trial.params.values()
            )
            if param == "zero_out"
        ]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance based on selection frequency.

        Returns:
            DataFrame with feature importance metrics
        """
        if not hasattr(self, "study"):
            raise RuntimeError("Must run optimize() before getting feature importance")

        # Calculate selection frequency for each feature
        feature_counts = {name: 0 for name in self.feature_names}
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                for name, param in zip(self.feature_names, trial.params.values()):
                    if param == "keep":
                        feature_counts[name] += 1

        # Create importance DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": list(feature_counts.keys()),
                "selection_frequency": [
                    count / len(self.study.trials) for count in feature_counts.values()
                ],
            }
        )
        importance_df = importance_df.sort_values(
            "selection_frequency", ascending=False
        ).reset_index(drop=True)

        return importance_df
