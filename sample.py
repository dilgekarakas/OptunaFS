import logging
import time
import warnings
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split

from feature_selector import FeatureSelector

warnings.filterwarnings("ignore")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("FeatureSelection")
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler("feature_selection.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_sample_data(n_samples: int = 1000, n_features: int = 25) -> tuple:
    """Create a sample classification dataset with some irrelevant features"""
    logger = logging.getLogger("FeatureSelection")
    logger.info(
        f"Generating synthetic dataset with {n_samples} samples and {n_features} features"
    )

    start_time = time.time()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=15,
        n_repeated=0,
        n_classes=2,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(X, columns=feature_names)

    logger.info(
        f"Dataset generation completed in {time.time() - start_time:.2f} seconds"
    )
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y


def run_sfs_benchmark(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    logger: logging.Logger,
) -> Dict:
    """Run Sequential Feature Selection benchmark"""
    logger.info("\n" + "=" * 50)
    logger.info("Starting Sequential Feature Selection (SFS) benchmark")

    start_time = time.time()

    base_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )

    sfs = SequentialFeatureSelector(
        base_model,
        n_features_to_select=10,
        direction="forward",
        scoring="roc_auc",
        cv=4,
        n_jobs=-1,
    )

    sfs.fit(X_train, y_train)

    selected_features = list(X_train.columns[sfs.get_support()])

    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)

    final_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    final_model.fit(X_train_sfs, y_train)

    train_score = final_model.score(X_train_sfs, y_train)
    test_score = final_model.score(X_test_sfs, y_test)

    execution_time = time.time() - start_time

    logger.info("\nSFS Results:")
    logger.info(f"Execution time: {execution_time:.2f} seconds")
    logger.info(f"Selected features: {selected_features}")
    logger.info(f"Number of selected features: {len(selected_features)}")
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")

    return {
        "method": "SFS",
        "execution_time": execution_time,
        "selected_features": selected_features,
        "train_score": train_score,
        "test_score": test_score,
    }


def run_optuna_selector(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Union[np.ndarray[Any, Any], pd.Series[Any]],
    y_test: pd.DataFrame,
    logger: logging.Logger,
) -> Dict:
    """Run Optuna-based feature selection"""
    logger.info("\n" + "=" * 50)
    logger.info("Starting Optuna-based feature selection")

    start_time = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )

    selector = FeatureSelector(
        model=rf_model,
        X=X_train,
        y=y_train,
        scoring="roc_auc",
        cv=4,
        random_state=42,
        n_jobs=-1,
        optimization_direction="maximize",
        early_stopping_rounds=10,
    )

    result = selector.optimize(n_trials=200, show_progress_bar=True)

    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    final_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    final_model.fit(X_train_selected, y_train)

    train_score = final_model.score(X_train_selected, y_train)
    test_score = final_model.score(X_test_selected, y_test)

    execution_time = time.time() - start_time

    logger.info("\nOptuna-based Feature Selection Results:")
    logger.info(f"Execution time: {execution_time:.2f} seconds")
    logger.info(f"Selected features: {result.selected_features}")
    logger.info(f"Number of selected features: {len(result.selected_features)}")
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")

    return {
        "method": "Optuna",
        "execution_time": execution_time,
        "selected_features": result.selected_features,
        "train_score": train_score,
        "test_score": test_score,
    }


def compare_results(
    sfs_results: Dict, optuna_results: Dict, logger: logging.Logger
) -> None:
    """Compare and display results"""
    logger.info("\n" + "=" * 50)
    logger.info("Comparison Results:")

    comparison_df = pd.DataFrame([sfs_results, optuna_results])
    comparison_df = comparison_df.set_index("method")

    logger.info("\nPerformance Comparison:")
    logger.info("\nExecution Time:")
    logger.info(f"SFS: {sfs_results['execution_time']:.2f} seconds")
    logger.info(f"Optuna: {optuna_results['execution_time']:.2f} seconds")
    logger.info(
        f"Time difference: {abs(sfs_results['execution_time'] - optuna_results['execution_time']):.2f} seconds"
    )

    logger.info("\nTest Accuracy:")
    logger.info(f"SFS: {sfs_results['test_score']:.4f}")
    logger.info(f"Optuna: {optuna_results['test_score']:.4f}")
    logger.info(
        f"Accuracy difference: {abs(sfs_results['test_score'] - optuna_results['test_score']):.4f}"
    )

    logger.info("\nNumber of Selected Features:")
    logger.info(f"SFS: {len(sfs_results['selected_features'])}")
    logger.info(f"Optuna: {len(optuna_results['selected_features'])}")

    common_features = set(sfs_results["selected_features"]).intersection(
        set(optuna_results["selected_features"])
    )
    logger.info(f"\nCommon features between methods: {len(common_features)}")
    logger.info(f"Common features: {sorted(list(common_features))}")


def main() -> None:
    logger = setup_logger()
    logger.info("Starting feature selection comparison")

    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sfs_results = run_sfs_benchmark(X_train, X_test, y_train, y_test, logger)
    optuna_results = run_optuna_selector(X_train, X_test, y_train, y_test, logger)
    compare_results(sfs_results, optuna_results, logger)

    logger.info("\nComparison completed successfully")


if __name__ == "__main__":
    main()
