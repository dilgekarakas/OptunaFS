# OptunaFS

<p align="center">
  <img src="https://raw.githubusercontent.com/dilgekarakas/OptunaFS/refs/heads/main/assets/images/logo.svg" width="200" alt="OptunaFS Logo">
</p>

OptunaFS is a Python library that enhances feature selection in machine learning workflows by leveraging Optuna's optimization framework. It provides an intelligent way to identify and select the most impactful features for your models.

## Key Features

- Automated feature selection through Optuna's hyperparameter optimization
- Supports any scikit-learn compatible estimator
- Built-in cross-validation for robust feature evaluation
- Support for feature grouping and early stopping
- Detailed feature importance analysis
- Type-safe implementation with comprehensive error handling

## Installation

```bash
pip install optunafs
```

## Usage Example

```python
from optunafs import FeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create example dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=25,
    n_informative=10,
    random_state=42
)

# Initialize model
model = RandomForestClassifier(random_state=42)

# Create feature selector
selector = FeatureSelector(
    model=model,
    X=X,
    y=y,
    scoring='roc_auc',
    cv=4,
    optimization_direction='maximize'
)

# Run optimization
result = selector.optimize(n_trials=100)

# Get selected features
print(f"Selected features: {result.selected_features}")
print(f"Best score: {result.best_score:.4f}")

# Transform data using selected features
X_transformed = selector.transform(X)
```

## Useful Features

### Feature Groups

You can define groups of features that should be selected together:

```python
feature_groups = {
    'group1': ['feature1', 'feature2', 'feature3'],
    'group2': ['feature4', 'feature5']
}

selector = FeatureSelector(
    model=model,
    X=X,
    y=y,
    scoring='accuracy',
    feature_groups=feature_groups
)
```

### Early Stopping

Enable early stopping to automatically halt optimization when no improvement is seen:

```python
selector = FeatureSelector(
    model=model,
    X=X,
    y=y,
    scoring='accuracy',
    early_stopping_rounds=10
)
```

### Feature Importance Analysis

Get detailed insights into feature selection patterns:

```python
importance_df = selector.get_feature_importance()
print(importance_df.sort_values('selection_frequency', ascending=False))
```

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/optunafs.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

This project is licensed under the terms of the MIT license.