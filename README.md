# OptunaFS: Efficient Feature Selection with Optuna

OptunaFS is a Python library that combines the power of Optuna's hyperparameter optimization with feature selection, enabling automatic discovery of the most impactful features for your machine learning models.

## 🌟 Features

- **Automated Feature Selection**: Leverages Optuna's optimization capabilities to identify the most relevant features
- **Model Agnostic**: Works with any scikit-learn compatible model
- **Easy Integration**: Seamlessly fits into existing machine learning pipelines
- **Cross-Validation Support**: Built-in cross-validation for robust feature selection
- **Flexible Scoring**: Supports various scikit-learn scoring metrics

## 🚀 Installation

```bash
pip install optunafs
```

## 📊 Quick Start

```python
from optunafs import FeatureSelector
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Initialize model
model = LogisticRegression()

# Create feature selector
selector = FeatureSelector(
    model=model,
    X=X,
    y=y,
    scoring='accuracy',
    cv=5
)

# Run optimization
selector.optimize(n_trials=100)

# Get selected features
selected_features = selector.get_best_zero_out_features()
print(f"Selected features: {selected_features}")
```

## 🔍 How It Works

OptunaFS uses Optuna's optimization framework to systematically explore different feature combinations. For each trial:

1. Features are randomly selected to be either kept or "zeroed out"
2. The model is evaluated using cross-validation with the selected features
3. Optuna's optimization algorithm learns which features contribute most to model performance
4. The process continues until the best feature combination is found

## 🛠️ Advanced Usage

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# With pandas DataFrames
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

# Using a different model and metric
selector = FeatureSelector(
    model=RandomForestClassifier(),
    X=X,
    y=y,
    scoring='roc_auc',
    cv=10
)

# More optimization trials
selector.optimize(n_trials=200)

# Transform data using selected features
X_transformed = selector.transform(X)
```

## 📈 Performance Impact

Feature selection can significantly improve model performance by:
- Reducing overfitting
- Decreasing model complexity
- Improving training speed
- Enhancing model interpretability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

```bash
# Clone the repository
git clone https://github.com/yourusername/OptunaFS.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```