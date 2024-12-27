import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from feature_selector import FeatureSelector


def main():
    # Load sample dataset
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Initialize model
    model = LogisticRegression(random_state=42)

    # Create feature selector
    print("\nInitializing feature selection...")
    selector = FeatureSelector(
        model=model,
        X=X_train_scaled,
        y=y_train,
        scoring="accuracy",
        cv=5,
        random_state=42,
    )

    # Run optimization
    print("\nOptimizing feature selection...")
    result = selector.optimize(n_trials=50, show_progress_bar=True)

    # Print results
    print("\nFeature Selection Results:")
    print(f"Best CV Score: {result.best_score:.4f}")
    print(f"\nSelected Features ({len(result.selected_features)}):")
    for feature in result.selected_features:
        print(f"- {feature}")

    print(f"\nZeroed Out Features ({len(result.zero_out_features)}):")
    for feature in result.zero_out_features:
        print(f"- {feature}")

    # Get feature importance
    importance_df = selector.get_feature_importance()
    print("\nFeature Importance (Top 10):")
    print(importance_df.head(10))

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(importance_df["feature"][:10], importance_df["selection_frequency"][:10])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Feature Selection Frequencies")
    plt.tight_layout()
    plt.show()

    # Compare model performance
    print("\nComparing model performance...")

    # Original model (all features)
    model_all = LogisticRegression(random_state=42)
    model_all.fit(X_train_scaled, y_train)
    y_pred_all = model_all.predict(X_test_scaled)

    print("\nClassification Report (All Features):")
    print(classification_report(y_test, y_pred_all))

    # Selected features model
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    model_selected = LogisticRegression(random_state=42)
    model_selected.fit(X_train_selected, y_train)
    y_pred_selected = model_selected.predict(X_test_selected)

    print("\nClassification Report (Selected Features):")
    print(classification_report(y_test, y_pred_selected))

    # Calculate feature reduction
    reduction_percent = (len(result.zero_out_features) / len(X.columns)) * 100
    print(f"\nFeature Reduction: {reduction_percent:.1f}%")


if __name__ == "__main__":
    main()
