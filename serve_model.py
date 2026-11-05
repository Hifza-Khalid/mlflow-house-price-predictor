"""
serve_model.py
--------------
Load models:/CaliforniaHousingPredictor@champion and predict on RAW features.
Because the model is a Pipeline, the scaler is applied automatically.
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing

MODEL_NAME = "CaliforniaHousingPredictor"
ALIAS = "champion"


def main():
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    print(f"Loaded: {model}")

    X, _ = fetch_california_housing(return_X_y=True)
    sample = X[:5]

    # Pipeline applies StandardScaler internally — pass RAW features
    preds = model.predict(sample)

    print("\nSample predictions (in $100k units):")
    for i, p in enumerate(preds):
        print(f"  row {i}: {p:.3f}  → ${p * 100_000:,.0f}")

    # Sanity check against actuals
    _, y = fetch_california_housing(return_X_y=True)
    actuals = y[:5]
    print("\nActual vs predicted:")
    for i, (a, p) in enumerate(zip(actuals, preds)):
        print(f"  row {i}: actual=${a * 100_000:,.0f}  predicted=${p * 100_000:,.0f}")


if __name__ == "__main__":
    main()