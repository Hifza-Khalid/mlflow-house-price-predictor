import argparse, os, tempfile, warnings
import mlflow, mlflow.sklearn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
EXPERIMENT_NAME = "ridge_california_housing"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def load_and_split(test_size, random_state):
    X, y = fetch_california_housing(return_X_y=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=random_state)
    return {"X_tr": X_tr, "y_tr": y_tr, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}


def run_one(alpha, data):
    with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipe.fit(data["X_tr"], data["y_tr"])

        val_pred = pipe.predict(data["X_val"])
        test_pred = pipe.predict(data["X_test"])

        vm_rmse = float(np.sqrt(mean_squared_error(data["y_val"], val_pred)))
        tm_rmse = float(np.sqrt(mean_squared_error(data["y_test"], test_pred)))

        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("val_rmse", round(vm_rmse, 4))
        mlflow.log_metric("test_rmse", round(tm_rmse, 4))
        mlflow.log_metric("val_r2", round(r2_score(data["y_val"], val_pred), 4))

        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(val_pred, data["y_val"] - val_pred, alpha=0.4, s=8)
            ax.axhline(0, color="red", ls="--")
            p = os.path.join(tmp, f"residuals_{alpha}.png")
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            mlflow.log_artifact(p)

        # 2.x style — works out of the box
        mlflow.sklearn.log_model(pipe, artifact_path="ridge_model")

        print(f"alpha={alpha:<8} val_rmse={vm_rmse:.4f} test_rmse={tm_rmse:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_and_split(0.30, 42)
    if args.sweep:
        for a in ALPHAS:
            run_one(a, data)
    else:
        run_one(args.alpha, data)


if __name__ == "__main__":
    main()