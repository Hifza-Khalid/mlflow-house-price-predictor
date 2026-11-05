import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "ridge_california_housing"
MODEL_NAME = "CaliforniaHousingPredictor"


def main():
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["metrics.val_rmse ASC"])
    if runs.empty:
        raise RuntimeError("No runs. Run: python train_ridge.py --sweep")

    best_run_id = runs.iloc[0]["run_id"]
    print(f"Best run: {best_run_id}  alpha={runs.iloc[0]['params.alpha']}")

    # 2.x — register from runs:/ URI directly
    reg = mlflow.register_model(f"runs:/{best_run_id}/ridge_model", MODEL_NAME)
    print(f"Registered {reg.name} v{reg.version}")

    # 2.x stages — work as documented
    client.transition_model_version_stage(MODEL_NAME, reg.version, "Staging")
    client.transition_model_version_stage(MODEL_NAME, reg.version, "Production",
                                           archive_existing_versions=True)
    print(f"v{reg.version} → Production")

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    print(f"Loaded: {model}")


if __name__ == "__main__":
    main()