import mlflow, mlflow.sklearn

EXPERIMENT_NAME = "ridge_california_housing"


def main():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["metrics.val_rmse ASC"])
    if runs.empty:
        raise RuntimeError("No runs. Run: python train_ridge.py --sweep")

    best = runs.iloc[0]
    print("TOP RUNS:")
    print(runs[["run_id", "tags.mlflow.runName", "params.alpha", "metrics.val_rmse"]].head(5).to_string(index=False))
    print(f"\nBEST: alpha={best['params.alpha']}  rmse={best['metrics.val_rmse']}")
    print(f"run_id={best['run_id']}")


if __name__ == "__main__":
    main()