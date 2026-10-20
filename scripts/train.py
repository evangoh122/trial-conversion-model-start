import argparse
import itertools
import tempfile
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score

from trial_conversion_model.features import build_training_data
from trial_conversion_model.train import (
    DEFAULT_PARAMS,
    fit_model,
    load_training_split,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "trial-conversion-model"

# Every combination of these values is tried. Edit freely; subsample and
# colsample_bytree stay at their DEFAULT_PARAMS values unless added here.
GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 4, 8],
    "n_estimators": [300, 600],
}


def configure(experiment: str) -> None:
    """Point MLflow at the project's SQLite store and select the experiment."""
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)


def combinations(grid: dict) -> list[dict]:
    """Expand a grid into one dict per combination, in a stable order."""
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def run_name(combo: dict) -> str:
    """Readable name built from the parameters that vary in the grid."""
    return "grid-" + "-".join(f"{key}={value}" for key, value in combo.items())


def train_once() -> None:
    """Train one model with the default parameters."""
    configure(EXPERIMENT)

    # Enable XGBoost autologging
    mlflow.xgboost.autolog(
        log_models=True,
        log_model_signatures=True,
        log_input_examples=True,
    )

    # Prepare training data
    build_training_data()

    with mlflow.start_run(run_name="xgboost-training"):
        results = train()

        # Log held-out test metric
        mlflow.log_metric("test_auc", results["test_auc"])

        print(results)
        print("Run ID:", mlflow.active_run().info.run_id)
        print("Tracking URI:", mlflow.get_tracking_uri())


def sweep(grid: dict = GRID, max_runs: int = 0) -> None:
    """Try every combination in the grid, logging each as its own MLflow run."""
    configure(EXPERIMENT)

    build_training_data()
    X_train, X_test, y_train, y_test = load_training_split()

    combos = combinations(grid)
    if max_runs:
        combos = combos[:max_runs]

    print(f"Trying {len(combos)} combinations against {len(X_train)} training rows")

    rows = []
    for i, combo in enumerate(combos, start=1):
        params = {**DEFAULT_PARAMS, **combo}
        with mlflow.start_run(run_name=run_name(combo)):
            mlflow.set_tag("search", "grid")
            mlflow.log_params(params)

            model = fit_model(X_train, y_train, combo)
            train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            mlflow.log_metrics(
                {"train_auc": float(train_auc), "test_auc": float(test_auc)}
            )
            run_id = mlflow.active_run().info.run_id

        rows.append(
            {
                **params,
                "train_auc": float(train_auc),
                "test_auc": float(test_auc),
                "run_id": run_id,
            }
        )
        print(f"[{i}/{len(combos)}] test_auc={test_auc:.4f}  {combo}")

    results = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    best = results.iloc[0]
    best_combo = {
        key: best[key].item() if hasattr(best[key], "item") else best[key]
        for key in grid
    }

    with mlflow.start_run(run_name="best-model"):
        mlflow.set_tags({"selection": "best_of_grid", "source_run_id": best["run_id"]})
        mlflow.log_params({**DEFAULT_PARAMS, **best_combo})
        mlflow.log_metric("test_auc", float(best["test_auc"]))
        mlflow.log_metric("train_auc", float(best["train_auc"]))

        model = fit_model(X_train, y_train, best_combo)

        # Declare count columns as doubles in the signature so a batch with
        # missing values at inference time does not fail schema enforcement.
        example = X_train.head(5).astype(
            {col: "float64" for col in X_train.select_dtypes("integer").columns}
        )
        signature = infer_signature(example, model.predict_proba(example)[:, 1])
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=example,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sweep_results.csv"
            results.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

        best_model_run_id = mlflow.active_run().info.run_id

    print("\nSweep results (best first):")
    print(results.drop(columns=["run_id"]).head(10).to_string(index=False))
    print(f"\nBest params: {best_combo}")
    print(f"Best test_auc: {best['test_auc']:.4f} (run {best['run_id']})")
    print(f"Best model logged in run {best_model_run_id}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Grid-search the hyperparameters instead of training once",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Cap the number of combinations tried (0 = every combination)",
    )
    args = parser.parse_args()

    if args.sweep:
        sweep(max_runs=args.max_runs)
    else:
        train_once()
