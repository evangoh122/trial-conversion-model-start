import mlflow
import mlflow.xgboost
from pathlib import Path

from trial_conversion_model.features import build_training_data
from trial_conversion_model.train import train


if __name__ == "__main__":

    # 1. Locate project root
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # 2. Configure MLflow SQLite database
    db_path = PROJECT_ROOT / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path.as_posix()}"
    )

    # 3. Select experiment
    mlflow.set_experiment("trial-conversion-model")

    # 4. Enable XGBoost autologging
    mlflow.xgboost.autolog(
        log_models=True,
        log_model_signatures=True,
        log_input_examples=True
    )

    # 5. Prepare training data
    build_training_data()

    # 6. Start MLflow run
    with mlflow.start_run(run_name="xgboost-training"):

        results = train()

        # Log held-out test metric
        mlflow.log_metric("test_auc", results["test_auc"])

        print(results)
        print("Run ID:", mlflow.active_run().info.run_id)
        print("Tracking URI:", mlflow.get_tracking_uri())