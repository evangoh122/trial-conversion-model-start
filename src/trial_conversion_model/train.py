import json
from pathlib import Path

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from trial_conversion_model.data import load_processed
from trial_conversion_model.features import TARGET

MODEL_DIR = Path("models")
TEST_SIZE = 0.25
RANDOM_STATE = 42

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}


def load_training_split():
    """Load the processed table and split it once, stratified on the target."""
    table = load_processed()
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def fit_model(X_train, y_train, params: dict | None = None) -> XGBClassifier:
    """Fit an XGBoost classifier, overriding DEFAULT_PARAMS with params."""
    settings = {**DEFAULT_PARAMS, **(params or {})}
    settings.setdefault("eval_metric", "auc")
    settings.setdefault("random_state", RANDOM_STATE)
    model = XGBClassifier(**settings)
    model.fit(X_train, y_train)
    return model


def train(params: dict | None = None, model_dir: Path = MODEL_DIR) -> dict:
    """Train the trial conversion model from the processed training table."""
    X_train, X_test, y_train, y_test = load_training_split()
    model = fit_model(X_train, y_train, params)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "model.json")
    metrics = {
        "test_auc": round(float(auc), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X_train.columns),
        "params": {**DEFAULT_PARAMS, **(params or {})},
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
