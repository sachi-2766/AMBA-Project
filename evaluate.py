from __future__ import annotations

import time
from typing import Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score_model(model: Any, X_train: Any, y_train: Any, X_valid: Any, y_valid: Any) -> tuple[dict[str, float], Any]:
    """Fit a model and return consistent validation metrics and predictions."""
    started = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_valid, predictions))),
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "r2": float(r2_score(y_valid, predictions)),
        "train_seconds": float(time.perf_counter() - started),
    }
    return metrics, predictions

