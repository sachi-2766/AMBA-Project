from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TARGET
from src.preprocessing import BigMartFeatureEngineer
from src.structural_model import StructuralSalesModel
from src.train import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, prediction)))


class OutletMRPRidge:
    """Ridge model with outlet-specific quadratic response to product price."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OutletMRPRidge":
        self.outlets_ = sorted(X["Outlet_Identifier"].unique())
        self.model_ = Ridge(alpha=1.0)
        self.model_.fit(self._matrix(X), y)
        return self

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        indicators = np.column_stack([(X["Outlet_Identifier"] == outlet).to_numpy(float) for outlet in self.outlets_])
        price = X["Item_MRP"].to_numpy(float)[:, None] / 100.0
        return np.hstack([indicators, indicators * price, indicators * price**2])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model_.predict(self._matrix(X)), 0)


CAT_COLUMNS = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Category", "Item_MRP_Bin"]
NUMERIC_COLUMNS = ["Item_Weight", "Item_Visibility_Adjusted", "Item_MRP", "Outlet_Age", "Visibility_Was_Zero"]


def catboost_frame(fitted_engineer: BigMartFeatureEngineer, X: pd.DataFrame) -> pd.DataFrame:
    """Return CatBoost-ready data using a feature engineer fitted on a fold only."""
    frame = fitted_engineer.transform(X)[CAT_COLUMNS + NUMERIC_COLUMNS].copy()
    for col in CAT_COLUMNS:
        frame[col] = frame[col].astype(str)
    return frame


def catboost_predictions(X_fit: pd.DataFrame, y_fit: np.ndarray, X_apply: pd.DataFrame, seed: int) -> tuple[np.ndarray, object]:
    from catboost import CatBoostRegressor
    engineer = BigMartFeatureEngineer().fit(X_fit)
    fit_frame, apply_frame = catboost_frame(engineer, X_fit), catboost_frame(engineer, X_apply)
    model = CatBoostRegressor(iterations=600, learning_rate=.03, depth=4, l2_leaf_reg=10, loss_function="RMSE", verbose=False, random_seed=seed, thread_count=-1)
    model.fit(fit_frame, y_fit, cat_features=CAT_COLUMNS)
    return np.maximum(model.predict(apply_frame), 0), (engineer, model)


def lightgbm_predictions(X_fit: pd.DataFrame, y_fit: np.ndarray, X_apply: pd.DataFrame, seed: int) -> tuple[np.ndarray, object]:
    from lightgbm import LGBMRegressor
    engineer = BigMartFeatureEngineer().fit(X_fit)
    fit_frame, apply_frame = catboost_frame(engineer, X_fit), catboost_frame(engineer, X_apply)
    # Explicit category vocabulary comes solely from the fold training rows.
    for col in CAT_COLUMNS:
        categories = pd.Index(fit_frame[col].unique())
        fit_frame[col] = pd.Categorical(fit_frame[col], categories=categories)
        apply_frame[col] = pd.Categorical(apply_frame[col], categories=categories)
    model = LGBMRegressor(n_estimators=350, learning_rate=.02, num_leaves=12, min_child_samples=60, subsample=.8, subsample_freq=1, reg_lambda=10, random_state=seed, verbosity=-1, n_jobs=-1)
    model.fit(fit_frame, y_fit, categorical_feature=CAT_COLUMNS)
    return np.maximum(model.predict(apply_frame), 0), (engineer, model)


def blend_weights(y: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
    """Choose non-negative weights summing to one from OOF predictions."""
    n_models = prediction_matrix.shape[1]
    objective = lambda w: np.mean((y - prediction_matrix @ w) ** 2)
    solution = minimize(objective, x0=np.repeat(1 / n_models, n_models), bounds=[(0, 1)] * n_models, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    if not solution.success:
        return np.repeat(1 / n_models, n_models)
    return solution.x


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X, y = train.drop(columns=TARGET), train[TARGET].to_numpy(float)
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    factories: dict[str, Callable[[], object]] = {
        "structural": StructuralSalesModel,
        "outlet_mrp_ridge": OutletMRPRidge,
        "gradient_boosting": lambda: make_pipeline(GradientBoostingRegressor(n_estimators=140, learning_rate=.05, max_depth=2, loss="huber", random_state=RANDOM_STATE), tree=True),
    }
    optional = {"catboost": catboost_predictions, "lightgbm": lightgbm_predictions}
    available = []
    for name in optional:
        try:
            __import__(name)
            available.append(name)
        except ImportError:
            pass
    names = list(factories) + available
    oof = {name: np.zeros(len(train)) for name in names}
    for fold, (fit_idx, valid_idx) in enumerate(kfold.split(X), 1):
        X_fit, X_valid, y_fit = X.iloc[fit_idx], X.iloc[valid_idx], y[fit_idx]
        for name, factory in factories.items():
            model = factory().fit(X_fit, y_fit)
            oof[name][valid_idx] = model.predict(X_valid)
        for name in available:
            predictor = optional[name]
            # Average three seeded runs, matching the reference's diversity idea.
            preds = [predictor(X_fit, y_fit, X_valid, seed)[0] for seed in (0, 1, 2)]
            oof[name][valid_idx] = np.mean(preds, axis=0)
        print(f"Completed fold {fold}/5")
    matrix = np.column_stack([oof[name] for name in names])
    weights = blend_weights(y, matrix)
    blended = matrix @ weights
    rows = [{"model": name, "rmse": rmse(y, oof[name]), "mae": float(mean_absolute_error(y, oof[name])), "r2": float(r2_score(y, oof[name]))} for name in names]
    rows.append({"model": "oof_weighted_blend", "rmse": rmse(y, blended), "mae": float(mean_absolute_error(y, blended)), "r2": float(r2_score(y, blended))})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("rmse").to_csv(OUTPUT_DIR / "oof_model_comparison.csv", index=False)
    pd.DataFrame({**{f"prediction_{name}": oof[name] for name in names}, "actual": y, "blend_prediction": blended}).to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    # Final fit and test prediction use no validation labels beyond the previously
    # recorded OOF blend selection.
    final_predictions: dict[str, np.ndarray] = {}
    final_artifacts: dict[str, object] = {}
    for name, factory in factories.items():
        model = factory().fit(X, y)
        final_predictions[name] = model.predict(test)
        final_artifacts[name] = model
    for name in available:
        seeds = [optional[name](X, y, test, seed) for seed in (0, 1, 2)]
        final_predictions[name] = np.mean([p[0] for p in seeds], axis=0)
        final_artifacts[name] = [p[1] for p in seeds]
    final_matrix = np.column_stack([final_predictions[name] for name in names])
    final_prediction = np.maximum(final_matrix @ weights, 0)
    submission = test[["Item_Identifier", "Outlet_Identifier"]].copy()
    submission[TARGET] = final_prediction
    if len(submission) != len(test) or not np.isfinite(final_prediction).all():
        raise ValueError("OOF ensemble submission validation failed.")
    submission.to_csv(OUTPUT_DIR / "submission_oof_ensemble.csv", index=False)
    metadata = {"models": names, "weights": {name: float(weight) for name, weight in zip(names, weights)}, "blend_oof_rmse": rmse(y, blended), "note": "All model transformations and target-derived features were fitted inside each training fold."}
    (OUTPUT_DIR / "oof_ensemble_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": final_artifacts, "weights": metadata["weights"], "model_order": names}, MODEL_DIR / "oof_ensemble_models.joblib")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
