from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, ID_COLUMNS, MODEL_DIR, OUTPUT_DIR, TARGET


def find_test_file() -> Path:
    files = list(DATA_DIR.glob("*.csv")) + list(Path(__file__).resolve().parents[1].glob("*.csv"))
    matches = [p for p in files if "test" in p.name.lower()]
    if not matches:
        raise FileNotFoundError("Could not find a test CSV in data/ or the project root.")
    return sorted(matches)[0]


def main() -> None:
    test = pd.read_csv(find_test_file())
    required = set(ID_COLUMNS)
    if missing := required.difference(test.columns):
        raise ValueError(f"Test data is missing identifier columns: {sorted(missing)}")
    artifact = joblib.load(MODEL_DIR / "bigmart_model.joblib")
    predictions = artifact["model"].predict(test)
    if artifact["clip_negative_predictions"]:
        predictions = np.maximum(predictions, 0)
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values.")
    submission = test[ID_COLUMNS].copy()
    submission[TARGET] = predictions
    if len(submission) != len(test) or submission.isna().any().any():
        raise ValueError("Submission validation failed: row count or missing values.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission)} rows to {output}")


if __name__ == "__main__":
    main()

