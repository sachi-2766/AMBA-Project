from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


TARGET = "Item_Outlet_Sales"


def recover_base_prices(frame: pd.DataFrame, target: Iterable[float]) -> dict[str, float]:
    work = frame[["Item_Identifier", "Item_MRP"]].copy()
    work["_target"] = np.asarray(target)
    recovered: dict[str, float] = {}
    for item, group in work.groupby("Item_Identifier", sort=False):
        sales = group["_target"].to_numpy(dtype=float)
        median_mrp = float(group["Item_MRP"].median())
        candidates = median_mrp + np.arange(-40, 41) / 10
        errors = np.max(np.abs(sales[:, None] / candidates[None, :] - np.round(sales[:, None] / candidates[None, :])), axis=0)
        price = float(candidates[int(np.argmin(errors))])
        if float(np.min(errors)) < 0.01 and price > 0:
            recovered[str(item)] = price
    return recovered


@dataclass
class StructuralSalesModel:
    """Predict sales as recovered item price × fitted outlet unit-sales rate."""

    base_prices: dict[str, float] | None = None
    outlet_rates: pd.Series | None = None
    global_rate: float | None = None

    def fit(self, X: pd.DataFrame, y: Iterable[float]) -> "StructuralSalesModel":
        y_array = np.asarray(y, dtype=float)
        self.base_prices = recover_base_prices(X, y_array)
        price = self._price(X)
        rate = y_array / price
        self.outlet_rates = pd.Series(rate).groupby(X["Outlet_Identifier"].to_numpy()).mean()
        self.global_rate = float(np.mean(rate))
        return self

    def _price(self, X: pd.DataFrame) -> np.ndarray:
        if self.base_prices is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return X["Item_Identifier"].map(self.base_prices).fillna(X["Item_MRP"]).to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.outlet_rates is None or self.global_rate is None:
            raise RuntimeError("Model must be fitted before prediction.")
        rates = X["Outlet_Identifier"].map(self.outlet_rates).fillna(self.global_rate).to_numpy(dtype=float)
        return np.maximum(self._price(X) * rates, 0)

