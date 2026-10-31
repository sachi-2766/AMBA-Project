from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BigMartFeatureEngineer(BaseEstimator, TransformerMixin):
    
    required_columns = {
        "Item_Identifier", "Item_Weight", "Item_Fat_Content", "Item_Visibility",
        "Item_Type", "Item_MRP", "Outlet_Identifier", "Outlet_Establishment_Year",
        "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
    }

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BigMartFeatureEngineer":
        self._check_columns(X)
        frame = X.copy()
        self.weight_median_ = float(frame["Item_Weight"].median())
        self.visibility_nonzero_median_ = float(frame.loc[frame["Item_Visibility"] > 0, "Item_Visibility"].median())
        self.item_weight_medians_ = frame.groupby("Item_Identifier")["Item_Weight"].median().to_dict()
        # Outlet identifiers are stable attributes in this competition. The mode is
        # learned only from the fit partition; attribute-mode is the fallback.
        self.outlet_size_by_id_ = self._mode_map(frame, "Outlet_Identifier", "Outlet_Size")
        self.outlet_size_by_type_ = self._mode_map(frame, "Outlet_Type", "Outlet_Size")
        self.outlet_size_default_ = frame["Outlet_Size"].mode(dropna=True).iat[0]
        return self

    @staticmethod
    def _mode_map(frame: pd.DataFrame, group: str, value: str) -> dict[str, str]:
        clean = frame[[group, value]].dropna()
        return clean.groupby(group)[value].agg(lambda x: x.mode().iat[0]).to_dict()

    @classmethod
    def _check_columns(cls, X: pd.DataFrame) -> None:
        missing = cls.required_columns.difference(X.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        df = X.copy()
        fat_map = {"LF": "Low Fat", "low fat": "Low Fat", "Low Fat": "Low Fat",
                   "reg": "Regular", "Regular": "Regular"}
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(fat_map)
        df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Identifier"].map(self.item_weight_medians_))
        df["Item_Weight"] = df["Item_Weight"].fillna(self.weight_median_)
        df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Identifier"].map(self.outlet_size_by_id_))
        df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Type"].map(self.outlet_size_by_type_))
        df["Outlet_Size"] = df["Outlet_Size"].fillna(self.outlet_size_default_)
        df["Item_Category"] = df["Item_Identifier"].str[:2].map({"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}).fillna("Other")
        df.loc[df["Item_Category"].eq("Non-Consumable"), "Item_Fat_Content"] = "Non-Edible"
        df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]
        df["Visibility_Was_Zero"] = (df["Item_Visibility"] == 0).astype(int)
        df["Item_Visibility_Adjusted"] = df["Item_Visibility"].mask(df["Item_Visibility"] == 0, self.visibility_nonzero_median_)
        df["Item_MRP_Bin"] = pd.cut(df["Item_MRP"], bins=[0, 70, 120, 170, np.inf], labels=["Low", "Medium", "High", "Premium"])
        return df

