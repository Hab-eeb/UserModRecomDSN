from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RatingBaselineModels:
    global_mean: float
    user_mean: Dict[str, float]
    item_mean: Dict[str, float]
    user_bias: Dict[str, float]
    item_bias: Dict[str, float]
    regularization: float = 10.0

    def predict_global(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.global_mean, dtype=float)

    def predict_user_mean(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.user_mean.get(u, self.global_mean) for u in df["user_id"]], dtype=float)

    def predict_item_mean(self, df: pd.DataFrame) -> np.ndarray:
        keys = zip(df["domain"], df["parent_asin"])
        return np.array([self.item_mean.get(f"{d}::{a}", self.global_mean) for d, a in keys], dtype=float)

    def predict_bias(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            item_key = f"{row['domain']}::{row['parent_asin']}"
            pred = self.global_mean + self.user_bias.get(row["user_id"], 0.0) + self.item_bias.get(item_key, 0.0)
            preds.append(min(5.0, max(1.0, pred)))
        return np.asarray(preds, dtype=float)


def fit_rating_baselines(train: pd.DataFrame, regularization: float = 10.0) -> RatingBaselineModels:
    global_mean = float(train["rating"].mean())

    user_mean = train.groupby("user_id")["rating"].mean().to_dict()
    item_mean_series = train.groupby(["domain", "parent_asin"])["rating"].mean()
    item_mean = {f"{d}::{a}": float(v) for (d, a), v in item_mean_series.items()}

    user_stats = train.groupby("user_id")["rating"].agg(["sum", "count"])
    user_bias = ((user_stats["sum"] - user_stats["count"] * global_mean) / (user_stats["count"] + regularization)).to_dict()

    item_stats = train.groupby(["domain", "parent_asin"])["rating"].agg(["sum", "count"])
    item_bias = {}
    for (d, a), row in item_stats.iterrows():
        item_bias[f"{d}::{a}"] = float((row["sum"] - row["count"] * global_mean) / (row["count"] + regularization))

    return RatingBaselineModels(
        global_mean=global_mean,
        user_mean={str(k): float(v) for k, v in user_mean.items()},
        item_mean=item_mean,
        user_bias={str(k): float(v) for k, v in user_bias.items()},
        item_bias=item_bias,
        regularization=regularization,
    )
