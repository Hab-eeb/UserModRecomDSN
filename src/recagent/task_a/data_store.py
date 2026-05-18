from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class TaskADataConfig:
    train_path: str
    internal_val_path: str
    shadow_test_path: Optional[str]
    items_path: str


class TaskADataStore:
    """Thin data access layer for Task A API/demo and batch evaluation."""

    def __init__(self, config: TaskADataConfig):
        self.config = config
        self.train_df = pd.read_parquet(config.train_path)
        self.internal_val_df = pd.read_parquet(config.internal_val_path)
        self.shadow_test_df = (
            pd.read_parquet(config.shadow_test_path)
            if config.shadow_test_path and Path(config.shadow_test_path).exists()
            else None
        )
        self.items_df = pd.read_parquet(config.items_path)
        self.items_df = self.items_df.drop_duplicates(["domain", "parent_asin"]).copy()
        self.item_lookup = {
            (str(row["domain"]), str(row["parent_asin"])): row.to_dict()
            for _, row in self.items_df.iterrows()
        }

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TaskADataStore":
        return cls(
            TaskADataConfig(
                train_path=cfg["train_path"],
                internal_val_path=cfg["internal_val_path"],
                shadow_test_path=cfg.get("shadow_test_path"),
                items_path=cfg["items_path"],
            )
        )

    def get_eval_df(self, split: str = "internal_val") -> pd.DataFrame:
        if split == "internal_val":
            return self.internal_val_df
        if split == "shadow_test":
            if self.shadow_test_df is None:
                raise ValueError("shadow_test split is not loaded or path does not exist.")
            return self.shadow_test_df
        raise ValueError("split must be 'internal_val' or 'shadow_test'")

    def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        counts = (
            self.train_df.groupby("user_id")
            .agg(n_train_reviews=("rating", "size"), avg_train_rating=("rating", "mean"))
            .reset_index()
            .sort_values(["n_train_reviews", "user_id"], ascending=[False, True])
        )
        counts = counts.iloc[offset : offset + limit]
        return counts.to_dict(orient="records")

    def list_eval_items_for_user(
        self,
        user_id: str,
        split: str = "internal_val",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        eval_df = self.get_eval_df(split)
        rows = eval_df[eval_df["user_id"].astype(str) == str(user_id)].copy()
        if rows.empty:
            return []
        rows = rows.sort_values(["timestamp", "domain", "parent_asin"]).head(limit)
        records = []
        for _, row in rows.iterrows():
            meta = self.item_lookup.get((str(row["domain"]), str(row["parent_asin"])), {})
            records.append(
                {
                    "user_id": str(user_id),
                    "split": split,
                    "domain": str(row["domain"]),
                    "parent_asin": str(row["parent_asin"]),
                    "item_title": str(meta.get("title", "")),
                    "item_categories": str(meta.get("categories", "")),
                    "true_rating": float(row["rating"]),
                    "timestamp": int(row["timestamp"]),
                }
            )
        return records

    def get_ground_truth(
        self,
        user_id: str,
        target_domain: str,
        target_parent_asin: str,
        split: str = "internal_val",
    ) -> Optional[Dict[str, Any]]:
        eval_df = self.get_eval_df(split)
        rows = eval_df[
            (eval_df["user_id"].astype(str) == str(user_id))
            & (eval_df["domain"].astype(str) == str(target_domain))
            & (eval_df["parent_asin"].astype(str) == str(target_parent_asin))
        ].copy()
        if rows.empty:
            return None
        row = rows.sort_values(["timestamp"]).iloc[0]
        return {
            "split": split,
            "rating": float(row["rating"]),
            "review_title": str(row.get("review_title", "") or ""),
            "review_text": str(row.get("review_text", "") or ""),
            "timestamp": int(row["timestamp"]),
        }
