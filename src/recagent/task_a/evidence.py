from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

from recagent.llm.client import clean_text
from recagent.task_a.data_store import TaskADataStore


def rating_distribution(series: pd.Series) -> Dict[str, int]:
    counts = series.round().astype(int).value_counts().to_dict()
    return {str(k): int(counts.get(k, 0)) for k in range(1, 6)}


def summarize_item(
    store: TaskADataStore,
    domain: str,
    parent_asin: str,
    max_chars: int = 700,
) -> Dict[str, Any]:
    meta = store.item_lookup.get((str(domain), str(parent_asin)), {})
    return {
        "domain": str(domain),
        "parent_asin": str(parent_asin),
        "title": clean_text(meta.get("title", ""), 180),
        "store": clean_text(meta.get("store", ""), 80),
        "categories": clean_text(meta.get("categories", ""), 250),
        "features": clean_text(meta.get("features", ""), max_chars),
        "description": clean_text(meta.get("description", ""), max_chars),
        "price": meta.get("price", None),
    }


def compact_review_row(
    store: TaskADataStore,
    row: pd.Series,
    include_item_meta: bool = True,
) -> Dict[str, Any]:
    item = summarize_item(store, row["domain"], row["parent_asin"], max_chars=300) if include_item_meta else {}
    return {
        "domain": str(row["domain"]),
        "parent_asin": str(row["parent_asin"]),
        "rating": float(row["rating"]),
        "review_title": clean_text(row.get("review_title", ""), 120),
        "review_text": clean_text(row.get("review_text", ""), 700),
        "timestamp": int(row["timestamp"]),
        "verified_purchase": bool(row.get("verified_purchase", False)),
        "helpful_vote": int(row.get("helpful_vote", 0)),
        "item_title": item.get("title", ""),
        "item_categories": item.get("categories", ""),
    }


def build_user_evidence_packet(
    store: TaskADataStore,
    user_id: str,
    target_domain: Optional[str] = None,
    target_parent_asin: Optional[str] = None,
    max_examples: int = 12,
    n_recent: int = 4,
    n_high: int = 3,
    n_low: int = 3,
    n_same_domain: int = 4,
) -> Dict[str, Any]:
    """Build a compact, train-only evidence packet for one user."""
    train = store.train_df
    user_hist = train[train["user_id"].astype(str) == str(user_id)].copy()
    if user_hist.empty:
        raise ValueError(f"No train history found for user_id={user_id}")

    user_hist = user_hist.sort_values(["timestamp", "domain", "parent_asin"]).reset_index(drop=True)
    ratings = user_hist["rating"].astype(float)
    text_lengths = user_hist["review_text"].fillna("").map(lambda x: len(str(x).split()))
    title_lengths = user_hist["review_title"].fillna("").map(lambda x: len(str(x).split()))

    domain_counts = user_hist["domain"].value_counts().to_dict()
    domain_avg_ratings = user_hist.groupby("domain")["rating"].mean().round(3).to_dict()

    recent = user_hist.tail(n_recent)
    high = user_hist.sort_values(["rating", "timestamp"], ascending=[False, False]).head(n_high)
    low = user_hist.sort_values(["rating", "timestamp"], ascending=[True, False]).head(n_low)

    groups = [("recent", recent), ("high_rating", high), ("low_rating", low)]
    if target_domain:
        same_domain = user_hist[user_hist["domain"].astype(str) == str(target_domain)].tail(n_same_domain)
        if not same_domain.empty:
            groups.append(("same_domain", same_domain))

    seen = set()
    examples = []
    for reason, df in groups:
        for _, row in df.iterrows():
            key = (row["domain"], row["parent_asin"], int(row["timestamp"]), float(row["rating"]))
            if key in seen:
                continue
            seen.add(key)
            ex = compact_review_row(store, row)
            ex["selection_reason"] = reason
            examples.append(ex)
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    return {
        "user_id": str(user_id),
        "history_scope": "train_only",
        "n_train_reviews": int(len(user_hist)),
        "domains_seen": sorted([str(x) for x in user_hist["domain"].unique()]),
        "domain_counts": {str(k): int(v) for k, v in domain_counts.items()},
        "domain_avg_ratings": {str(k): float(v) for k, v in domain_avg_ratings.items()},
        "rating_stats": {
            "mean": float(ratings.mean()),
            "median": float(ratings.median()),
            "min": float(ratings.min()),
            "max": float(ratings.max()),
            "std": float(ratings.std()) if len(ratings) > 1 else 0.0,
            "distribution": rating_distribution(ratings),
        },
        "review_style_stats": {
            "avg_review_words": float(text_lengths.mean()),
            "median_review_words": float(text_lengths.median()),
            "avg_title_words": float(title_lengths.mean()),
            "empty_review_frac": float((text_lengths == 0).mean()),
            "verified_purchase_frac": float(user_hist["verified_purchase"].mean())
            if "verified_purchase" in user_hist
            else None,
        },
        "selected_review_examples": examples,
    }
