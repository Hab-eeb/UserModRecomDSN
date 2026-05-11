from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


@dataclass
class PopularityRecommender:
    top_items_global: List[Tuple[str, str]]
    top_items_by_domain: Dict[str, List[Tuple[str, str]]]
    user_seen_items: Dict[str, Set[Tuple[str, str]]]

    def recommend_global(self, user_id: str, k: int = 10, pool_size: int = 500) -> List[str]:
        seen = self.user_seen_items.get(user_id, set())
        recs: List[str] = []
        for domain, item in self.top_items_global[:pool_size]:
            if (domain, item) not in seen:
                recs.append(f"{domain}::{item}")
            if len(recs) >= k:
                break
        return recs

    def recommend_by_domain(self, user_id: str, domain: str, k: int = 10, pool_size: int = 500) -> List[str]:
        seen = self.user_seen_items.get(user_id, set())
        recs: List[str] = []
        for d, item in self.top_items_by_domain.get(domain, [])[:pool_size]:
            if (d, item) not in seen:
                recs.append(f"{d}::{item}")
            if len(recs) >= k:
                break
        return recs


def fit_popularity_recommender(train: pd.DataFrame) -> PopularityRecommender:
    # Popularity is based only on train interactions to avoid temporal leakage.
    pop = (
        train.groupby(["domain", "parent_asin"])
        .agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"))
        .reset_index()
    )
    pop["pop_score"] = pop["n_ratings"] * pop["mean_rating"]
    pop = pop.sort_values(["pop_score", "n_ratings", "mean_rating"], ascending=False)
    top_items_global = [(r.domain, r.parent_asin) for r in pop.itertuples(index=False)]

    top_items_by_domain: Dict[str, List[Tuple[str, str]]] = {}
    for domain, group in pop.groupby("domain", sort=False):
        group = group.sort_values(["pop_score", "n_ratings", "mean_rating"], ascending=False)
        top_items_by_domain[domain] = [(r.domain, r.parent_asin) for r in group.itertuples(index=False)]

    user_seen_items: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for r in train[["user_id", "domain", "parent_asin"]].itertuples(index=False):
        user_seen_items[r.user_id].add((r.domain, r.parent_asin))

    return PopularityRecommender(
        top_items_global=top_items_global,
        top_items_by_domain=top_items_by_domain,
        user_seen_items=dict(user_seen_items),
    )
