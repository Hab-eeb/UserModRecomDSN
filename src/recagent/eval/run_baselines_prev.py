from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    # Allow direct execution via `python src/recagent/...` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recagent.baselines.rating import fit_rating_baselines
from recagent.baselines.recommendation import fit_popularity_recommender
from recagent.eval.metrics import hit_at_k, mae, ndcg_at_k, rmse
from recagent.utils.io import dump_json, ensure_dir, load_yaml


def read_split(output_dir: Path, split: str) -> pd.DataFrame:
    path = output_dir / "splits" / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run prepare_dataset.py first.")
    return pd.read_parquet(path)


def evaluate_rating_models(train: pd.DataFrame, eval_df: pd.DataFrame, regularization: float) -> Dict[str, Dict[str, float]]:
    models = fit_rating_baselines(train, regularization=regularization)
    y = eval_df["rating"].astype(float).to_numpy()

    preds = {
        "global_mean": models.predict_global(eval_df),
        "user_mean": models.predict_user_mean(eval_df),
        "item_mean": models.predict_item_mean(eval_df),
        "user_item_bias": models.predict_bias(eval_df),
    }
    return {
        name: {"rmse": rmse(y, pred), "mae": mae(y, pred)}
        for name, pred in preds.items()
    }


def _relevant_by_user(eval_df: pd.DataFrame) -> Dict[str, Set[str]]:
    relevant: Dict[str, Set[str]] = {}
    for user_id, group in eval_df.groupby("user_id"):
        relevant[str(user_id)] = {f"{r.domain}::{r.parent_asin}" for r in group.itertuples(index=False)}
    return relevant


def _dominant_eval_domain_by_user(eval_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for user_id, group in eval_df.groupby("user_id"):
        out[str(user_id)] = str(group["domain"].mode().iloc[0])
    return out


def evaluate_popularity_recommenders(train: pd.DataFrame, eval_df: pd.DataFrame, k: int, pool_size: int) -> Dict[str, Dict[str, float]]:
    recommender = fit_popularity_recommender(train)
    relevant = _relevant_by_user(eval_df)
    eval_domain = _dominant_eval_domain_by_user(eval_df)

    results: Dict[str, List[float]] = {
        "popularity_global_hit": [],
        "popularity_global_ndcg": [],
        "popularity_domain_hit": [],
        "popularity_domain_ndcg": [],
    }

    for user_id, rel_items in relevant.items():
        recs_global = recommender.recommend_global(user_id, k=k, pool_size=pool_size)
        rel_global = [1 if item in rel_items else 0 for item in recs_global]
        results["popularity_global_hit"].append(hit_at_k(recs_global, rel_items, k))
        results["popularity_global_ndcg"].append(ndcg_at_k(rel_global, k))

        domain = eval_domain[user_id]
        recs_domain = recommender.recommend_by_domain(user_id, domain=domain, k=k, pool_size=pool_size)
        rel_domain = [1 if item in rel_items else 0 for item in recs_domain]
        results["popularity_domain_hit"].append(hit_at_k(recs_domain, rel_items, k))
        results["popularity_domain_ndcg"].append(ndcg_at_k(rel_domain, k))

    return {
        "popularity_global": {
            f"hit@{k}": float(np.mean(results["popularity_global_hit"])) if results["popularity_global_hit"] else float("nan"),
            f"ndcg@{k}": float(np.mean(results["popularity_global_ndcg"])) if results["popularity_global_ndcg"] else float("nan"),
        },
        "popularity_by_eval_domain": {
            f"hit@{k}": float(np.mean(results["popularity_domain_hit"])) if results["popularity_domain_hit"] else float("nan"),
            f"ndcg@{k}": float(np.mean(results["popularity_domain_ndcg"])) if results["popularity_domain_ndcg"] else float("nan"),
        },
    }


def route_breakdown(train: pd.DataFrame, eval_df: pd.DataFrame, cold_start_threshold: int) -> Dict[str, int]:
    train_counts = train.groupby("user_id").size().to_dict()
    train_domains = train.groupby("user_id")["domain"].apply(set).to_dict()
    counts = {
        "normal_eval_rows": 0,
        "light_cold_start_eval_rows": 0,
        "cross_domain_eval_rows": 0,
        "unknown_user_eval_rows": 0,
    }
    for row in eval_df.itertuples(index=False):
        n_train = int(train_counts.get(row.user_id, 0))
        domains = train_domains.get(row.user_id, set())
        if n_train == 0:
            counts["unknown_user_eval_rows"] += 1
        elif n_train < cold_start_threshold:
            counts["light_cold_start_eval_rows"] += 1
        elif row.domain not in domains:
            counts["cross_domain_eval_rows"] += 1
        else:
            counts["normal_eval_rows"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first milestone baseline metrics.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--use-dev-as-train-for-final", action="store_true",
                        help="For final_test only, fit baselines on train+internal_val after model selection.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = Path(config["output_dir"])
    reports_dir = ensure_dir(output_dir / "reports")

    train = read_split(output_dir, "train")
    internal = read_split(output_dir, "internal_val")
    final = read_split(output_dir, "final_test")

    k = int(config["top_k"])
    pool_size = int(config["max_popular_pool"])
    reg = float(config["rating_bias_regularization"])
    cold_threshold = int(config["cold_start_train_threshold"])

    report: Dict[str, object] = {
        "config": config,
        "n_train": int(len(train)),
        "n_internal_val": int(len(internal)),
        "n_final_test": int(len(final)),
        "internal_val": {},
        "final_test": {},
        "notes": [
            "All baselines use train-only aggregates unless --use-dev-as-train-for-final is set.",
            "Popularity recommendations exclude items already seen in the training split.",
            "Use internal_val for iteration. Keep final_test untouched until the paper/debug freeze.",
        ],
    }

    for split_name, eval_df, fit_df in [
        ("internal_val", internal, train),
        ("final_test", final, pd.concat([train, internal], ignore_index=True) if args.use_dev_as_train_for_final else train),
    ]:
        report[split_name] = {
            "rating_baselines": evaluate_rating_models(fit_df, eval_df, regularization=reg),
            "recommendation_baselines": evaluate_popularity_recommenders(fit_df, eval_df, k=k, pool_size=pool_size),
            "route_breakdown": route_breakdown(fit_df, eval_df, cold_start_threshold=cold_threshold),
        }

    out = reports_dir / "baseline_report.json"
    dump_json(report, out)

    print(f"Wrote baseline report to: {out}")
    print("\nInternal validation rating baselines:")
    for name, metrics in report["internal_val"]["rating_baselines"].items():
        print(f"  {name:18s} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}")
    print("\nInternal validation recommendation baselines:")
    for name, metrics in report["internal_val"]["recommendation_baselines"].items():
        print(f"  {name:28s} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print("\nRoute breakdown:")
    print(report["internal_val"]["route_breakdown"])


if __name__ == "__main__":
    main()
