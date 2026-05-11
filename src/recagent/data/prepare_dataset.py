from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from recagent.utils.io import dump_json, ensure_dir, iter_jsonl, load_yaml

REVIEW_COLUMNS = [
    "domain",
    "user_id",
    "asin",
    "parent_asin",
    "rating",
    "review_title",
    "review_text",
    "timestamp",
    "verified_purchase",
    "helpful_vote",
]

META_COLUMNS = [
    "domain",
    "parent_asin",
    "main_category",
    "title",
    "store",
    "categories",
    "features",
    "description",
    "price",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_clean_text(x) for x in value if x is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_review(row: Dict[str, Any], domain: str) -> Optional[Dict[str, Any]]:
    user_id = row.get("user_id")
    parent_asin = row.get("parent_asin") or row.get("asin")
    rating = row.get("rating")
    timestamp = row.get("timestamp")

    if not user_id or not parent_asin or rating is None or timestamp is None:
        return None

    try:
        rating = float(rating)
        timestamp = int(timestamp)
    except (TypeError, ValueError):
        return None

    return {
        "domain": domain,
        "user_id": str(user_id),
        "asin": str(row.get("asin") or parent_asin),
        "parent_asin": str(parent_asin),
        "rating": rating,
        "review_title": _clean_text(row.get("title")),
        "review_text": _clean_text(row.get("text")),
        "timestamp": timestamp,
        "verified_purchase": bool(row.get("verified_purchase", False)),
        "helpful_vote": int(row.get("helpful_vote") or 0),
    }


def normalize_meta(row: Dict[str, Any], domain: str) -> Optional[Dict[str, Any]]:
    parent_asin = row.get("parent_asin")
    if not parent_asin:
        return None
    price = row.get("price")
    try:
        price = float(price) if price not in (None, "", "None") else None
    except (TypeError, ValueError):
        price = None

    return {
        "domain": domain,
        "parent_asin": str(parent_asin),
        "main_category": _clean_text(row.get("main_category")),
        "title": _clean_text(row.get("title")),
        "store": _clean_text(row.get("store")),
        "categories": _clean_text(row.get("categories")),
        "features": _clean_text(row.get("features")),
        "description": _clean_text(row.get("description")),
        "price": price,
    }


def stream_reviews_to_parquet(config: Dict[str, Any], output_dir: Path) -> Path:
    parquet_dir = ensure_dir(output_dir / "reviews_parquet")
    chunk_size = int(config.get("chunk_size", 50_000))
    max_reviews_per_domain = config.get("max_reviews_per_domain")

    written_files: List[str] = []
    for domain, rel_path in config["raw_reviews"].items():
        path = Path(rel_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw review file for {domain}: {path}. Update configs/baseline.yaml."
            )
        rows: List[Dict[str, Any]] = []
        part = 0
        pbar = tqdm(iter_jsonl(path, limit=max_reviews_per_domain), desc=f"reviews:{domain}")
        for raw in pbar:
            norm = normalize_review(raw, domain)
            if norm is None:
                continue
            rows.append(norm)
            if len(rows) >= chunk_size:
                out = parquet_dir / f"{domain}_part_{part:05d}.parquet"
                pd.DataFrame(rows, columns=REVIEW_COLUMNS).to_parquet(out, index=False)
                written_files.append(str(out))
                rows.clear()
                part += 1
        if rows:
            out = parquet_dir / f"{domain}_part_{part:05d}.parquet"
            pd.DataFrame(rows, columns=REVIEW_COLUMNS).to_parquet(out, index=False)
            written_files.append(str(out))

    if not written_files:
        raise RuntimeError("No review parquet files were written. Check the raw data paths/schema.")
    return parquet_dir


def build_filtered_splits(config: Dict[str, Any], output_dir: Path, parquet_dir: Path) -> Dict[str, Any]:
    con = duckdb.connect()
    glob_path = str(parquet_dir / "*.parquet")
    con.execute("CREATE OR REPLACE VIEW reviews AS SELECT * FROM read_parquet(?)", [glob_path])

    min_user = int(config["min_user_reviews_total"])
    min_item = int(config["min_item_reviews_total"])
    min_train = int(config["min_train_reviews"])
    internal_frac = float(config["internal_frac"])
    shadow_frac = float(config.get("shadow_frac", 0.0))
    final_frac = float(config["final_frac"])

    if internal_frac < 0 or shadow_frac < 0 or final_frac < 0:
        raise ValueError("Split fractions must be non-negative.")
    if internal_frac + shadow_frac + final_frac >= 1.0:
        raise ValueError(
            "internal_frac + shadow_frac + final_frac must be < 1.0 so train keeps data."
        )

    # Keep only users/items with enough signal for meaningful chronological evaluation.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE eligible_users AS
        SELECT
            user_id,
            COUNT(*) AS n_reviews,
            COUNT(DISTINCT domain) AS n_domains
        FROM reviews
        GROUP BY user_id
        HAVING COUNT(*) >= ?
        """,
        [max(min_user, min_train + 2 + (1 if shadow_frac > 0 else 0))],
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE eligible_items AS
        SELECT domain, parent_asin, COUNT(*) AS n_reviews
        FROM reviews
        GROUP BY domain, parent_asin
        HAVING COUNT(*) >= ?
        """,
        [min_item],
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE filtered AS
        SELECT r.*
        FROM reviews r
        INNER JOIN eligible_users u ON r.user_id = u.user_id
        INNER JOIN eligible_items i
          ON r.domain = i.domain AND r.parent_asin = i.parent_asin
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE split_reviews AS
        WITH ordered AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp, domain, parent_asin) AS rn,
                COUNT(*) OVER (PARTITION BY user_id) AS n_user_reviews
            FROM filtered
        ), counts AS (
            SELECT
                *,
                GREATEST(1, CAST(FLOOR(n_user_reviews * {final_frac}) AS INTEGER)) AS n_final,
                CASE
                    WHEN {shadow_frac} > 0
                    THEN GREATEST(1, CAST(FLOOR(n_user_reviews * {shadow_frac}) AS INTEGER))
                    ELSE 0
                END AS n_shadow,
                GREATEST(1, CAST(FLOOR(n_user_reviews * {internal_frac}) AS INTEGER)) AS n_internal
            FROM ordered
        )
        SELECT
            domain, user_id, asin, parent_asin, rating, review_title, review_text,
            timestamp, verified_purchase, helpful_vote, n_user_reviews,
            CASE
                WHEN rn <= n_user_reviews - n_final - n_shadow - n_internal THEN 'train'
                WHEN rn <= n_user_reviews - n_final - n_shadow THEN 'internal_val'
                WHEN rn <= n_user_reviews - n_final THEN 'shadow_test'
                ELSE 'final_test'
            END AS split
        FROM counts
        WHERE n_user_reviews - n_final - n_shadow - n_internal >= {min_train}
        """
    )

    splits_dir = ensure_dir(output_dir / "splits")
    split_names = ["train", "internal_val"]
    if shadow_frac > 0:
        split_names.append("shadow_test")
    split_names.append("final_test")

    for split in split_names:
        out = splits_dir / f"{split}.parquet"
        con.execute("COPY (SELECT * FROM split_reviews WHERE split = ?) TO ? (FORMAT PARQUET)", [split, str(out)])

    # Useful summary for the paper and for leakage checks.
    summary_rows = con.execute(
        """
        SELECT split, domain, COUNT(*) AS n_reviews,
               COUNT(DISTINCT user_id) AS n_users,
               COUNT(DISTINCT parent_asin) AS n_items,
               AVG(rating) AS avg_rating
        FROM split_reviews
        GROUP BY split, domain
        ORDER BY split, domain
        """
    ).fetchdf()

    route_rows = con.execute(
        """
        WITH train_domains AS (
            SELECT user_id, domain AS train_domain
            FROM split_reviews
            WHERE split = 'train'
            GROUP BY user_id, domain
        ), eval_rows AS (
            SELECT r.split, r.domain, r.user_id, r.parent_asin,
                   CASE WHEN td.train_domain IS NULL THEN 1 ELSE 0 END AS is_cross_domain_candidate
            FROM split_reviews r
            LEFT JOIN train_domains td
              ON r.user_id = td.user_id AND r.domain = td.train_domain
            WHERE r.split IN ('internal_val', 'shadow_test', 'final_test')
        )
        SELECT split, domain,
               COUNT(*) AS n_eval_reviews,
               SUM(is_cross_domain_candidate) AS n_cross_domain_candidate_reviews
        FROM eval_rows
        GROUP BY split, domain
        ORDER BY split, domain
        """
    ).fetchdf()

    summary = {
        "split_summary": summary_rows.to_dict(orient="records"),
        "route_summary": route_rows.to_dict(orient="records"),
        "config": config,
    }
    dump_json(summary, output_dir / "split_summary.json")
    con.close()
    return summary


def stream_metadata_for_selected_items(config: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    splits_dir = output_dir / "splits"
    train_path = splits_dir / "train.parquet"
    if not train_path.exists():
        return None

    all_splits = [str(p) for p in splits_dir.glob("*.parquet")]
    con = duckdb.connect()
    selected = con.execute(
        "SELECT DISTINCT domain, parent_asin FROM read_parquet(?)",
        [all_splits],
    ).fetchdf()
    con.close()

    selected_pairs = set(zip(selected["domain"], selected["parent_asin"]))
    if not selected_pairs:
        return None

    rows: List[Dict[str, Any]] = []
    for domain, rel_path in config.get("raw_meta", {}).items():
        path = Path(rel_path)
        if not path.exists():
            print(f"[warn] Missing metadata file for {domain}: {path}; skipping.")
            continue
        for raw in tqdm(iter_jsonl(path), desc=f"meta:{domain}"):
            parent_asin = raw.get("parent_asin")
            if parent_asin and (domain, str(parent_asin)) in selected_pairs:
                norm = normalize_meta(raw, domain)
                if norm:
                    rows.append(norm)

    if not rows:
        print("[warn] No metadata rows matched selected items.")
        return None

    out = output_dir / "items.parquet"
    pd.DataFrame(rows, columns=META_COLUMNS).drop_duplicates(["domain", "parent_asin"]).to_parquet(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chronological train/internal/shadow/final splits for Amazon Books + Movies baseline harness.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--skip-meta", action="store_true", help="Skip metadata extraction for a faster first run.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = ensure_dir(config["output_dir"])

    parquet_dir = stream_reviews_to_parquet(config, output_dir)
    summary = build_filtered_splits(config, output_dir, parquet_dir)
    if not args.skip_meta:
        stream_metadata_for_selected_items(config, output_dir)

    print("\nPrepared dataset summary:")
    for row in summary["split_summary"]:
        print(row)
    print(f"\nWrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
