from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

from recagent.task_a.metrics import evaluate_prediction_dataframe
from recagent.task_a.predictor import build_task_a_predictor



def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate Task A rating/review predictions.")
    parser.add_argument("--config", default="configs/task_a.yaml")
    parser.add_argument("--split", default=None, choices=[None, "internal_val", "shadow_test"])
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--provider", default=None, choices=[None, "gemini", "deepseek"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--compute-rouge", action="store_true")
    parser.add_argument("--compute-bertscore", action="store_true")
    parser.add_argument("--sleep-s", type=float, default=None)
    parser.add_argument("--force-profile-refresh", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = cfg.get("evaluation", {})
    pred_cfg = cfg.get("prediction", {})

    split = args.split or eval_cfg.get("split", "internal_val")
    sample_n = args.sample_n if args.sample_n is not None else int(eval_cfg.get("sample_n", 20))
    random_state = args.random_state if args.random_state is not None else int(eval_cfg.get("random_state", 42))
    sleep_s = args.sleep_s if args.sleep_s is not None else float(eval_cfg.get("sleep_s", 0))
    compute_rouge = bool(args.compute_rouge or eval_cfg.get("compute_rouge", False))
    compute_bertscore = bool(args.compute_bertscore or eval_cfg.get("compute_bertscore", False))

    predictor = build_task_a_predictor(args.config)
    eval_df = predictor.store.get_eval_df(split).copy()
    if sample_n and len(eval_df) > sample_n:
        eval_df = eval_df.sample(n=sample_n, random_state=random_state).copy()

    records = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Task A eval:{split}"):
        base = {
            "user_id": str(row["user_id"]),
            "domain": str(row["domain"]),
            "parent_asin": str(row["parent_asin"]),
            "true_rating": float(row["rating"]),
            "true_review_title": str(row.get("review_title", "") or ""),
            "true_review_text": str(row.get("review_text", "") or ""),
            "pred_rating": None,
            "pred_review_title": None,
            "pred_review_text": None,
            "pred_rationale": None,
            "rating_confidence": None,
            "provider": args.provider or cfg.get("llm", {}).get("provider"),
            "model": args.model,
            "attempts": None,
            "repaired": None,
            "error": None,
        }
        try:
            pred = predictor.predict(
                user_id=row["user_id"],
                target_domain=row["domain"],
                target_parent_asin=row["parent_asin"],
                provider=args.provider,
                model=args.model,
                locale_hint=pred_cfg.get("locale_hint"),
                force_profile_refresh=args.force_profile_refresh or bool(pred_cfg.get("force_profile_refresh", False)),
                include_ground_truth=False,
            )
            p = pred["prediction"]
            base.update(
                {
                    "pred_rating": p.get("predicted_rating"),
                    "pred_review_title": p.get("generated_review_title"),
                    "pred_review_text": p.get("generated_review_text"),
                    "pred_rationale": p.get("rating_rationale"),
                    "rating_confidence": p.get("rating_confidence"),
                    "provider": pred.get("provider"),
                    "model": pred.get("model"),
                    "attempts": pred.get("attempts"),
                    "repaired": pred.get("repaired"),
                }
            )
        except Exception as exc:
            base["error"] = repr(exc)
        records.append(base)
        if sleep_s:
            time.sleep(sleep_s)

    results = pd.DataFrame(records)
    metrics = evaluate_prediction_dataframe(
        results,
        compute_rouge=compute_rouge,
        compute_bertscore=compute_bertscore,
        bertscore_lang=eval_cfg.get("bertscore_lang", "en"),
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(eval_cfg.get("output_dir", "data/processed/task_a_iteration_v0/eval_runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions to: {out_dir / 'predictions.csv'}")
    print(f"Wrote metrics to: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
