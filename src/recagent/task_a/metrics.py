from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd


def rating_metrics(
    df: pd.DataFrame,
    true_col: str = "true_rating",
    pred_col: str = "pred_rating",
) -> Dict[str, Any]:
    valid = df[[true_col, pred_col]].dropna().copy()
    if valid.empty:
        return {"n_rating_examples": 0, "rmse": None, "mae": None}

    err = valid[pred_col].astype(float) - valid[true_col].astype(float)
    return {
        "n_rating_examples": int(len(valid)),
        "rmse": float(math.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
    }


def combine_review_text(title: Any, text: Any) -> str:
    title = "" if title is None else str(title).strip()
    text = "" if text is None else str(text).strip()
    return (title + "\n" + text).strip()


def rouge_l_metrics(
    references: List[str],
    predictions: List[str],
    *,
    use_stemmer: bool = True,
) -> Dict[str, Any]:
    """
    ROUGE-L text similarity.

    This is local and lightweight. It does not require torch, transformers,
    Hugging Face downloads, or an API call.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError(
            "rouge-score is not installed. Run: pip install rouge-score==0.1.2"
        ) from exc

    pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not pairs:
        return {
            "n_rouge_examples": 0,
            "rouge_l_precision": None,
            "rouge_l_recall": None,
            "rouge_l_f1": None,
        }

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    precisions = []
    recalls = []
    f1s = []

    for pred, ref in pairs:
        score = scorer.score(ref, pred)["rougeL"]
        precisions.append(score.precision)
        recalls.append(score.recall)
        f1s.append(score.fmeasure)

    return {
        "n_rouge_examples": int(len(pairs)),
        "rouge_l_precision": float(sum(precisions) / len(precisions)),
        "rouge_l_recall": float(sum(recalls) / len(recalls)),
        "rouge_l_f1": float(sum(f1s) / len(f1s)),
    }


def bertscore_metrics(
    references: List[str],
    predictions: List[str],
    *,
    lang: str = "en",
    model_type: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Optional BERTScore metrics.

    Requires torch + bert-score. If the model is not cached locally,
    BERTScore downloads model files once, then computes locally.
    """
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError("BERTScore is not installed. Run: pip install bert-score torch") from exc

    pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not pairs:
        return {
            "n_bertscore_examples": 0,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
        }

    preds, refs = zip(*pairs)
    kwargs = {"lang": lang, "batch_size": batch_size, "verbose": False}
    if model_type:
        kwargs["model_type"] = model_type

    precision, recall, f1 = bert_score(list(preds), list(refs), **kwargs)

    return {
        "n_bertscore_examples": int(len(pairs)),
        "bertscore_precision": float(precision.mean().item()),
        "bertscore_recall": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }


def evaluate_prediction_dataframe(
    df: pd.DataFrame,
    *,
    compute_rouge: bool = False,
    compute_bertscore: bool = False,
    bertscore_lang: str = "en",
) -> Dict[str, Any]:
    metrics = rating_metrics(df)
    metrics["n_rows"] = int(len(df))
    metrics["n_success"] = int(df["pred_rating"].notna().sum()) if "pred_rating" in df else 0
    metrics["n_failed"] = int(df["error"].notna().sum()) if "error" in df else 0

    refs = [
        combine_review_text(t, x)
        for t, x in zip(df.get("true_review_title", []), df.get("true_review_text", []))
    ]
    preds = [
        combine_review_text(t, x)
        for t, x in zip(df.get("pred_review_title", []), df.get("pred_review_text", []))
    ]

    if compute_rouge:
        try:
            metrics.update(rouge_l_metrics(refs, preds))
        except Exception as exc:
            metrics["rouge_l_error"] = repr(exc)

    if compute_bertscore:
        try:
            metrics.update(bertscore_metrics(refs, preds, lang=bertscore_lang))
        except Exception as exc:
            metrics["bertscore_error"] = repr(exc)

    return metrics