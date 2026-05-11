from __future__ import annotations

import math
from typing import Iterable, Sequence, Set

import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def dcg_at_k(relevance: Sequence[int | float], k: int) -> float:
    rel = list(relevance)[:k]
    return float(sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel)))


def ndcg_at_k(relevance: Sequence[int | float], k: int) -> float:
    if not relevance:
        return 0.0
    dcg = dcg_at_k(relevance, k)
    ideal = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else float(dcg / idcg)


def hit_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> int:
    return int(any(item in relevant for item in recommended[:k]))
