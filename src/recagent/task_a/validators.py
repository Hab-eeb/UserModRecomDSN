from __future__ import annotations

from typing import Any, Dict, List, Optional

from recagent.llm.client import clean_text

TASK_A_REQUIRED_KEYS = {
    "predicted_rating",
    "rating_confidence",
    "rating_rationale",
    "generated_review_title",
    "generated_review_text",
    "style_match_notes",
    "possible_failure_modes",
}

PROFILE_REQUIRED_KEYS = {
    "user_summary",
    "rating_behavior",
    "domain_preferences",
    "review_style",
    "useful_prediction_cues",
    "uncertainties",
}


def _as_clean_string(value: Any, max_chars: Optional[int] = None) -> str:
    return clean_text(value, max_chars=max_chars)


def _as_string_list(value: Any, fallback: Optional[List[str]] = None) -> List[str]:
    fallback = [] if fallback is None else fallback
    if value is None:
        return fallback
    if isinstance(value, list):
        out = [_as_clean_string(v, max_chars=300) for v in value]
        return [v for v in out if v]
    if isinstance(value, str):
        value = clean_text(value)
        return [value] if value else fallback
    return [_as_clean_string(value, max_chars=300)]


def validate_task_a_prediction(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"Task A prediction must be a JSON object. Got: {type(obj)}")

    missing = TASK_A_REQUIRED_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"Task A prediction missing keys: {sorted(missing)}")

    try:
        rating = float(obj["predicted_rating"])
        rating = max(1.0, min(5.0, rating))
    except Exception as exc:
        raise ValueError(f"Invalid predicted_rating={obj.get('predicted_rating')!r}: {repr(exc)}")

    confidence = clean_text(obj.get("rating_confidence", "")).lower()
    if confidence not in {"low", "medium", "high"}:
        raise ValueError(f"Invalid rating_confidence={obj.get('rating_confidence')!r}")

    cleaned = {
        "predicted_rating": rating,
        "rating_confidence": confidence,
        "rating_rationale": _as_clean_string(obj.get("rating_rationale", ""), max_chars=600),
        "generated_review_title": _as_clean_string(obj.get("generated_review_title", ""), max_chars=180),
        "generated_review_text": _as_clean_string(obj.get("generated_review_text", ""), max_chars=1800),
        "style_match_notes": _as_string_list(obj.get("style_match_notes"), fallback=[]),
        "possible_failure_modes": _as_string_list(obj.get("possible_failure_modes"), fallback=[]),
    }

    if not cleaned["generated_review_text"]:
        raise ValueError("generated_review_text is empty.")
    return cleaned


def validate_user_profile(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"User profile must be a JSON object. Got: {type(obj)}")

    missing = PROFILE_REQUIRED_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"User profile missing keys: {sorted(missing)}")

    obj["user_summary"] = _as_clean_string(obj.get("user_summary", ""), max_chars=1000)
    obj["useful_prediction_cues"] = _as_string_list(obj.get("useful_prediction_cues"), fallback=[])
    obj["uncertainties"] = _as_string_list(obj.get("uncertainties"), fallback=[])

    for key in ["rating_behavior", "domain_preferences", "review_style"]:
        if not isinstance(obj.get(key), dict):
            raise ValueError(f"{key} must be a JSON object.")
    return obj
