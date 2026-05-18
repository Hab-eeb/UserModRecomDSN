from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

import requests


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    gemini_model: str = "gemini-2.5-flash-lite"
    deepseek_model: str = "deepseek-v4-flash"
    temperature: float = 0.1
    max_output_tokens: int = 2200
    retries: int = 2
    timeout: int = 90


class JSONValidationError(ValueError):
    pass


def clean_text(value: Any, max_chars: Optional[int] = None) -> str:
    if value is None:
        text = ""
    elif isinstance(value, list):
        text = " ".join(clean_text(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)

    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON even if the model wraps it in markdown or extra text."""
    if text is None:
        raise JSONValidationError("LLM returned empty response.")

    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def make_json_repair_prompt(bad_raw_text: str, error: Exception, schema_name: str) -> str:
    if schema_name == "task_a":
        schema = """
{
  "predicted_rating": 4.0,
  "rating_confidence": "medium",
  "rating_rationale": "string",
  "generated_review_title": "string",
  "generated_review_text": "string",
  "style_match_notes": ["string"],
  "possible_failure_modes": ["string"]
}
""".strip()
    elif schema_name == "profile":
        schema = """
{
  "user_summary": "string",
  "rating_behavior": {
    "average_rating_interpretation": "string",
    "rating_scale_harshness": "lenient",
    "what_gets_5_stars": "string",
    "what_gets_low_ratings": "string",
    "calibration_rule": "string"
  },
  "domain_preferences": {
    "Books": "string",
    "Movies_and_TV": "string",
    "cross_domain": "string",
    "Genres": "string"
  },
  "review_style": {
    "typical_length": "medium",
    "tone": "string",
    "specificity": "medium",
    "common_focus": ["string"],
    "title_style": "string",
    "common_words": "string"
  },
  "useful_prediction_cues": ["string"],
  "uncertainties": ["string"]
}
""".strip()
    else:
        schema = "{ }"

    return f"""
Your previous response was not valid JSON or did not match the required schema.

Error:
{repr(error)}

Repair the response below.

Rules:
- Return exactly one valid JSON object.
- No markdown fences.
- No explanation outside JSON.
- No comments.
- No trailing commas.
- No NaN, Infinity, null, or undefined.
- Escape quotes and newlines inside strings correctly.
- Use this schema shape:

{schema}

Broken response:
{bad_raw_text}
""".strip()


class LLMClient:
    """Provider-switchable Gemini / DeepSeek JSON client."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "LLMClient":
        return cls(
            LLMConfig(
                provider=cfg.get("provider", "deepseek"),
                gemini_model=cfg.get("gemini_model", "gemini-2.5-flash-lite"),
                deepseek_model=cfg.get("deepseek_model", "deepseek-v4-flash"),
                temperature=float(cfg.get("temperature", 0.1)),
                max_output_tokens=int(cfg.get("max_output_tokens", 2200)),
                retries=int(cfg.get("retries", 2)),
                timeout=int(cfg.get("timeout", 90)),
            )
        )

    def _raw_call(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        json_mode: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        provider = (provider or self.config.provider).lower().strip()
        temperature = self.config.temperature if temperature is None else temperature
        max_output_tokens = max_output_tokens or self.config.max_output_tokens
        timeout = timeout or self.config.timeout

        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

            model = model or self.config.gemini_model
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                },
            }
            if json_mode:
                payload["generationConfig"]["responseMimeType"] = "application/json"

            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            try:
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as exc:
                raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc

        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

            model = model or self.config.deepseek_model
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise recommendation and user-modelling assistant. "
                            "Return exactly one valid JSON object. Do not use markdown. "
                            "Do not include prose outside JSON. Do not use comments, "
                            "trailing commas, NaN, Infinity, or undefined."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": False,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            try:
                raw_text = data["choices"][0]["message"]["content"]
            except Exception as exc:
                raise RuntimeError(f"Unexpected DeepSeek response shape: {data}") from exc
        else:
            raise ValueError("provider must be 'gemini' or 'deepseek'")

        return {"provider": provider, "model": model, "raw_text": raw_text}

    def generate_json(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        schema_name: Optional[str] = None,
        validator=None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call provider, parse JSON, optionally validate, and repair broken JSON."""
        retries = self.config.retries if retries is None else retries
        current_prompt = prompt
        last_error = None
        last_raw_text = None
        repaired = False
        current_temperature = self.config.temperature if temperature is None else temperature

        for attempt in range(retries + 1):
            result = self._raw_call(
                current_prompt,
                provider=provider,
                model=model,
                temperature=current_temperature,
                max_output_tokens=max_output_tokens,
                json_mode=True,
            )
            raw_text = result["raw_text"]
            last_raw_text = raw_text

            try:
                parsed = safe_json_loads(raw_text)
                if validator:
                    parsed = validator(parsed)
                return {
                    "provider": result["provider"],
                    "model": result["model"],
                    "raw_text": raw_text,
                    "json": parsed,
                    "attempts": attempt + 1,
                    "repaired": repaired,
                }
            except Exception as exc:
                last_error = exc
                current_prompt = make_json_repair_prompt(
                    bad_raw_text=raw_text,
                    error=exc,
                    schema_name=schema_name or "generic",
                )
                repaired = True
                current_temperature = 0.0

        raise ValueError(
            "LLM JSON/schema parsing failed after "
            f"{retries + 1} attempt(s). "
            f"Last error={repr(last_error)}. "
            f"Last raw response preview={repr((last_raw_text or '')[:2500])}"
        )
