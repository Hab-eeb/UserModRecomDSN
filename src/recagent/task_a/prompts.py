from __future__ import annotations

import json
from typing import Any, Dict, Optional

PROFILE_PROMPT_VERSION = "user_profile_v2_strict_json"
TASK_A_PROMPT_VERSION = "task_a_rating_review_v2_strict_json"


def make_user_profile_prompt(evidence_packet: Dict[str, Any]) -> str:
    return f"""
You are building a compact behavioural profile for a recommender-system user.

Use ONLY the evidence packet provided below.
Do not invent demographics, nationality, age, gender, location, occupation, or life context.

The profile should help another model predict:
1. the user's likely star rating for unseen Books or Movies_and_TV items
2. the user's review tone, length, and writing style
3. transferable preferences across Books and Movies_and_TV

JSON rules:
- Return exactly one valid JSON object and nothing else.
- Do not use markdown fences.
- Do not include comments.
- Do not include trailing commas.
- Do not use NaN, Infinity, null, or undefined.
- All strings must escape internal quotes and newlines correctly.
- rating_scale_harshness must be exactly one of: "lenient", "moderate", "harsh", "unknown".
- typical_length must be exactly one of: "short", "medium", "long", "mixed".
- specificity must be exactly one of: "low", "medium", "high".

Return this exact JSON structure:
{{
  "user_summary": "2-4 sentence summary of the user's preference pattern.",
  "rating_behavior": {{
    "average_rating_interpretation": "How to interpret the user's average rating.",
    "rating_scale_harshness": "moderate",
    "what_gets_5_stars": "Patterns that appear to earn 5 stars from this user.",
    "what_gets_low_ratings": "Patterns that appear to earn low ratings from this user.",
    "calibration_rule": "How to anchor future predictions to this user's mean and rating distribution."
  }},
  "domain_preferences": {{
    "Books": "Book-specific preferences supported by evidence.",
    "Movies_and_TV": "Movie/TV-specific preferences supported by evidence.",
    "cross_domain": "Domain-agnostic preferences that transfer between books and movies.",
    "Genres": "Likely genres or content types, only if supported by evidence."
  }},
  "review_style": {{
    "typical_length": "medium",
    "tone": "The user's likely tone.",
    "specificity": "medium",
    "common_focus": ["Aspects the user often comments on."],
    "title_style": "How this user tends to title reviews.",
    "common_words": "Repeated words or phrasing patterns, if visible."
  }},
  "useful_prediction_cues": ["Concrete cues that should influence future rating/review predictions."],
  "uncertainties": ["Important limitations or missing evidence."]
}}

Evidence packet:
{json.dumps(evidence_packet, ensure_ascii=False)}
""".strip()


def make_task_a_prompt(
    user_profile: Dict[str, Any],
    evidence_packet: Dict[str, Any],
    target_item: Dict[str, Any],
    locale_hint: Optional[str] = None,
) -> str:
    locale_instruction = ""
    if locale_hint:
        locale_instruction = f"""
Locale/context instruction:
The user context includes: {locale_hint}
Reflect this only if it is natural for the review style.
Avoid caricature, forced slang, exaggerated pidgin, or stereotypes.
""".strip()

    return f"""
You are simulating how a specific Amazon reviewer would rate and review an unseen item.

Use ONLY:
- the user's behavioural profile
- the user's selected past reviews
- the user's rating statistics
- the target item metadata

Core task:
1. Predict the star rating first.
2. Anchor the rating to the user's rating distribution and calibration rule.
3. Then generate a review title and review text that match the predicted rating.
4. Match the user's usual review length, tone, specificity, and title style.

Important behavioural rules:
- Do not predict only based on item popularity.
- Do not make every prediction positive.
- If the user's history is mostly generous, reflect that.
- If the user's history is harsh or selective, reflect that.
- If target metadata is weak, rely more on user rating behaviour and state uncertainty in possible_failure_modes.
- Do not mention facts not supported by the target item metadata.
- Do not copy any previous review text.
- Do not mention that you are an AI, model, simulator, or recommender system.

JSON rules:
- Return exactly one valid JSON object and nothing else.
- Do not use markdown fences.
- Do not include comments.
- Do not include trailing commas.
- Do not use NaN, Infinity, null, or undefined.
- All string values must escape internal quotes and newlines correctly.
- predicted_rating must be a number between 1.0 and 5.0, not a string.
- rating_confidence must be exactly one of: "low", "medium", "high".
- style_match_notes must be an array of strings.
- possible_failure_modes must be an array of strings.

Return this exact JSON structure:
{{
  "predicted_rating": 4.0,
  "rating_confidence": "medium",
  "rating_rationale": "Brief reason anchored to the user's profile, rating stats, selected evidence, and target item fit.",
  "generated_review_title": "Short title in the user's likely style.",
  "generated_review_text": "Review text in the user's likely style.",
  "style_match_notes": ["Short note about tone, length, specificity, or title style."],
  "possible_failure_modes": ["Short note about uncertainty, sparse metadata, or weak evidence."]
}}

{locale_instruction}

User profile:
{json.dumps(user_profile, ensure_ascii=False)}

Selected past-review evidence:
{json.dumps(evidence_packet["selected_review_examples"], ensure_ascii=False)}

User rating stats:
{json.dumps(evidence_packet["rating_stats"], ensure_ascii=False)}

Target item:
{json.dumps(target_item, ensure_ascii=False)}
""".strip()
