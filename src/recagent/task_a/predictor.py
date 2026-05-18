from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from recagent.llm.client import LLMClient
from recagent.task_a.data_store import TaskADataStore
from recagent.task_a.evidence import build_user_evidence_packet, summarize_item
from recagent.task_a.profile import UserProfileGenerator
from recagent.task_a.prompts import PROFILE_PROMPT_VERSION, TASK_A_PROMPT_VERSION, make_task_a_prompt
from recagent.task_a.validators import validate_task_a_prediction


class TaskAPredictor:
    """Clean service layer that FastAPI can wrap directly."""

    def __init__(
        self,
        store: TaskADataStore,
        llm: LLMClient,
        profile_generator: UserProfileGenerator,
        evidence_config: Optional[Dict[str, Any]] = None,
        locale_hint: Optional[str] = None,
    ):
        self.store = store
        self.llm = llm
        self.profile_generator = profile_generator
        self.evidence_config = evidence_config or {}
        self.locale_hint = locale_hint

    def list_users(self, limit: int = 100, offset: int = 0):
        return self.store.list_users(limit=limit, offset=offset)

    def list_eval_items_for_user(self, user_id: str, split: str = "internal_val", limit: int = 100):
        return self.store.list_eval_items_for_user(user_id=user_id, split=split, limit=limit)

    def predict(
        self,
        user_id: str,
        target_domain: str,
        target_parent_asin: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        locale_hint: Optional[str] = None,
        force_profile_refresh: bool = False,
        include_ground_truth: bool = False,
        ground_truth_split: str = "internal_val",
    ) -> Dict[str, Any]:
        evidence_packet = build_user_evidence_packet(
            self.store,
            user_id=user_id,
            target_domain=target_domain,
            target_parent_asin=target_parent_asin,
            **self.evidence_config,
        )
        profile_payload = self.profile_generator.generate(
            user_id=user_id,
            target_domain=target_domain,
            target_parent_asin=target_parent_asin,
            provider=provider,
            model=model,
            force_refresh=force_profile_refresh,
        )
        target_item = summarize_item(self.store, target_domain, target_parent_asin, max_chars=900)
        prompt = make_task_a_prompt(
            user_profile=profile_payload["profile"],
            evidence_packet=evidence_packet,
            target_item=target_item,
            locale_hint=locale_hint if locale_hint is not None else self.locale_hint,
        )
        result = self.llm.generate_json(
            prompt,
            provider=provider,
            model=model,
            max_output_tokens=1600,
            schema_name="task_a",
            validator=validate_task_a_prediction,
        )

        out = {
            "user_id": str(user_id),
            "target_domain": str(target_domain),
            "target_parent_asin": str(target_parent_asin),
            "provider": result["provider"],
            "model": result["model"],
            "attempts": result.get("attempts"),
            "repaired": result.get("repaired"),
            "profile_prompt_version": PROFILE_PROMPT_VERSION,
            "task_a_prompt_version": TASK_A_PROMPT_VERSION,
            "target_item": target_item,
            "profile": profile_payload["profile"],
            "prediction": result["json"],
        }

        if include_ground_truth:
            truth = self.store.get_ground_truth(
                user_id=user_id,
                target_domain=target_domain,
                target_parent_asin=target_parent_asin,
                split=ground_truth_split,
            )
            out["ground_truth"] = truth
            if truth is not None:
                err = float(result["json"]["predicted_rating"]) - float(truth["rating"])
                out["single_item_error"] = {
                    "absolute_error": abs(err),
                    "squared_error": err * err,
                }
        return out


def build_task_a_predictor(config_path: str = "configs/task_a.yaml") -> TaskAPredictor:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    store = TaskADataStore.from_dict(cfg["data"])
    llm = LLMClient.from_dict(cfg.get("llm", {}))
    evidence_cfg = cfg.get("profile", {})
    profile_generator = UserProfileGenerator(
        store=store,
        llm=llm,
        cache_dir=cfg["data"]["profile_cache_dir"],
        evidence_config=evidence_cfg,
    )
    return TaskAPredictor(
        store=store,
        llm=llm,
        profile_generator=profile_generator,
        evidence_config=evidence_cfg,
        locale_hint=cfg.get("prediction", {}).get("locale_hint"),
    )
