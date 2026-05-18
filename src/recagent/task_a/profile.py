from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from recagent.llm.client import LLMClient
from recagent.task_a.data_store import TaskADataStore
from recagent.task_a.evidence import build_user_evidence_packet
from recagent.task_a.prompts import PROFILE_PROMPT_VERSION, make_user_profile_prompt
from recagent.task_a.validators import validate_user_profile


class UserProfileGenerator:
    def __init__(
        self,
        store: TaskADataStore,
        llm: LLMClient,
        cache_dir: str,
        evidence_config: Optional[Dict[str, Any]] = None,
    ):
        self.store = store
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_config = evidence_config or {}

    def profile_cache_path(self, user_id: str, provider: Optional[str] = None, model: Optional[str] = None) -> Path:
        provider = provider or self.llm.config.provider
        model = model or (self.llm.config.gemini_model if provider == "gemini" else self.llm.config.deepseek_model)
        key = f"{PROFILE_PROMPT_VERSION}|{provider}|{model}|{user_id}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def generate(
        self,
        user_id: str,
        target_domain: Optional[str] = None,
        target_parent_asin: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        cache_path = self.profile_cache_path(user_id, provider=provider, model=model)
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        evidence_packet = build_user_evidence_packet(
            self.store,
            user_id=user_id,
            target_domain=target_domain,
            target_parent_asin=target_parent_asin,
            **self.evidence_config,
        )
        prompt = make_user_profile_prompt(evidence_packet)
        result = self.llm.generate_json(
            prompt,
            provider=provider,
            model=model,
            max_output_tokens=1800,
            schema_name="profile",
            validator=validate_user_profile,
        )

        payload = {
            "user_id": str(user_id),
            "provider": result["provider"],
            "model": result["model"],
            "prompt_version": PROFILE_PROMPT_VERSION,
            "created_at": time.time(),
            "attempts": result.get("attempts"),
            "repaired": result.get("repaired"),
            "evidence_packet": evidence_packet,
            "profile": result["json"],
        }
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
