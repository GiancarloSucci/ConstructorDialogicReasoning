from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


DEFAULT_API_URL = "https://training.constructor.app/api/platform-kmapi/v1"


@dataclass(frozen=True)
class ConstructorPlatformConfig:
    """Connection parameters for ConstructorPlatform."""

    api_url: Optional[str] = None
    api_key: Optional[str] = None
    km_id: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "ConstructorPlatformConfig":
        return cls(
            api_url=os.getenv("CONSTRUCTOR_API_URL"),
            api_key=os.getenv("CONSTRUCTOR_API_KEY"),
            km_id=os.getenv("CONSTRUCTOR_KM_ID") or os.getenv("KNOWLEDGE_MODEL_ID"),
        )

    def resolved_api_url(self) -> str:
        return (
            self.api_url
            or os.getenv("CONSTRUCTOR_API_URL")
            or DEFAULT_API_URL
        ).rstrip("/")

    def resolved_api_key(self) -> str:
        api_key = self.api_key or os.getenv("CONSTRUCTOR_API_KEY")

        if not api_key:
            raise RuntimeError("Missing CONSTRUCTOR_API_KEY.")

        return api_key

    def resolved_km_id(self, optional: bool = False) -> str | None:
        km_id = (
            self.km_id
            or os.getenv("CONSTRUCTOR_KM_ID")
            or os.getenv("KNOWLEDGE_MODEL_ID")
        )

        if km_id is not None:
            km_id = km_id.strip()

        if km_id:
            return km_id

        if optional:
            return None

        raise RuntimeError(
            "Missing CONSTRUCTOR_KM_ID or KNOWLEDGE_MODEL_ID. "
            "Use mode='direct' or configure a knowledge model id."
        )