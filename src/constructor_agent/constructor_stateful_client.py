from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from constructor_agent.domain import EndpointSpec


@dataclass(frozen=True)
class ConstructorPlatformConfig:
    """Connection parameters for ConstructorAdapter."""

    api_url: Optional[str] = None
    api_key: Optional[str] = None
    km_id: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "ConstructorPlatformConfig":
        return cls(
            api_url=os.getenv("CONSTRUCTOR_API_URL"),
            api_key=os.getenv("CONSTRUCTOR_API_KEY"),
            km_id=os.getenv("CONSTRUCTOR_KM_ID"),
        )


class StatefulConstructorClient:
    """
    Concrete adapter around constructor_adapter.StatefulConstructorAdapter.

    One StatefulConstructorAdapter instance is cached per (llm_alias, mode), so each
    endpoint keeps its own Constructor chat session during one run.
    """

    def __init__(self, config: ConstructorPlatformConfig | None = None) -> None:
        self.config = config or ConstructorPlatformConfig.from_environment()
        self._adapters: dict[tuple[str, str], object] = {}

    def ask(self, endpoint: EndpointSpec, prompt: str) -> str:
        adapter = self._get_or_create_adapter(endpoint)
        query = getattr(adapter, "query")
        return str(
            query(
                prompt,
                timeout=endpoint.timeout,
                request_timeout=endpoint.request_timeout,
                retry_delay=endpoint.retry_delay,
            )
        )

    def restart_all_sessions(self) -> None:
        for adapter in self._adapters.values():
            restart = getattr(adapter, "restart_session", None)
            if callable(restart):
                restart()

    def _get_or_create_adapter(self, endpoint: EndpointSpec) -> object:
        key = (endpoint.llm_alias, endpoint.mode)
        if key not in self._adapters:
            self._adapters[key] = self._new_adapter(endpoint)
        return self._adapters[key]

    def _new_adapter(self, endpoint: EndpointSpec) -> object:
        try:
            from constructor_adapter import StatefulConstructorAdapter
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import constructor_adapter.StatefulConstructorAdapter. "
                "Install ConstructorAdapter first, for example with: "
                "pip install git+https://github.com/GiancarloSucci/ConstructorAdapter.git"
            ) from exc

        kwargs = {
            "mode": endpoint.mode,
            "llm_alias": endpoint.llm_alias,
        }
        if self.config.api_url:
            kwargs["api_url"] = self.config.api_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.km_id:
            kwargs["km_id"] = self.config.km_id

        return StatefulConstructorAdapter(**kwargs)
