from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Optional
import requests

from constructor_agent.domain import EndpointSpec
DEFAULT_API_URL = "https://training.constructor.app/api/platform-kmapi/v1"

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

    def resolved_api_url(self) -> str:
        return (self.api_url or os.getenv("CONSTRUCTOR_API_URL") or DEFAULT_API_URL).rstrip("/")

    def resolved_api_key(self) -> str:
        api_key = self.api_key or os.getenv("CONSTRUCTOR_API_KEY")
        if not api_key:
            raise RuntimeError("Missing CONSTRUCTOR_API_KEY.")
        return api_key


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

    def list_language_models(self) -> list[ConstructorLanguageModelInfo]:
        response = requests.get(
            f"{self.config.resolved_api_url()}/language_models",
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        models = response.json().get("results", [])

        result: list[ConstructorLanguageModelInfo] = []
        for model in models:
            model_id = str(model.get("id") or "")
            alias = str(model.get("alias") or "")
            name = str(model.get("name") or alias or model_id)

            if not alias:
                continue

            result.append(
                ConstructorLanguageModelInfo(
                    id=model_id,
                    alias=alias,
                    name=name,
                    raw=model,
                )
            )

        return result

    def list_endpoint_candidates(
            self,
            include_direct: bool = True,
            include_model: bool = True,
    ) -> list[ConstructorEndpointCandidate]:
        modes: list[str] = []

        if include_direct:
            modes.append("direct")

        if include_model:
            modes.append("model")

        candidates: list[ConstructorEndpointCandidate] = []

        for model in self.list_language_models():
            safe_alias = self._safe_identifier(model.alias)

            for mode in modes:
                endpoint_id = f"{mode}_{safe_alias}"
                snippet = self._endpoint_xml_snippet(endpoint_id, model.alias, mode)

                candidates.append(
                    ConstructorEndpointCandidate(
                        endpoint_id=endpoint_id,
                        llm_alias=model.alias,
                        llm_name=model.name,
                        llm_id=model.id,
                        mode=mode,
                        xml_snippet=snippet,
                    )
                )

        return candidates

    def _headers(self) -> dict[str, str]:
        return {"X-KM-AccessKey": f"Bearer {self.config.resolved_api_key()}"}

    @staticmethod
    def _safe_identifier(value: str) -> str:
        chars = []

        for char in value.lower():
            if char.isalnum():
                chars.append(char)
            else:
                chars.append("_")

        compact = "_".join(part for part in "".join(chars).split("_") if part)
        return compact or "unknown"

    @staticmethod
    def _endpoint_xml_snippet(endpoint_id: str, llm_alias: str, mode: str) -> str:
        return (
            f'<endpoint id="{endpoint_id}" llm_alias="{llm_alias}" mode="{mode}" '
            f'role="review_and_improve" timeout="300" request_timeout="15" retry_delay="3">\n'
            f'    <description>Endpoint using {llm_alias} in {mode} mode.</description>\n'
            f'</endpoint>'
        )

@dataclass(frozen=True)
class ConstructorLanguageModelInfo:
    id: str
    alias: str
    name: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ConstructorEndpointCandidate:
    endpoint_id: str
    llm_alias: str
    llm_name: str
    llm_id: str
    mode: str
    xml_snippet: str