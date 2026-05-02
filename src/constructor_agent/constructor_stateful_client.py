from __future__ import annotations

from dataclasses import dataclass, replace
import os
import time
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


class StatefulConstructorClient:
    """
    Concrete adapter around constructor_adapter.StatefulConstructorAdapter.

    One StatefulConstructorAdapter instance is cached per effective
    (llm_alias, mode), so each endpoint keeps its own Constructor chat session
    during one run.

    Important implementation detail:
    ConstructorAdapter currently requires a knowledge model id even for
    direct mode, because StatefulConstructorAdapter creates chat sessions under
    /knowledge-models/{km_id}/chat-sessions. Therefore, this wrapper creates
    a temporary empty knowledge model when the configured km_id is missing,
    invalid, or inaccessible.
    """

    def __init__(self, config: ConstructorPlatformConfig | None = None) -> None:
        self.config = config or ConstructorPlatformConfig()
        self._adapters: dict[tuple[str, str], object] = {}
        self._force_direct_mode = False
        self._direct_warning_printed = False
        self._temporary_direct_km_id: str | None = None
        self._patch_constructor_adapter_for_direct_mode()

    def ask(self, endpoint: EndpointSpec, prompt: str) -> str:
        adapter = self._get_or_create_adapter(endpoint)

        return self._query_adapter_robust(
            adapter=adapter,
            prompt=prompt,
            timeout=endpoint.timeout,
            request_timeout=endpoint.request_timeout,
            retry_delay=endpoint.retry_delay,
        )

    def restart_all_sessions(self) -> None:
        for adapter in self._adapters.values():
            restart = getattr(adapter, "restart_session", None)
            if callable(restart):
                restart()

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

    def _get_or_create_adapter(self, endpoint: EndpointSpec) -> object:
        effective_endpoint = self._effective_endpoint(endpoint)
        key = (effective_endpoint.llm_alias, effective_endpoint.mode)

        if key not in self._adapters:
            self._adapters[key] = self._new_adapter(effective_endpoint)

        return self._adapters[key]

    def _new_adapter(self, endpoint: EndpointSpec) -> object:
        from constructor_adapter import StatefulConstructorAdapter

        effective_endpoint = self._effective_endpoint(endpoint)

        if effective_endpoint.mode == "direct":
            return self._new_direct_adapter(effective_endpoint)

        kwargs = {
            "mode": "model",
            "api_url": self.config.resolved_api_url(),
            "api_key": self.config.resolved_api_key(),
            "km_id": self.config.resolved_km_id(optional=False),
            "llm_alias": effective_endpoint.llm_alias,
        }

        try:
            return StatefulConstructorAdapter(**kwargs)

        except Exception as exc:
            self._enable_direct_mode(
                "Knowledge model mode failed during adapter initialization: "
                f"{type(exc).__name__}: {exc}. "
                "Continuing with direct LLM mode for all endpoints."
            )

            direct_endpoint = self._endpoint_as_direct(effective_endpoint)
            return self._new_direct_adapter(direct_endpoint)

    def _new_direct_adapter(self, endpoint: EndpointSpec) -> object:
        from constructor_adapter import StatefulConstructorAdapter

        km_id = self._get_direct_mode_km_id()

        return StatefulConstructorAdapter(
            mode="direct",
            api_url=self.config.resolved_api_url(),
            api_key=self.config.resolved_api_key(),
            km_id=km_id,
            llm_alias=endpoint.llm_alias,
        )

    def _query_adapter_robust(
        self,
        adapter: object,
        prompt: str,
        timeout: int = 300,
        request_timeout: int = 15,
        retry_delay: int = 3,
    ) -> str:
        before_ids = self._current_message_ids(
            adapter,
            request_timeout=request_timeout,
        )

        send_message = getattr(adapter, "_send_message")
        send_message(prompt)

        start_time = time.time()

        while True:
            messages = self._get_session_messages(
                adapter,
                request_timeout=request_timeout,
            )

            answer = self._extract_new_done_ai_answer(
                messages=messages,
                before_ids=before_ids,
            )

            if answer is not None:
                return answer

            if time.time() - start_time > timeout:
                debug = self._format_message_debug(messages)
                raise TimeoutError(
                    "Model response timed out while waiting for a new done "
                    "ai_message.\n"
                    f"Last messages received from ConstructorPlatform:\n{debug}"
                )

            time.sleep(retry_delay)

    def _current_message_ids(
        self,
        adapter: object,
        request_timeout: int = 15,
    ) -> set[str]:
        try:
            messages = self._get_session_messages(
                adapter,
                request_timeout=request_timeout,
            )
        except Exception:
            return set()

        result: set[str] = set()

        for message in messages:
            message_id = message.get("id")
            if message_id is not None:
                result.add(str(message_id))

        return result

    def _get_session_messages(
        self,
        adapter: object,
        request_timeout: int = 15,
    ) -> list[dict[str, Any]]:
        api_url = getattr(adapter, "api_url")
        km_id = getattr(adapter, "km_id")
        session_id = getattr(adapter, "session_id")
        get_headers = getattr(adapter, "_get_headers")

        response = requests.get(
            f"{api_url}/knowledge-models/{km_id}/chat-sessions/"
            f"{session_id}/messages",
            headers=get_headers(),
            timeout=request_timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Could not retrieve chat messages. "
                f"Status: {response.status_code}. Response: {response.text}"
            )

        return response.json().get("results", [])

    def _extract_new_done_ai_answer(
        self,
        messages: list[dict[str, Any]],
        before_ids: set[str],
    ) -> str | None:
        for message in messages:
            message_id = message.get("id")

            if message_id is not None and str(message_id) in before_ids:
                continue

            message_type = str(message.get("type") or "").lower()

            if message_type not in {"ai_message", "assistant", "assistant_message"}:
                continue

            status = message.get("status") or {}
            status_name = str(
                status.get("name") or status.get("status") or ""
            ).lower()

            if status_name and status_name not in {
                "done",
                "completed",
                "complete",
                "success",
                "succeeded",
                "finished",
            }:
                continue

            content = message.get("content") or {}

            text = (
                content.get("text")
                or content.get("content")
                or content.get("answer")
                or message.get("text")
                or message.get("answer")
            )

            if text:
                return str(text)

        return None

    def _format_message_debug(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "No messages returned."

        lines: list[str] = []

        for index, message in enumerate(messages[:10], start=1):
            status = message.get("status") or {}
            content = message.get("content") or {}

            text = content.get("text")
            if text:
                text = str(text).replace("\n", " ")
                text = text[:300]
            else:
                text = ""

            lines.append(
                "Message "
                f"{index}: "
                f"id={message.get('id')}, "
                f"type={message.get('type')}, "
                f"status={status}, "
                f"content_keys={list(content.keys())}, "
                f"text_preview={text!r}"
            )

        return "\n".join(lines)

    def _effective_endpoint(self, endpoint: EndpointSpec) -> EndpointSpec:
        if self._force_direct_mode:
            return self._endpoint_as_direct(endpoint)

        if endpoint.mode == "model":
            configured_km_id = self.config.resolved_km_id(optional=True)

            if not self._is_valid_km_id(configured_km_id):
                self._enable_direct_mode(
                    "Knowledge model id is missing, invalid, or not accessible. "
                    "Continuing with direct LLM mode for all endpoints."
                )
                return self._endpoint_as_direct(endpoint)

        return endpoint

    def _endpoint_as_direct(self, endpoint: EndpointSpec) -> EndpointSpec:
        if getattr(endpoint, "mode", None) == "direct":
            return endpoint

        if hasattr(endpoint, "model_copy"):
            return endpoint.model_copy(update={"mode": "direct"})

        if hasattr(endpoint, "copy"):
            return endpoint.copy(update={"mode": "direct"})

        return replace(endpoint, mode="direct")

    def _get_direct_mode_km_id(self) -> str:
        configured_km_id = self.config.resolved_km_id(optional=True)

        if self._is_valid_km_id(configured_km_id):
            return str(configured_km_id)

        if configured_km_id:
            self._enable_direct_mode(
                "Configured knowledge model is not valid or not accessible: "
                f"{configured_km_id}. Creating a temporary empty knowledge model."
            )
        else:
            self._enable_direct_mode(
                "No knowledge model id configured. "
                "Creating a temporary empty knowledge model."
            )

        temporary_km_id = self._create_temporary_direct_km()

        if not self._is_valid_km_id(temporary_km_id):
            raise RuntimeError(
                "Temporary knowledge model was created but is not accessible: "
                f"{temporary_km_id}"
            )

        return temporary_km_id

    def _is_valid_km_id(self, km_id: str | None) -> bool:
        if not km_id or not km_id.strip():
            return False

        try:
            response = requests.get(
                f"{self.config.resolved_api_url()}/knowledge-models/"
                f"{km_id.strip()}",
                headers=self._headers(),
                timeout=30,
            )
        except requests.RequestException:
            return False

        return response.status_code == 200

    def _create_temporary_direct_km(self) -> str:
        if self._temporary_direct_km_id:
            return self._temporary_direct_km_id

        payload = {
            "name": "constructor-agent-direct-runtime",
            "description": (
                "Temporary empty knowledge model used only as a technical "
                "container for direct LLM sessions."
            ),
            "shared_type": "private",
            "share_documents": False,
        }

        response = requests.post(
            f"{self.config.resolved_api_url()}/knowledge-models",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                "Cannot create temporary knowledge model for direct mode. "
                f"Status: {response.status_code}. Response: {response.text}"
            )

        data = response.json()
        km_id = (
            data.get("id")
            or data.get("uuid")
            or data.get("knowledge_model_id")
        )

        if not km_id:
            raise RuntimeError(
                "ConstructorPlatform created a temporary knowledge model but "
                "did not return an id. "
                f"Response: {data}"
            )

        self._temporary_direct_km_id = str(km_id)

        self._enable_direct_mode(
            "Created temporary empty knowledge model for direct mode: "
            f"{self._temporary_direct_km_id}"
        )

        return self._temporary_direct_km_id

    def _enable_direct_mode(self, reason: str) -> None:
        self._force_direct_mode = True

        if not self._direct_warning_printed:
            print(f"[ConstructorAgent] {reason}")
            self._direct_warning_printed = True

    def _headers(self) -> dict[str, str]:
        return {"X-KM-AccessKey": f"Bearer {self.config.resolved_api_key()}"}

    @staticmethod
    def _safe_identifier(value: str) -> str:
        chars: list[str] = []

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
            f'<endpoint id="{endpoint_id}" llm_alias="{llm_alias}" '
            f'mode="{mode}" role="review_and_improve" timeout="900" '
            f'request_timeout="30" retry_delay="5">\n'
            f"    <description>Endpoint using {llm_alias} in {mode} "
            f"mode.</description>\n"
            f"</endpoint>"
        )

    def _patch_constructor_adapter_for_direct_mode(self) -> None:
        try:
            from constructor_adapter.constructor_adapter_base import ConstructorAdapter
        except Exception:
            return

        if getattr(ConstructorAdapter, "_constructor_agent_direct_patch", False):
            return

        original_check = getattr(ConstructorAdapter, "_check_if_km_exists", None)
        original_get_uploaded_files = getattr(
            ConstructorAdapter,
            "_get_already_uploaded_files",
            None,
        )

        def patched_check_if_km_exists(adapter_self):
            if getattr(adapter_self, "mode", None) == "direct":
                return None

            if original_check is None:
                return None

            return original_check(adapter_self)

        def patched_get_already_uploaded_files(adapter_self):
            if getattr(adapter_self, "mode", None) == "direct":
                adapter_self.uploaded_files = {}
                return None

            if original_get_uploaded_files is None:
                adapter_self.uploaded_files = {}
                return None

            return original_get_uploaded_files(adapter_self)

        ConstructorAdapter._check_if_km_exists = patched_check_if_km_exists
        ConstructorAdapter._get_already_uploaded_files = (
            patched_get_already_uploaded_files
        )
        ConstructorAdapter._constructor_agent_direct_patch = True