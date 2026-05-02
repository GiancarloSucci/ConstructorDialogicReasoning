from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional
import xml.etree.ElementTree as ET

import requests

from constructor_adapter import StatelessConstructorAdapter


DEFAULT_API_URL = "https://training.constructor.app/api/platform-kmapi/v1"


@dataclass(frozen=True)
class DialogicStep:
    id: str
    llm_alias: str
    mode: str
    role: str
    description: str
    timeout: int
    request_timeout: int
    retry_delay: int


@dataclass(frozen=True)
class DialogicConfiguration:
    name: str
    steps: list[DialogicStep]


class StatelessConstructorAdapterDialogic(StatelessConstructorAdapter):
    """
    Dialogic Constructor adapter.

    This adapter is compatible with ConstructorAdapter because it keeps the
    standard query(question: str, **kwargs) interface.

    If no XML configuration is loaded, query() delegates to the normal
    StatelessConstructorAdapter.query().

    If an XML configuration is loaded, query() executes the same user question
    through the configured sequence of endpoints. Each endpoint after the first
    receives the original question and the previous answer, and is asked to
    assess and improve it.
    """

    def __init__(self, **kwargs: Any) -> None:
        resolved_kwargs = self._prepare_constructor_kwargs(kwargs)

        super().__init__(**resolved_kwargs)

        self.configuration_xml: Optional[ET.ElementTree] = None
        self.dialogic_configuration: Optional[DialogicConfiguration] = None
        self._force_direct_mode: bool = resolved_kwargs["mode"] == "direct"

    def loadConfiguration(self, configuration_path: str | Path) -> None:
        """
        Load the XML configuration describing the dialogic path.

        Expected XML structure:

        <?xml version="1.0" encoding="UTF-8"?>
        <agentPath name="constructor-stateful-review-chain">
            <endpoint
                id="first"
                llm_alias="gpt-4o-mini"
                mode="direct"
                role="initial_answer"
                timeout="900"
                request_timeout="30"
                retry_delay="5">
                <description>Produce the first answer.</description>
            </endpoint>
        </agentPath>
        """

        path = Path(configuration_path)

        if not path.exists():
            raise FileNotFoundError(f"Dialogic configuration file not found: {path}")

        tree = ET.parse(path)
        root = tree.getroot()

        if root.tag != "agentPath":
            raise ValueError("Root XML element must be <agentPath>.")

        steps: list[DialogicStep] = []

        for endpoint_node in root.findall("endpoint"):
            endpoint_id = endpoint_node.attrib.get("id")
            llm_alias = endpoint_node.attrib.get("llm_alias")
            mode = endpoint_node.attrib.get("mode", "direct")
            role = endpoint_node.attrib.get("role", "review_and_improve")

            if not endpoint_id:
                raise ValueError("Each <endpoint> must define an id attribute.")

            if not llm_alias:
                raise ValueError(
                    f"Endpoint {endpoint_id} must define an llm_alias attribute."
                )

            if mode not in {"direct", "model"}:
                raise ValueError(
                    f"Endpoint {endpoint_id} has invalid mode {mode!r}. "
                    "Allowed values are 'direct' and 'model'."
                )

            description_node = endpoint_node.find("description")
            description = (
                description_node.text.strip()
                if description_node is not None and description_node.text
                else ""
            )

            steps.append(
                DialogicStep(
                    id=endpoint_id,
                    llm_alias=llm_alias,
                    mode=mode,
                    role=role,
                    description=description,
                    timeout=int(endpoint_node.attrib.get("timeout", "300")),
                    request_timeout=int(
                        endpoint_node.attrib.get("request_timeout", "15")
                    ),
                    retry_delay=int(endpoint_node.attrib.get("retry_delay", "3")),
                )
            )

        if not steps:
            raise ValueError("The dialogic XML configuration contains no endpoints.")

        self.configuration_xml = tree
        self.dialogic_configuration = DialogicConfiguration(
            name=root.attrib.get("name", "unnamed-dialogic-configuration"),
            steps=steps,
        )

    def load_configuration(self, configuration_path: str | Path) -> None:
        """
        Python-style alias for loadConfiguration().
        """
        self.loadConfiguration(configuration_path)

    def query(self, question: str, **kwargs: Any) -> str:
        """
        Execute the query.

        If no XML configuration was loaded, delegate to the standard stateless
        adapter. If a configuration was loaded, execute the configured dialogic
        chain and return the final answer plus a short execution description.
        """

        if self.dialogic_configuration is None:
            return super().query(question, **kwargs)

        current_answer: Optional[str] = None
        current_endpoint_id: Optional[str] = None
        trace: list[dict[str, str]] = []

        for step in self.dialogic_configuration.steps:
            effective_step = self._effective_step(step)

            prompt = self._build_dialogic_prompt(
                original_question=question,
                step=effective_step,
                previous_endpoint_id=current_endpoint_id,
                previous_answer=current_answer,
            )

            answer = self._query_single_step(
                step=effective_step,
                prompt=prompt,
                **kwargs,
            )

            trace.append(
                {
                    "endpoint_id": effective_step.id,
                    "llm_alias": effective_step.llm_alias,
                    "mode": effective_step.mode,
                    "role": effective_step.role,
                    "answer": answer,
                }
            )

            current_endpoint_id = effective_step.id
            current_answer = answer

        if current_answer is None:
            raise RuntimeError("Dialogic execution produced no answer.")

        return self._build_final_response(
            final_answer=current_answer,
            trace=trace,
        )

    def _query_single_step(
        self,
        step: DialogicStep,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        adapter = StatelessConstructorAdapter(
            mode=step.mode,
            api_url=self.api_url,
            api_key=self.api_key,
            km_id=self.km_id,
            llm_alias=step.llm_alias,
        )

        return adapter.query(prompt, **kwargs)

    def _build_dialogic_prompt(
        self,
        original_question: str,
        step: DialogicStep,
        previous_endpoint_id: Optional[str],
        previous_answer: Optional[str],
    ) -> str:
        if previous_answer is None:
            return (
                f"You are endpoint '{step.id}' with role '{step.role}'.\n\n"
                f"Endpoint description:\n{step.description}\n\n"
                f"Original user question:\n{original_question}\n\n"
                "Task:\n"
                "Produce a precise, complete, and well-structured answer to "
                "the user question.\n\n"
                "Return only the answer."
            )

        return (
            f"You are endpoint '{step.id}' with role '{step.role}'.\n\n"
            f"Endpoint description:\n{step.description}\n\n"
            f"Original user question:\n{original_question}\n\n"
            f"Previous endpoint:\n{previous_endpoint_id}\n\n"
            f"Previous answer:\n{previous_answer}\n\n"
            "Task:\n"
            "1. Decide whether the previous answer is correct, complete, "
            "and clear.\n"
            "2. Identify defects, omissions, ambiguities, or unsupported "
            "claims.\n"
            "3. Improve the answer.\n"
            "4. Preserve correct content from the previous answer.\n"
            "5. Do not merely comment on the previous answer; return an "
            "improved answer.\n\n"
            "Return only the improved answer."
        )

    def _build_final_response(
        self,
        final_answer: str,
        trace: list[dict[str, str]],
    ) -> str:
        lines: list[str] = []

        lines.append(final_answer)
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("How this answer was produced:")
        lines.append("")

        for index, item in enumerate(trace, start=1):
            lines.append(
                f"{index}. endpoint={item['endpoint_id']}, "
                f"llm_alias={item['llm_alias']}, "
                f"mode={item['mode']}, "
                f"role={item['role']}"
            )

        return "\n".join(lines)

    def _effective_step(self, step: DialogicStep) -> DialogicStep:
        if self._force_direct_mode and step.mode != "direct":
            return DialogicStep(
                id=step.id,
                llm_alias=step.llm_alias,
                mode="direct",
                role=step.role,
                description=step.description,
                timeout=step.timeout,
                request_timeout=step.request_timeout,
                retry_delay=step.retry_delay,
            )

        return step

    @classmethod
    def _prepare_constructor_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(kwargs)

        mode = prepared.get("mode") or "direct"
        api_url = (
            prepared.get("api_url")
            or os.getenv("CONSTRUCTOR_API_URL")
            or DEFAULT_API_URL
        ).rstrip("/")
        api_key = prepared.get("api_key") or os.getenv("CONSTRUCTOR_API_KEY")
        km_id = (
            prepared.get("km_id")
            or os.getenv("CONSTRUCTOR_KM_ID")
            or os.getenv("KNOWLEDGE_MODEL_ID")
        )
        llm_alias = prepared.get("llm_alias")

        if not api_key:
            raise RuntimeError("Missing CONSTRUCTOR_API_KEY.")

        if not cls._is_km_accessible(
            api_url=api_url,
            api_key=api_key,
            km_id=km_id,
        ):
            km_id = cls._create_temporary_km(
                api_url=api_url,
                api_key=api_key,
            )
            mode = "direct"

        prepared["mode"] = mode
        prepared["api_url"] = api_url
        prepared["api_key"] = api_key
        prepared["km_id"] = km_id

        if llm_alias is not None:
            prepared["llm_alias"] = llm_alias

        return prepared

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"X-KM-AccessKey": f"Bearer {api_key}"}

    @classmethod
    def _is_km_accessible(
        cls,
        api_url: str,
        api_key: str,
        km_id: Optional[str],
    ) -> bool:
        if not km_id or not str(km_id).strip():
            return False

        try:
            response = requests.get(
                f"{api_url}/knowledge-models/{str(km_id).strip()}",
                headers=cls._headers(api_key),
                timeout=30,
            )
        except requests.RequestException:
            return False

        return response.status_code == 200

    @classmethod
    def _create_temporary_km(
        cls,
        api_url: str,
        api_key: str,
    ) -> str:
        payload = {
            "name": "constructor-adapter-dialogic-runtime",
            "description": (
                "Temporary empty knowledge model used as a technical "
                "container for dialogic direct LLM calls."
            ),
            "shared_type": "private",
            "share_documents": False,
        }

        response = requests.post(
            f"{api_url}/knowledge-models",
            headers=cls._headers(api_key),
            json=payload,
            timeout=30,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                "Cannot create temporary knowledge model for "
                "StatelessConstructorAdapterDialogic. "
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

        return str(km_id)