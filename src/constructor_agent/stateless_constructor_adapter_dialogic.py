from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional
import xml.etree.ElementTree as ET

import requests

from constructor_adapter import StatelessConstructorAdapter
from constructor_agent.platform_config import DEFAULT_API_URL


@dataclass(frozen=True)
class DialogicQuestion:
    id: str
    llm_alias: str
    mode: str
    role: str
    description: str
    prompt: Optional[str]
    timeout: int
    request_timeout: int
    retry_delay: int


@dataclass(frozen=True)
class DialogicConfiguration:
    name: str
    questions: list[DialogicQuestion]


class StatelessConstructorAdapterDialogic(StatelessConstructorAdapter):
    """
    Dialogic Constructor adapter.

    This adapter keeps the standard query(prompt: str, **kwargs) interface.

    If no XML configuration is loaded, query() delegates to the normal
    StatelessConstructorAdapter.query().

    If an XML configuration is loaded, query() executes the prompt through the
    configured sequence of questions.

    Placeholder semantics:
        {prompt}
            Original user prompt.

        {answer}
            Answer produced by the previous question.

        {question_id.prompt}
            Prompt actually sent to a previous question.

        {question_id.answer}
            Answer produced by a previous question.
    """

    def __init__(self, **kwargs: Any) -> None:
        resolved_kwargs = self._prepare_constructor_kwargs(kwargs)

        super().__init__(**resolved_kwargs)

        self.configuration_xml: Optional[ET.ElementTree] = None
        self.dialogic_configuration: Optional[DialogicConfiguration] = None
        self._force_direct_mode: bool = resolved_kwargs["mode"] == "direct"

    def loadConfiguration(self, configuration_path: str | Path) -> None:
        path = Path(configuration_path)

        if not path.exists():
            raise FileNotFoundError(f"Dialogic configuration file not found: {path}")

        tree = ET.parse(path)
        root = tree.getroot()

        if root.tag != "questionPath":
            raise ValueError("Root XML element must be <questionPath>.")

        questions: list[DialogicQuestion] = []

        for question_node in root.findall("question"):
            question_id = question_node.attrib.get("id")
            llm_alias = question_node.attrib.get("llm_alias")
            mode = question_node.attrib.get("mode", "direct")
            role = question_node.attrib.get("role", "review_and_improve")

            if not question_id:
                raise ValueError("Each <question> must define an id attribute.")

            if not llm_alias:
                raise ValueError(
                    f"Question {question_id} must define an llm_alias attribute."
                )

            if mode not in {"direct", "model"}:
                raise ValueError(
                    f"Question {question_id} has invalid mode {mode!r}. "
                    "Allowed values are 'direct' and 'model'."
                )

            description_node = question_node.find("description")
            description = (
                description_node.text.strip()
                if description_node is not None and description_node.text
                else ""
            )

            prompt_node = question_node.find("prompt")
            question_prompt = (
                prompt_node.text.strip()
                if prompt_node is not None and prompt_node.text
                else None
            )

            questions.append(
                DialogicQuestion(
                    id=question_id,
                    llm_alias=llm_alias,
                    mode=mode,
                    role=role,
                    description=description,
                    prompt=question_prompt,
                    timeout=int(question_node.attrib.get("timeout", "300")),
                    request_timeout=int(
                        question_node.attrib.get("request_timeout", "15")
                    ),
                    retry_delay=int(question_node.attrib.get("retry_delay", "3")),
                )
            )

        if not questions:
            raise ValueError("The dialogic XML configuration contains no questions.")

        self.configuration_xml = tree
        self.dialogic_configuration = DialogicConfiguration(
            name=root.attrib.get("name", "unnamed-dialogic-configuration"),
            questions=questions,
        )

    def load_configuration(self, configuration_path: str | Path) -> None:
        self.loadConfiguration(configuration_path)

    def query(self, prompt: str, **kwargs: Any) -> str:
        if self.dialogic_configuration is None:
            return super().query(prompt, **kwargs)

        if not prompt.strip():
            raise ValueError("The prompt is empty.")

        original_prompt = prompt.strip()

        current_answer: Optional[str] = None
        current_question_id: Optional[str] = None
        prompt_by_question_id: dict[str, str] = {}
        answer_by_question_id: dict[str, str] = {}
        trace: list[dict[str, str]] = []

        for question in self.dialogic_configuration.questions:
            effective_question = self._effective_question(question)

            question_prompt = self._build_dialogic_prompt(
                original_prompt=original_prompt,
                question=effective_question,
                previous_question_id=current_question_id,
                previous_answer=current_answer,
                prompt_by_question_id=prompt_by_question_id,
                answer_by_question_id=answer_by_question_id,
            )

            answer = self._query_single_question(
                question=effective_question,
                prompt=question_prompt,
                default_kwargs=kwargs,
            )

            prompt_by_question_id[effective_question.id] = question_prompt
            answer_by_question_id[effective_question.id] = answer

            trace.append(
                {
                    "question_id": effective_question.id,
                    "llm_alias": effective_question.llm_alias,
                    "mode": effective_question.mode,
                    "role": effective_question.role,
                    "prompt": question_prompt,
                    "answer": answer,
                }
            )

            current_question_id = effective_question.id
            current_answer = answer

        if current_answer is None:
            raise RuntimeError("Dialogic execution produced no answer.")

        return self._build_final_response(
            final_answer=current_answer,
            trace=trace,
        )

    def _query_single_question(
        self,
        question: DialogicQuestion,
        prompt: str,
        default_kwargs: dict[str, Any],
    ) -> str:
        adapter = StatelessConstructorAdapter(
            mode=question.mode,
            api_url=self.api_url,
            api_key=self.api_key,
            km_id=self.km_id,
            llm_alias=question.llm_alias,
        )

        query_kwargs = dict(default_kwargs)

        query_kwargs.setdefault("timeout", question.timeout)
        query_kwargs.setdefault("request_timeout", question.request_timeout)
        query_kwargs.setdefault("retry_delay", question.retry_delay)

        return adapter.query(prompt, **query_kwargs)

    def _build_dialogic_prompt(
        self,
        original_prompt: str,
        question: DialogicQuestion,
        previous_question_id: Optional[str],
        previous_answer: Optional[str],
        prompt_by_question_id: dict[str, str],
        answer_by_question_id: dict[str, str],
    ) -> str:
        if question.prompt:
            replacements = {
                "prompt": original_prompt,
                "answer": previous_answer or "",
                "input": previous_answer if previous_answer is not None else original_prompt,
                "original_prompt": original_prompt,
                "previous_answer": previous_answer or "",
                "previous_question": previous_question_id or "unknown",
                "question_id": question.id,
                "question_role": question.role,
                "question_description": question.description,
            }

            for question_id, used_prompt in prompt_by_question_id.items():
                replacements[f"{question_id}.prompt"] = used_prompt

            for question_id, produced_answer in answer_by_question_id.items():
                replacements[f"{question_id}.answer"] = produced_answer

            return self._replace_placeholders(question.prompt, replacements)

        if previous_answer is None:
            return (
                f"You are question step '{question.id}' with role '{question.role}'.\n\n"
                f"Question description:\n{question.description}\n\n"
                f"Original user prompt:\n{original_prompt}\n\n"
                "Task:\n"
                "Produce a precise, complete, and well-structured answer to "
                "the user prompt.\n\n"
                "Return only the answer."
            )

        return (
            f"You are question step '{question.id}' with role '{question.role}'.\n\n"
            f"Question description:\n{question.description}\n\n"
            f"Original user prompt:\n{original_prompt}\n\n"
            f"Previous question:\n{previous_question_id}\n\n"
            f"Previous answer:\n{previous_answer}\n\n"
            "Task:\n"
            "1. Decide whether the previous answer is correct, complete, and clear.\n"
            "2. Identify defects, omissions, ambiguities, or unsupported claims.\n"
            "3. Improve the answer.\n"
            "4. Preserve correct content from the previous answer.\n"
            "5. Do not merely comment on the previous answer; return an improved answer.\n\n"
            "Return only the improved answer."
        )

    @staticmethod
    def _replace_placeholders(template: str, replacements: dict[str, str]) -> str:
        result = template

        for key in sorted(replacements.keys(), key=len, reverse=True):
            result = result.replace("{" + key + "}", replacements[key])

        return result

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
                f"{index}. question={item['question_id']}, "
                f"llm_alias={item['llm_alias']}, "
                f"mode={item['mode']}, "
                f"role={item['role']}"
            )

        return "\n".join(lines)

    def _effective_question(self, question: DialogicQuestion) -> DialogicQuestion:
        if self._force_direct_mode and question.mode != "direct":
            return DialogicQuestion(
                id=question.id,
                llm_alias=question.llm_alias,
                mode="direct",
                role=question.role,
                description=question.description,
                prompt=question.prompt,
                timeout=question.timeout,
                request_timeout=question.request_timeout,
                retry_delay=question.retry_delay,
            )

        return question

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