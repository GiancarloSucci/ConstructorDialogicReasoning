from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from constructor_agent.statefull_constructor_client import StatefulConstructorClient
from constructor_agent.domain import QuestionSpec


class ConstructorStatefulChatModel(BaseChatModel):
    """LangChain chat model backed by StatefulConstructorAdapter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: StatefulConstructorClient = Field(exclude=True)
    question: QuestionSpec

    @property
    def _llm_type(self) -> str:
        return "constructor_stateful_adapter"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "question_id": self.question.id,
            "llm_alias": self.question.llm_alias,
            "mode": self.question.mode,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._messages_to_constructor_prompt(messages)
        answer = self.client.ask(self.question, prompt)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=answer),
                )
            ]
        )

    @staticmethod
    def _messages_to_constructor_prompt(messages: list[BaseMessage]) -> str:
        parts: list[str] = []

        for message in messages:
            role = message.type.upper()
            content = message.content
            parts.append(f"{role}:\n{content}")

        return "\n\n".join(parts)