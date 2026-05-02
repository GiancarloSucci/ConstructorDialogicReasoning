from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Optional, TypedDict
import operator

Mode = Literal["direct", "model"]


@dataclass(frozen=True)
class QuestionSpec:
    """One question step in the XML-defined question path."""

    id: str
    llm_alias: str
    role: str
    mode: Mode = "direct"
    description: str = ""
    prompt: Optional[str] = None
    timeout: int = 300
    request_timeout: int = 15
    retry_delay: int = 3


@dataclass(frozen=True)
class QuestionPathConfig:
    """Complete XML-defined question path."""

    name: str
    questions: tuple[QuestionSpec, ...]

    def validate(self) -> None:
        if not self.questions:
            raise ValueError(
                "The question path configuration must contain at least one question."
            )

        ids = [question.id for question in self.questions]
        duplicated = sorted({question_id for question_id in ids if ids.count(question_id) > 1})

        if duplicated:
            raise ValueError(f"Duplicated question ids in XML configuration: {duplicated}")


@dataclass(frozen=True)
class QuestionExchange:
    """Audit record for one question invocation."""

    question_id: str
    llm_alias: str
    role: str
    mode: str
    prompt: str
    answer: str


class AgentState(TypedDict, total=False):
    """
    LangGraph state.

    The Annotated list tells LangGraph to append exchanges produced by nodes.
    """

    original_prompt: str
    current_answer: Optional[str]
    current_question_id: Optional[str]
    exchanges: Annotated[list[QuestionExchange], operator.add]
    final_answer: Optional[str]
    explanation: Optional[str]