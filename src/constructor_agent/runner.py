from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constructor_agent.config_loader import XmlQuestionConfigLoader
from constructor_agent.domain import AgentState, QuestionPathConfig
from constructor_agent.graph_builder import AgentGraphBuilder

from constructor_agent.stateful_constructor_client import (
    ConstructorPlatformConfig,
    ConstructorQuestionCandidate,
    StatefulConstructorClient,
)


@dataclass(frozen=True)
class AgentRunResult:
    final_answer: str
    explanation: str
    state: AgentState


class ConstructorAgentRunner:
    """Facade for configuration loading, graph creation, and execution."""

    def __init__(
        self,
        question_config: QuestionPathConfig,
        platform_config: ConstructorPlatformConfig | None = None,
    ) -> None:
        self.question_config = question_config
        self.client = StatefulConstructorClient(platform_config)
        self.graph = AgentGraphBuilder(self.client).build(question_config)

    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        platform_config: ConstructorPlatformConfig | None = None,
    ) -> "ConstructorAgentRunner":
        config = XmlQuestionConfigLoader().load(xml_path)
        return cls(config, platform_config)

    def run(self, prompt: str) -> AgentRunResult:
        if not prompt.strip():
            raise ValueError("The prompt is empty.")

        initial_state: AgentState = {
            "original_prompt": prompt.strip(),
            "current_answer": None,
            "current_question_id": None,
            "exchanges": [],
        }

        state = self.graph.invoke(initial_state)

        return AgentRunResult(
            final_answer=state.get("final_answer") or "",
            explanation=state.get("explanation") or "",
            state=state,
        )

    @staticmethod
    def list_constructor_questions(
        platform_config: ConstructorPlatformConfig | None = None,
        include_direct: bool = True,
        include_model: bool = True,
    ) -> list[ConstructorQuestionCandidate]:
        client = StatefulConstructorClient(platform_config)

        return client.list_question_candidates(
            include_direct=include_direct,
            include_model=include_model,
        )