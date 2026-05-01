from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constructor_agent.config_loader import XmlPathConfigLoader
from constructor_agent.constructor_stateful_client import ConstructorPlatformConfig, StatefulConstructorClient
from constructor_agent.domain import AgentPathConfig, AgentState
from constructor_agent.graph_builder import AgentGraphBuilder


@dataclass(frozen=True)
class AgentRunResult:
    final_answer: str
    explanation: str
    state: AgentState


class ConstructorAgentRunner:
    """Facade for configuration loading, graph creation, and execution."""

    def __init__(
        self,
        path_config: AgentPathConfig,
        platform_config: ConstructorPlatformConfig | None = None,
    ) -> None:
        self.path_config = path_config
        self.client = StatefulConstructorClient(platform_config)
        self.graph = AgentGraphBuilder(self.client).build(path_config)

    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        platform_config: ConstructorPlatformConfig | None = None,
    ) -> "ConstructorAgentRunner":
        config = XmlPathConfigLoader().load(xml_path)
        return cls(config, platform_config)

    def run(self, query: str) -> AgentRunResult:
        if not query.strip():
            raise ValueError("The query is empty.")

        initial_state: AgentState = {
            "original_query": query.strip(),
            "current_answer": None,
            "current_endpoint_id": None,
            "exchanges": [],
        }
        state = self.graph.invoke(initial_state)
        return AgentRunResult(
            final_answer=state.get("final_answer") or "",
            explanation=state.get("explanation") or "",
            state=state,
        )
