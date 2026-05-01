from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, TypedDict
import operator

Mode = Literal["direct", "model"]


@dataclass(frozen=True)
class EndpointSpec:
    """One step in the XML-defined agent path."""

    id: str
    llm_alias: str
    role: str
    mode: Mode = "direct"
    description: str = ""
    timeout: int = 300
    request_timeout: int = 15
    retry_delay: int = 3


@dataclass(frozen=True)
class AgentPathConfig:
    """Complete XML-defined route."""

    name: str
    endpoints: tuple[EndpointSpec, ...]

    def validate(self) -> None:
        if not self.endpoints:
            raise ValueError("The path configuration must contain at least one endpoint.")
        ids = [endpoint.id for endpoint in self.endpoints]
        duplicated = sorted({endpoint_id for endpoint_id in ids if ids.count(endpoint_id) > 1})
        if duplicated:
            raise ValueError(f"Duplicated endpoint ids in XML configuration: {duplicated}")


@dataclass(frozen=True)
class EndpointExchange:
    """Audit record for one endpoint invocation."""

    endpoint_id: str
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

    original_query: str
    current_answer: Optional[str]
    current_endpoint_id: Optional[str]
    exchanges: Annotated[list[EndpointExchange], operator.add]
    final_answer: Optional[str]
    explanation: Optional[str]
