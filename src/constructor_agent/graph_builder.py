from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from constructor_agent.constructor_stateful_client import StatefulConstructorClient
from constructor_agent.domain import AgentPathConfig, AgentState, EndpointExchange, EndpointSpec
from constructor_agent.langchain_constructor_model import ConstructorStatefulChatModel
from constructor_agent.prompts import PromptFactory


class AgentGraphBuilder:
    """Builder pattern: creates a LangGraph from AgentPathConfig."""

    def __init__(
        self,
        client: StatefulConstructorClient,
        prompt_factory: PromptFactory | None = None,
    ) -> None:
        self.client = client
        self.prompt_factory = prompt_factory or PromptFactory()

    def build(self, config: AgentPathConfig):
        config.validate()
        graph = StateGraph(AgentState)

        for endpoint in config.endpoints:
            graph.add_node(endpoint.id, self._node_for_endpoint(endpoint))

        graph.add_node("finalize", self._finalize)
        graph.set_entry_point(config.endpoints[0].id)

        for current_endpoint, next_endpoint in zip(config.endpoints, config.endpoints[1:]):
            graph.add_edge(current_endpoint.id, next_endpoint.id)

        graph.add_edge(config.endpoints[-1].id, "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _node_for_endpoint(self, endpoint: EndpointSpec) -> Callable[[AgentState], dict]:
        def node(state: AgentState) -> dict:
            prompt = self.prompt_factory.build_endpoint_prompt(endpoint, state)
            model = ConstructorStatefulChatModel(client=self.client, endpoint=endpoint)
            result = model.invoke([HumanMessage(content=prompt)])
            answer = str(result.content)

            exchange = EndpointExchange(
                endpoint_id=endpoint.id,
                llm_alias=endpoint.llm_alias,
                role=endpoint.role,
                mode=endpoint.mode,
                prompt=prompt,
                answer=answer,
            )

            return {
                "current_answer": answer,
                "current_endpoint_id": endpoint.id,
                "exchanges": [exchange],
            }

        return node

    def _finalize(self, state: AgentState) -> dict:
        return {
            "final_answer": state.get("current_answer"),
            "explanation": self.prompt_factory.build_final_explanation(state),
        }
