from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from constructor_agent.stateful_constructor_client import StatefulConstructorClient
from constructor_agent.domain import AgentState, QuestionExchange, QuestionPathConfig, QuestionSpec
from constructor_agent.langchain_constructor_model import ConstructorStatefulChatModel
from constructor_agent.prompts import PromptFactory


class AgentGraphBuilder:
    """Builder pattern: creates a LangGraph from QuestionPathConfig."""

    def __init__(
        self,
        client: StatefulConstructorClient,
        prompt_factory: PromptFactory | None = None,
    ) -> None:
        self.client = client
        self.prompt_factory = prompt_factory or PromptFactory()

    def build(self, config: QuestionPathConfig):
        config.validate()
        graph = StateGraph(AgentState)

        for question in config.questions:
            graph.add_node(question.id, self._node_for_question(question))

        graph.add_node("finalize", self._finalize)
        graph.set_entry_point(config.questions[0].id)

        for current_question, next_question in zip(config.questions, config.questions[1:]):
            graph.add_edge(current_question.id, next_question.id)

        graph.add_edge(config.questions[-1].id, "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _node_for_question(self, question: QuestionSpec) -> Callable[[AgentState], dict]:
        def node(state: AgentState) -> dict:
            prompt = self.prompt_factory.build_question_prompt(question, state)
            model = ConstructorStatefulChatModel(client=self.client, question=question)
            result = model.invoke([HumanMessage(content=prompt)])
            answer = str(result.content)

            exchange = QuestionExchange(
                question_id=question.id,
                llm_alias=question.llm_alias,
                role=question.role,
                mode=question.mode,
                prompt=prompt,
                answer=answer,
            )

            return {
                "current_answer": answer,
                "current_question_id": question.id,
                "exchanges": [exchange],
            }

        return node

    def _finalize(self, state: AgentState) -> dict:
        return {
            "final_answer": state.get("current_answer"),
            "explanation": self.prompt_factory.build_final_explanation(state),
        }