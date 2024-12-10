from typing import Annotated, List, Dict, Optional

from fastapi import Depends
from langchain_core.runnables.graph import Node
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.llm_api import LlmApi
from utils.util import draw_lang_graph_flow


class State(TypedDict):
    message: Optional[str] = None
    title: Optional[str] = None
    response: Optional[dict] = None
    prompt_template: Optional[dict] = None
    model: Optional[dict] = None

class LlmGraph:
    def __init__(self, model: str, temperature: float):
        self.graph = StateGraph(State)
        self.llm_api = LlmApi(model, temperature)

    def __draw_graph(self):
        draw_lang_graph_flow(self.graph)

    def create_input_node(self, input_prompt: List[dict]) -> List[dict]:
        return input_prompt

    def run_first_chat_workflow(self, user_message: str):
        # try for lang graph with debugging first
        self.graph.add_node("create_input_node", self.create_input_node)
        self.graph.add_node("generate_conversation_title", self.llm_api.generate_conversation_title)
        self.graph.add_node("generate_conversation_response", self.llm_api.create_first_chat)

        self.graph.set_entry_point("create_input_node")

        self.graph.add_edge("create_input_node", "generate_conversation_title")
        self.graph.add_edge("create_input_node", "generate_conversation_response")
        self.graph.add_edge("generate_conversation_title", END)
        self.graph.add_edge("generate_conversation_response", END)

        try:
            graph = self.graph.compile()

            finish_state = graph.invoke({"message": user_message})

            return finish_state

        except Exception as e:
            print(e)
            raise e

    def __classify_first_chat(self):
        """
        decide whether this is a first chat or a continued chat
        the first chat will require a title generation edge for langgraph
        """
        pass


