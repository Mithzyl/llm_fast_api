from typing import Annotated, List, Dict, Optional

from fastapi import Depends
from langchain_core.runnables.graph import Node
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.llm_api import LlmApi
from utils.util import draw_lang_graph_flow


class State(TypedDict):
    message: Optional[str]
    history_messages: Optional[List[str] | None]
    title: Optional[str]
    response: Optional[dict]
    prompt_template: Optional[dict]
    model: Optional[dict]

class LlmGraph:
    def __init__(self, llm_api: LlmApi):
        self.graph = StateGraph(State)
        self.llm_api = llm_api

    def __draw_graph(self):
        draw_lang_graph_flow(self.graph.compile())

    def create_input_node(self, input_prompt: List[dict]) -> List[dict]:
        return input_prompt

    def run_first_chat_workflow(self, user_message: str, history_messages: List[str]):
        # try for lang graph with debugging first
        self.graph.add_node("create_input_node", self.create_input_node)
        self.graph.add_node("generate_conversation_title", self.llm_api.generate_conversation_title)
        self.graph.add_node("generate_conversation_response", self.llm_api.chat)

        self.graph.set_entry_point("create_input_node")

        self.graph.add_conditional_edges(
            "create_input_node",
            self.__classify_first_chat_tool,
            {"first_chat": "generate_conversation_title",
             "continued_chat": "generate_conversation_response"}
        )
        # self.graph.add_edge("create_input_node", "generate_conversation_title")
        # self.graph.add_edge("create_input_node", "generate_conversation_response")
        self.graph.add_edge("generate_conversation_title", "generate_conversation_response")
        self.graph.add_edge("generate_conversation_response", END)

        try:
            graph = self.graph.compile()
            # self.__draw_graph()

            finish_state = graph.invoke({"message": user_message,
                                         "history_messages": history_messages})

            # for event in graph.stream({"message": user_message,
            #                              "history_messages": history_messages}):
            #     print(event)

            return finish_state

        except Exception as e:
            print(e)
            raise e

    def __classify_first_chat_tool(self, state: State):
        """
        decide whether this is a first chat or a continued chat
        the first chat will require a title generation edge for langgraph
        """
        print("running chat classification")
        print(state)
        return "continued_chat" if state["history_messages"] else "first_chat"




