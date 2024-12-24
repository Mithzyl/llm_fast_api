from typing import Annotated, List, Dict, Optional

from fastapi import Depends
from langchain_core.runnables.graph import Node
from langsmith import traceable
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.llm_api import LlmApi
from utils.util import draw_lang_graph_flow


class State(TypedDict):
    """
    maintain and track information during the langgraph workflow
    message: user message input
    history_messages: chat history messages
    title: title of this conversation
    response: LLM generated response
    prompt_template: custom prompt template for specific tasks
    model: Currently selected model
    conversation_memory: short-term memory for this conversation
    user_memory: long-term memory for user
    user_id: The id of the user
    """
    message: Optional[str]
    history_messages: Optional[List[str] | None]
    title: Optional[str]
    web_search_result: Optional[dict]
    rag_context: Optional[dict]
    response: Optional[dict]
    prompt_template: Optional[dict]
    model: Optional[dict]
    conversation_memory: Optional[dict]
    user_memory: Optional[dict]
    user_id: str
    conversation_id: str

    call_web_search: bool

class LlmGraph:
    def __init__(self, llm_api: LlmApi):
        """
        class for langgraph agent workflow management
        Args:
            llm_api: api wrapper
        """
        self.graph = StateGraph(State)
        self.llm_api = llm_api

    def _draw_graph(self):
        draw_lang_graph_flow(self.graph.compile())

    def create_input_node(self, state: dict) -> dict:
        """

        Args:
            state: converts the input dict to a langgraph state.

        Returns:
            a state only with the user message input
        """
        return state

    @traceable
    def run_chat_workflow(self, conversation_id: str,
                          user_message: str,
                          history_messages: List[str],
                          user_id: str) -> dict:
        """
        build the chat workflow using lang graph, which includes a classification of first chat,
         where a title will be generated.
         graph: input -> first_chat_tool -> search user memory -> search conversation memory
             -> rewrite prompt(add the memory) -> llm_call
        Args:
            user_message: the message input from the user
            history_messages: a list of history messages from previous interactions
            user_id: the id of the user

        Returns:
            the state after the api call
        """

        self.graph.add_node("create_input_node", self.create_input_node)
        self.graph.add_node("generate_conversation_title", self.llm_api.generate_conversation_title)
        self.graph.add_node("generate_conversation_response", self.llm_api.chat)
        self.graph.add_node("construct_prompt", self.llm_api.construct_prompt)
        self.graph.add_node("search_conversation_memory", self.llm_api.search_conversation_memory)
        self.graph.add_node("search_user_memory", self.llm_api.search_user_memory)
        self.graph.add_node("search_web", self.llm_api.search_web)
        self.graph.add_node("search_local_vector", self.llm_api.search_rag_context)

        self.graph.set_entry_point("create_input_node")

        self.graph.add_conditional_edges(
            "create_input_node",
            self._classify_first_chat_tool,
            {"first_chat": "generate_conversation_title",
             "continued_chat": "search_user_memory"}
        )

        self.graph.add_edge("generate_conversation_title", "search_user_memory")
        self.graph.add_edge("search_user_memory", "search_conversation_memory")
        self.graph.add_edge("search_conversation_memory", "search_local_vector")

        self.graph.add_conditional_edges(
            "search_local_vector",
            self._classify_web_search_tool,
            {"web_search": "search_web",
                      "prompt_construct": "construct_prompt",}
        )

        self.graph.add_edge("search_web", "construct_prompt")
        self.graph.add_edge("construct_prompt", "generate_conversation_response")


        self.graph.add_edge("generate_conversation_response", END)

        try:
            graph = self.graph.compile()
            # self._draw_graph()

            # # Use the astream method for streaming
            # async for event in graph.astream({"message": user_message,
            #                                   "history_messages": history_messages,
            #                                   "user_id": user_id}):
            #     print(event)

            finish_state = graph.invoke({"conversation_id": conversation_id,
                                         "message": user_message,
                                         "history_messages": history_messages,
                                         "user_id": user_id})

            return finish_state

        except Exception as e:
            print(e)
            raise e

    @traceable
    def build_rag_flow(self):
        pass

    def _classify_first_chat_tool(self, state: State) -> str:
        """
        decide whether this is a first chat or a continued chat
        the first chat will require a title generation edge for langgraph
        Args:
            state:

        Returns:
            strings of classification result
        """
        print("running chat classification")
        print(state)
        return "continued_chat" if state["history_messages"] else "first_chat"

    def _classify_web_search_tool(self, state: State) -> str:
        call_web_search = state.get("call_web_search", False)
        if call_web_search:
            return "web_search"
        else:
            return "prompt_construct"




