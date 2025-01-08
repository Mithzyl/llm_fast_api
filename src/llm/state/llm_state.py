import operator
import traceback
from typing import Annotated, List, Dict, Optional

from fastapi import Depends
from langchain_core.runnables.graph import Node
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.llm_api import LlmApi
from utils.util import draw_lang_graph_flow
from llm.state.planner_state import ReWOO, PlannerWorkflow

def merge_dicts(state: Dict, new_data: Dict) -> Dict:
    """Merges two dictionaries."""
    updated_state = state.copy()
    updated_state.update(new_data)
    return updated_state

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
    message: Annotated[list, add_messages]
    history_messages: Annotated[list, add_messages]
    title: str
    # web_search_result: Optional[dict]
    # rag_context: Optional[dict]
    response: Annotated[list, add_messages]
    prompt_template: Annotated[list, add_messages]
    conversation_memory: Annotated[list, add_messages]
    user_memory: Annotated[list, add_messages]
    user_id: str
    conversation_id: str

    call_web_search: bool
    task: str
    # plan_string: Annotated[str, operator.concat]
    steps: Annotated[list, operator.add]
    # results: Annotated[dict, merge_dicts]  # results of ReWoo planning
    results: dict
    result: Annotated[list, add_messages]
    node_history: Annotated[list, operator.add]


class LlmGraph:
    def __init__(self, llm_api: LlmApi):
        """
        class for langgraph agent workflow management
        Args:
            llm_api: api wrapper
        """
        self.graph = StateGraph(State)
        self.llm_api = llm_api
        self.planner = PlannerWorkflow()


    def _draw_graph(self, graph: CompiledStateGraph):
        draw_lang_graph_flow(graph)

    def create_input_node(self, state: dict) -> dict:
        """
        Initialize the state with default values for lists and other fields
        Args:
            state: converts the input dict to a langgraph state.

        Returns:
            a state with initialized fields
        """
        # Initialize lists
        state["history_messages"] = state.get("history_messages", [{
            "role": "system",
            "content": "history message initialized"
        }])
        state["steps"] = state.get("steps", [])
        state["message"] = state.get("message", "")
        
        # Initialize optional fields
        state["title"] = state.get("title", None)
        state["web_search_result"] = state.get("web_search_result", None)
        state["rag_context"] = state.get("rag_context", None)
        state["response"] = state.get("response", None)
        state["prompt_template"] = state.get("prompt_template", None)
        state["model"] = state.get("model", None)
        state["conversation_memory"] = state.get("conversation_memory", [{
            "role": "system",
            "content": "Conversation history initialized"
        }])
        state["user_memory"] = state.get("user_memory", [{
            "role": "system",
            "content": "User memory initialized"
        }])
        state["steps"] = state.get("steps", [])
        state["results"] = state.get("results", [])
        state["result"] = state.get("result", [])
        state["plan_string"] = state.get("plan_string", None)
        state["task"] = state.get("task", "")
        state["node_history"] = []
        
        # Ensure required fields are present
        if "user_id" not in state:
            raise ValueError("user_id is required")
        if "conversation_id" not in state:
            raise ValueError("conversation_id is required")
        
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

    @traceable
    async def run_integrated_workflow(self, conversation_id: str,
                                      user_message: str,
                                      history_messages: List[str],
                                      user_id: str) -> dict:
        """
        Build an integrated workflow that combines title generation, memory retrieval, and planning
        flow: input -> (title generation | memory retrieval) -> planner subgraph -> add memory
        
        Args:
            conversation_id: unique identifier for the conversation
            user_message: the message input from the user
            history_messages: a list of history messages from previous interactions
            user_id: the id of the user

        Returns:
            the final state after workflow execution
        """
        # Create planner subgraph
        # planner_graph = self._create_planner_subgraph()
        # self._draw_graph(planner_graph)

        # Add nodes
        self.graph.add_node("create_input_node", self.create_input_node)
        self.graph.add_node("generate_conversation_title", self.llm_api.generate_conversation_title)
        self.graph.add_node("search_user_memory", self.llm_api.search_user_memory)
        self.graph.add_node("search_conversation_memory", self.llm_api.search_conversation_memory)
        self.graph.add_node("construct_prompt", self.llm_api.construct_prompt)
        
        # Add planner subgraph as a node
        # self.graph.add_node("planner_workflow", planner_graph)
        self.graph.add_node("get_plan", self.llm_api.get_plan)
        self.graph.add_node("tool_execution", self.llm_api.tool_execution)
        self.graph.add_node("solve", self.llm_api.solve)

        # node for joining result
        self.graph.add_node("join_result", self._join_results)

        # Set entry point
        self.graph.set_entry_point("create_input_node")

        # Add parallel execution paths
        self.graph.add_edge("create_input_node", "search_user_memory")
        self.graph.add_edge("create_input_node", "search_conversation_memory")
        self.graph.add_edge("search_user_memory", "join_result")
        self.graph.add_edge("search_conversation_memory", "join_result")
        self.graph.add_edge("join_result", "construct_prompt")


        # Continue with memory retrieval path
        self.graph.add_edge("construct_prompt", "get_plan")
        self.graph.add_edge("get_plan", "tool_execution")
        self.graph.add_edge("tool_execution", "solve")


        self.graph.add_conditional_edges("solve",
                                         self._classify_first_chat_tool,
                                         {"first_chat": "generate_conversation_title",
                                          "continued_chat": END})

        self.graph.add_edge("generate_conversation_title", END)

        try:
            graph = self.graph.compile()
            # self._draw_graph(graph)
            for s in graph.stream({
                "conversation_id": conversation_id,
                "message": user_message,
                "user_id": user_id,
                "task": user_message,  # Use the user message as the task for planning
            }, stream_mode=["messages"],
            config={"recursion_limit": 10}):
                try:
                    if s[1][1]['langgraph_node'] == 'solve':
                        yield s
                except Exception as e:
                    raise e
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            raise e


    def _create_planner_subgraph(self) -> CompiledStateGraph:
        """
        Create a subgraph for the planner workflow with prompt reconstruction
        flow: plan -> tool -> construct_prompt -> solve -> END
        """
        planner_graph = StateGraph(State)
        
        # Add planner nodes
        planner_graph.add_node("plan", self.planner.get_plan)
        planner_graph.add_node("tool", self.planner.tool_execution)
        planner_graph.add_node("construct_prompt", self.llm_api.construct_prompt)
        planner_graph.add_node("solve", self.planner.solve)

        # Set entry point
        planner_graph.set_entry_point("plan")

        # Add edges
        planner_graph.add_edge("plan", "tool")

        # Add conditional edges for tool execution loop
        planner_graph.add_conditional_edges(
            "tool",
            self.planner.route,
            {
                "solve": "construct_prompt",
                "tool": "tool"
            }
        )

        # Add final edges and ensure termination
        planner_graph.add_edge("construct_prompt", "solve")
        planner_graph.add_edge("solve", END)  # Add explicit END edge

        return planner_graph.compile()

    def _add_memory(self, state: State) -> dict:
        """
        Add the conversation result to both user and conversation memory
        """
        try:
            # Add to conversation memory
            self.llm_api.memory_client.add_memory_by_conversation_id(
                f"User: {state['message']}\nAssistant: {state['result']}",
                conversation_id=state['conversation_id']
            )

            # Add to user memory
            self.llm_api.memory_client.add_memory_by_user_id(
                f"User: {state['message']}\nAssistant: {state['result']}",
                user_id=state['user_id']
            )

            return state
        except Exception as e:
            print(f"Error adding memory: {e}")
            return state

    def _join_results(self, state: State) -> dict:
        """
        Join the results from parallel execution paths (title generation and memory retrieval)
        """
        # The state already contains all results from both paths
        # We just need to ensure it's properly formatted
        return state

    # def _route(self, state: State) -> str:
    #     """
    #     Updated routing logic to handle prompt construction
    #     """
    #     _step = self._get_current_task(state)
    #     if _step is None:
    #         # We have executed all tasks, move to prompt construction
    #         return "solve"
    #     else:
    #         # We are still executing tasks, loop back to the "tool" node
    #         return "tool"




