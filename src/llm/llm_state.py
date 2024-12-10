from typing import Annotated, List, Dict, Optional

from langchain_core.runnables.graph import Node
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    message: Optional[str] = None
    title: Optional[str] = None
    response: Optional[dict] = None
    prompt_template: Optional[dict] = None
