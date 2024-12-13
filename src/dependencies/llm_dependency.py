from deprecated.sphinx import deprecated
from fastapi import Depends
from sqlmodel import Session

from db.db import get_session
from llm.llm_api import LlmApi
from llm.llm_state import LlmGraph
from llm.mem0.mem0_client import MemoryClient, mem0_config
from models.param.message_param import ChatCreateParam
from services.llm_service import LlmService


def get_llm_service(session: Session = Depends(get_session)) -> LlmService:
    return LlmService(session)

@deprecated(version="1.0", reason="The llm api is wrapped into llm graph for llm agent workflow")
def get_llm_api(llm_param: ChatCreateParam) -> LlmApi:
    default_model = 'qwen2:0.5b'
    if not llm_param.model:
        model = default_model
    else:
        model = llm_param.model
    return LlmApi(model=model, temperature=llm_param.temperature)

def get_llm_graph(llm_param: ChatCreateParam) -> LlmGraph:
    default_model = 'qwen2:0.5b'
    if not llm_param.model:
        model = default_model
    else:
        model = llm_param.model

    llm_api = LlmApi(model=model, temperature=llm_param.temperature)

    return LlmGraph(llm_api)