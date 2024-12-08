from fastapi import Depends
from sqlmodel import Session

from db.db import get_session
from llm.llm_api import LlmApi
from models.param.message_param import ChatCreateParam
from services.llm_service import LlmService


def get_llm_service(session: Session = Depends(get_session)) -> LlmService:
    return LlmService(session)


def get_llm_api(llm_param: ChatCreateParam, temperature: float = 0.9) -> LlmApi:
    default_model = 'qwen2:0.5b'
    if not llm_param.model:
        model = default_model
    else:
        model = llm_param.model
    return LlmApi(model=model, temperature=temperature)
