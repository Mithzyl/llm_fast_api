from fastapi import FastAPI, APIRouter, Depends, Body, Path
from fastapi.openapi.models import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from sqlmodel import Session

from controller.user_controller import security
from db.db import get_session
from models.param.message_param import MessageDao,ChatCreateParam
from models.dto.messgage_dto import Response
# from services import llm_service
from services.llm_service import LlmService, get_llm_service, get_conversation_history_service
from services.user_service import UserService, get_user_service

llm_router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)

#Create a new session of chat
@llm_router.post("/chat", response_model=Response)
async def create_chat(
        create_param: ChatCreateParam = Body(),
        token: HTTPAuthorizationCredentials = Depends(security),
        llm_service=Depends(get_llm_service)
) -> Response:
    return llm_service.create_chat(create_param, token)


# Get model list
@llm_router.get("/get_model", response_model=Response)
async def get_models(llm_service: LlmService = Depends(get_llm_service)) -> Response:
    return llm_service.get_model_list()


@llm_router.get("/conversation/{message_id}", response_model=Response)
async def demo_conversation(message_id: int, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_message_by_message_id(message_id)

@llm_router.get("/{conversation_id}", response_model=Response)
async def get_conversation_history(conversation_id: str, llm_service: LlmService = Depends(get_conversation_history_service)):
    return llm_service.get_messages_by_conversation_id(conversation_id)


@llm_router.get("/{user_id}/sessions", response_model=Response)
async def get_all_sessions_by_user_id(user_id: str, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_sessions_by_user_id(user_id)





