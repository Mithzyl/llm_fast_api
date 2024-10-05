from fastapi import FastAPI, APIRouter, Depends, Body, Path
from fastapi.openapi.models import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from sqlmodel import Session

from controller.user_controller import security
from models.dao.message_dao import MessageDao
from models.dto.messgage_dto import Message
from services import llm_service
from services.llm_service import LlmService, get_llm_service

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)


# @router.get("/")
# async def demo_response():
#     return llm_service.get_response()


@router.get("/conversation/{message_id}", response_model=Message)
async def demo_conversation(message_id: int, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_message_by_message_id(message_id)


@router.get("/{chat_session}", response_model=Message)
async def get_chat_history(chat_session: str, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_messages_by_session(chat_session)


@router.get("/{user_id}/sessions", response_model=Message)
async def get_all_sessions_by_user_id(user_id: str, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_sessions_by_user_id(user_id)


#Create a new session of chat
@router.post("/chat")
async def create_chat(token: HTTPAuthorizationCredentials = Depends(security),
                      llm_service: LlmService = Depends(get_llm_service),
                      message: MessageDao = Body()):

    return llm_service.create_chat(message, token)
