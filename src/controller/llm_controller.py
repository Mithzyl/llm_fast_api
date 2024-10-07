from fastapi import FastAPI, APIRouter, Depends, Body, Path
from fastapi.openapi.models import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from sqlmodel import Session

from controller.user_controller import security
from models.dao.message_dao import MessageDao
from models.dto.messgage_dto import Response
from services import llm_service
from services.llm_service import LlmService, get_llm_service
from services.user_service import UserService, get_user_service

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)



@router.get("/conversation/{message_id}", response_model=Response)
async def demo_conversation(message_id: int, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_message_by_message_id(message_id)


@router.get("/{conversation_id}", response_model=Response)
async def get_conversation_history(conversation_id: str, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_messages_by_conversation_id(conversation_id)


@router.get("/{user_id}/sessions", response_model=Response)
async def get_all_sessions_by_user_id(user_id: str, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_sessions_by_user_id(user_id)


#Create a new session of chat
@router.post("/chat", response_model=Response)
async def create_chat(token: HTTPAuthorizationCredentials = Depends(security),
                      llm_service: LlmService = Depends(get_llm_service),
                      message: MessageDao = Body()) -> Response:

    return llm_service.create_chat(message, token)

# Get model list
@router.get("/get_model", response_model=Response)
async def get_models() -> Response:
    print('called get_models')
    return Response(code="200", message="ok")
