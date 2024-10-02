from fastapi import FastAPI, APIRouter, Depends, Body, Path
from sqlmodel import Session

from db.db import get_session
from models.dao.message_dao import MessageDao
from models.dto.messgage_dto import Message
from services import llm_service

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)


# @router.get("/")
# async def demo_response():
#     return llm_service.get_response()


@router.get("/conversation/{message_id}", response_model=Message)
async def demo_conversation(message_id: int, session: Session = Depends(get_session)):
    return llm_service.get_message_by_message_id(message_id, session)


@router.get("/{chat_session}", response_model=Message)
async def get_chat_history(chat_session: str, session: Session = Depends(get_session)):
    return llm_service.get_messages_by_session(chat_session, session)


@router.get("/{user_id}/sessions", response_model=Message)
async def get_all_sessions_by_user_id(user_id: str, session: Session = Depends(get_session)):
    return llm_service.get_sessions_by_user_id(user_id, session)


#Create a new session of chat
@router.post("/chat")
async def create_chat(message: MessageDao, session: Session = Depends(get_session, use_cache=False)):
    print(message)
    return llm_service.create_chat(message, session)
