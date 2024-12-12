from fastapi import FastAPI, APIRouter, Depends, Body, Path
from fastapi.openapi.models import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from redis import Redis
from sqlmodel import Session

from dependencies.llm_dependency import get_llm_service, get_llm_api, get_llm_graph
from fastapiredis.redis_client import get_redis
from routers.user_router import security
from models.param.message_param import ChatCreateParam
from models.response.messgage_response import Response
from services.llm_service import LlmService

llm_router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)

#Create a new session of chat
@llm_router.post("/chat", response_model=Response)
async def create_chat(
        llm_param: ChatCreateParam = Body(),
        token: HTTPAuthorizationCredentials = Depends(security),
        llm_service = Depends(get_llm_service),
        llm_graph = Depends(get_llm_graph),
        redis_client = Depends(get_redis),
) -> Response:
        return llm_service.create_chat(llm_param, token, llm_graph, redis_client)


# Get model list
@llm_router.get("/get_model", response_model=Response)
async def get_models(llm_service: LlmService = Depends(get_llm_service)) -> Response:
    return llm_service.get_model_list()


@llm_router.get("/conversation/{message_id}", response_model=Response)
async def get_message_by_id(message_id: int, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_message_by_message_id(message_id)

@llm_router.get("/{conversation_id}", response_model=Response)
async def get_conversation_history(conversation_id: str, llm_service: LlmService = Depends(get_llm_service),
                                   redis_client: Redis = Depends(get_redis)):
    return llm_service.get_messages_by_conversation_id(conversation_id, redis_client)


@llm_router.get("/{user_id}/sessions", response_model=Response)
async def get_all_sessions_by_user_id(user_id: str, llm_service: LlmService = Depends(get_llm_service)):
    return llm_service.get_sessions_by_user_id(user_id)





