from typing import Any

from fastapi import APIRouter, Depends, Body
from starlette.responses import JSONResponse

from dependencies.memory_dependency import get_memory_service
from models.param.memory_param import MemoryParam
from services import memory_service
from services.memory_service import MemoryService

memory_router = APIRouter(
    prefix="/memory",
    tags=["memory"],
)


@memory_router.get("/conversation/{conversation_id}")
async def get_memory_by_user_id(conversation_id: str,
                                memory_service: MemoryService = Depends(get_memory_service)) -> JSONResponse:
    return memory_service.get_memories_by_conversation_id(conversation_id)

@memory_router.get("/users/{user_id}")
async def get_memory_by_user_id(user_id: str,
                                memory_service: MemoryService = Depends(get_memory_service)) -> JSONResponse:
    return memory_service.get_memories_by_user_id(user_id)


@memory_router.get("/{memory_id}")
async def get_memory_history(memory_id: str,
                             memory_service: MemoryService = Depends(get_memory_service)) -> JSONResponse:
    return memory_service.get_memory_history_by_memory_id(memory_id)


@memory_router.get("/all")
async def get_all_memory(memory_service: MemoryService = Depends(get_memory_service)):
    """
    Admin memory management
    Args:
        memory_service:

    Returns:

    """
    return memory_service.get_all_memory()


@memory_router.delete("/{memory_id}")
async def delete_memory_by_id(memory_id: str,
                              memory_service: MemoryService = Depends(get_memory_service)) -> JSONResponse:
    return memory_service.delete_memory_by_memory_id(memory_id)

@memory_router.put("/{memory_id}")
async def update_memory_by_id(memory_id: str,
                              memory_param: MemoryParam = Body(),
                              memory_service: MemoryService = Depends(get_memory_service)) -> JSONResponse:
    return memory_service.update_memory_by_memory_id(memory_param, memory_id)