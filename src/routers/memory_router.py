from typing import Any

from fastapi import APIRouter, Depends

from dependencies.memory_dependency import get_memory_service
from models.response.messgage_response import Response
from services import memory_service
from services.memory_service import MemoryService

memory_router = APIRouter(
    prefix="/memory",
    tags=["memory"],
)

@memory_router.get("/users/{user_id}", response_model=Response)
async def get_memory_by_user_id(user_id: str,
                                memory_service: MemoryService = Depends(get_memory_service)) -> Response:
    return memory_service.get_memories_by_user_id(user_id)

@memory_router.get("{/memory_id}", response_model=Response)
async def get_memory_history(memory_id: str,
                             memory_service: MemoryService = Depends(get_memory_service)) -> Response:
    return memory_service.get_memory_history_by_memory_id(memory_id)


@memory_router.get("/all", response_model=Response)
async def get_all_memory(memory_service: MemoryService = Depends(get_memory_service)):
    """
    Admin memory management
    Args:
        memory_service:

    Returns:

    """
    return memory_service.get_all_memory()


@memory_router.delete("/{memory_id}", response_model=Response)
async def delete_memory_by_id(memory_id: str,
                              memory_service: MemoryService = Depends(get_memory_service)) -> Response:
    return memory_service.delete_memory_by_memory_id(memory_id)