from fastapi import APIRouter, Depends

from dependencies.memory_dependency import get_memory_service
from services.memory_service import MemoryService

memory_router = APIRouter(
    prefix="/memory",
    tags=["memory"],
)

@memory_router.get("/user/{user_id}")
async def get_memory_by_user_id(user_id: str,
                                memory_service: MemoryService = Depends(get_memory_service)):
    return memory_service.get_memories_by_user_id(user_id)
