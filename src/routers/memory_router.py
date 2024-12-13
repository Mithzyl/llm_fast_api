from fastapi import APIRouter, Depends

from dependencies.memory_dependency import get_memory_client
from llm.mem0.mem0_client import CustomMemoryClient

memory_router = APIRouter(
    prefix="/memory",
    tags=["memory"],
)

@memory_router.get("/user/{user_id}")
async def get_memory_by_user_id(user_id,
                                memory_client: CustomMemoryClient = Depends(get_memory_client)):
    return memory_client.get_all_memory_by_user_id(user_id)
