from fastapi import Depends
from sqlmodel import Session

from db.db import get_session
from llm.mem0.mem0_client import CustomMemoryClient, mem0_config
from services.memory_service import MemoryService


def get_memory_client() -> CustomMemoryClient:
    config = mem0_config
    return CustomMemoryClient(config)

def get_memory_service(memory_client: CustomMemoryClient = Depends(get_memory_client)) -> MemoryService:
    return MemoryService(memory_client)
