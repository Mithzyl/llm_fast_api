from sqlmodel import Session

from llm.mem0.mem0_client import CustomMemoryClient
from models.response.messgage_response import Response


class MemoryService:
    session: Session

    def __init__(self, memory_client: CustomMemoryClient):
        self._memory_client = memory_client

    def get_memories_by_user_id(self, user_id: str) -> Response:
        memories = self._memory_client.get_all_memory_by_user_id(user_id=user_id)

        return Response(code="200", message=memories)
