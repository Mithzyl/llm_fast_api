from http.client import HTTPException

from sqlmodel import Session

from llm.mem0.mem0_client import CustomMemoryClient
from models.param.memory_param import MemoryParam
from models.response.messgage_response import Response


class MemoryService:
    session: Session

    def __init__(self, memory_client: CustomMemoryClient):
        self._memory_client = memory_client

    def get_memories_by_user_id(self, user_id: str) -> Response:
        memories = self._memory_client.get_all_memory_by_user_id(user_id=user_id)

        return Response(code="200", message=memories)

    def get_memory_history_by_memory_id(self, memory_id: str) -> Response:
        histories = self._memory_client.memory.history(memory_id=memory_id)


        return Response(code="200", message=histories)
    
    def get_memories_by_conversation_id(self, conversation_id: str) -> Response:
        histories = self._memory_client.get_all_memory_by_conversation_id(conversation_id=conversation_id)


        return Response(code="200", message=histories)

    def get_all_memory(self):
        return self._memory_client.memory.get_all()

    def delete_memory_by_memory_id(self, memory_id: str) -> Response:
        try:
            messages = self._memory_client.memory.delete(memory_id=memory_id)
            return Response(code="200", message=messages)
        except Exception as e:
            raise HTTPException(e)

    def update_memory_by_memory_id(self, memory_param: MemoryParam, memory_id: str):
        try:
            messages = self._memory_client.memory.update(memory_id=memory_id, data=memory_param.message)
            return Response(code="200", message=messages)
        except Exception as e:
            raise HTTPException(e)


