from http.client import HTTPException

from sqlmodel import Session
from starlette.responses import JSONResponse

from llm.mem0.mem0_client import CustomMemoryClient
from models.param.memory_param import MemoryParam
from models.response.messgage_response import Response


class MemoryService:
    session: Session

    def __init__(self, memory_client: CustomMemoryClient):
        self._memory_client = memory_client

    def get_memories_by_user_id(self, user_id: str) -> JSONResponse:
        memories = self._memory_client.get_all_memory_by_user_id(user_id=user_id)

        return JSONResponse(status_code=200, content=memories)

    def get_memory_history_by_memory_id(self, memory_id: str) -> JSONResponse:
        histories = self._memory_client.memory.history(memory_id=memory_id)


        return JSONResponse(status_code=200, content=histories)
    
    def get_memories_by_conversation_id(self, conversation_id: str) -> JSONResponse:
        histories = self._memory_client.get_all_memory_by_conversation_id(conversation_id=conversation_id)


        return JSONResponse(status_code=200, content=histories)

    def get_all_memory(self):
        return self._memory_client.memory.get_all()

    def delete_memory_by_memory_id(self, memory_id: str) -> JSONResponse:
        try:
            messages = self._memory_client.memory.delete(memory_id=memory_id)
            return JSONResponse(status_code=200, content=messages)
        except Exception as e:
            raise HTTPException(e)

    def update_memory_by_memory_id(self, memory_param: MemoryParam, memory_id: str) -> JSONResponse:
        try:
            messages = self._memory_client.memory.update(memory_id=memory_id, data=memory_param.message)
            return JSONResponse(status_code=200, content=messages)
        except Exception as e:
            raise HTTPException(e)


