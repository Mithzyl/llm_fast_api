"""Services module."""
import json
from typing import List

from redis.asyncio import ConnectionPool
from redis import Redis

from fastapiredis.redis import init_redis_pool
from models.model.llm_message import llm_message


class RedisClient:
    def __init__(self, redis_pool: ConnectionPool) -> None:
        self.redis = Redis(connection_pool=redis_pool)

    async def process(self) -> str:
        """
        An example method of redis
        """
        await self.redis.set("my-key", "value")
        return await self.redis.get("my-key")

    def get_conversation_history(self, conversation_id: str) -> List[llm_message]:
        str_messages = self.redis.get(conversation_id)
        if not str_messages or str_messages == "[]":
            return None
        else:
            json_messages = json.loads(str_messages)
            messages = [llm_message.from_dict(message) for message in json_messages]
            return messages

    def set_conversation_by_conversation_id(self, conversation_id: str, messages: List[llm_message]) -> None:
        json_messages = [message.to_dict() for message in messages]
        json_messages = json.dumps(json_messages)
        self.redis.set(conversation_id, json_messages)



async def get_redis() -> RedisClient:
    redis_pool = await init_redis_pool()

    return RedisClient(redis_pool)