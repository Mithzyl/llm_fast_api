"""Services module."""

from redis.asyncio import ConnectionPool
from redis import Redis

from fastapiredis.redis import init_redis_pool


class RedisClient:
    def __init__(self, redis_pool: ConnectionPool) -> None:
        self.redis = Redis(connection_pool=redis_pool)

    async def process(self) -> str:
        """
        An example method of redis
        """
        await self.redis.set("my-key", "value")
        return await self.redis.get("my-key")


async def get_redis() -> RedisClient:
    redis_pool = await init_redis_pool()

    return RedisClient(redis_pool)