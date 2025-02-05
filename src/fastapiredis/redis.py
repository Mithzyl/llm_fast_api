from typing import AsyncIterator

import redis


def init_redis_pool():
    # session = from_url(f"redis://{host}", password=password, encoding="utf-8", decode_responses=True)
    session = redis.ConnectionPool(host='localhost',
                                   port=6379,
                                   encoding="utf-8",
                                   decode_responses=True)

    return session
    # yield session
    # session.close()
    # await session.wait_closed()
