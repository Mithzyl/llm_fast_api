from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI

from dependencies.memory_dependency import get_memory_client
from routers import user_router, llm_router
from db.db import create_db_and_tables, create_db
from routers.memory_router import memory_router
from utils.util import set_api_key_environ


async def lifespan(app: FastAPI):
    print("Application Startup")
    try:
        set_api_key_environ("./key.json")
        create_db_and_tables()
    except Exception as e:
        db_name = "test.db"
        print(e)
        create_db(db_name)

    yield
    print("Application Shutdown")


app = FastAPI(lifespan=lifespan)

app.include_router(user_router.user_router)
app.include_router(llm_router.llm_router)
app.include_router(memory_router)


