from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import user_router, llm_router
from db.db import create_db_and_tables, create_db
from routers.memory_router import memory_router
from utils.util import set_api_key_environ


async def lifespan(app: FastAPI):
    print("Application Startup")
    try:
        set_api_key_environ("./key.json")
        # create_db_and_tables()
    except Exception as e:
        db_name = "test.db"
        print(e)
        create_db(db_name)

    yield
    print("Application Shutdown")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "*"

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router.user_router)
app.include_router(llm_router.llm_router)
app.include_router(memory_router)


