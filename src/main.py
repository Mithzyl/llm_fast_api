from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controller import user_controller, llm_controller
from db.db import create_db_and_tables, create_db
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods like GET, POST, etc.
    allow_headers=["*"],  # Allows all headers
)

app.include_router(user_controller.user_router)
app.include_router(llm_controller.llm_router)


