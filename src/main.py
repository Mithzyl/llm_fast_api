from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI

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

app.include_router(user_controller.router)
app.include_router(llm_controller.router)


