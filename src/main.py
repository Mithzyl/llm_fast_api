from contextlib import asynccontextmanager

import yaml

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.error_config import http_exception_handler, default_error_handler
from middleware.jwt_middleware import JWTMiddleware
from routers import user_router, llm_router
from db.db import create_db_and_tables, create_db
from routers.memory_router import memory_router
from utils.util import set_api_key_environ, find_root_dir


async def lifespan(app: FastAPI):
    print("Application Startup")
    try:
        set_api_key_environ("./key.json")
        find_root_dir()
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
    allow_origins=[
    "*"
],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# app.add_middleware(JWTMiddleware)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, default_error_handler)

app.include_router(user_router.user_router)
app.include_router(llm_router.llm_router)
app.include_router(memory_router)
