from contextlib import asynccontextmanager

import yaml

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient

from config.error_config import http_exception_handler, default_error_handler
from llm.mcp.mcp_tool_manager import MCPToolManager
from middleware.jwt_middleware import JWTMiddleware
from routers import user_router, llm_router
from db.db import create_db_and_tables, create_db
from routers.memory_router import memory_router
from utils.util import set_api_key_environ, find_root_dir

mcp_server_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_server_manager
    print("Application Startup")
    try:
        set_api_key_environ("./key.json")
        find_root_dir()
        # create_db_and_tables()
    except Exception as e:
        db_name = "test.db"
        print(f"DB setup error: {e}, creating {db_name}")
        create_db(db_name)

    # Initialize LLM and Tooling Resources
    try:
        print("Loading mcp servers")
        # Configure MCP client
        mcp_server_configs = {
            # "internal_tools": {
            #     "command": "python",  # <-- Use absolute path,
            #     "args": ["/Users/mith/Desktop/project/llm_fast_api/src/llm/mcp/math.py"],
            #     "transport": "stdio",
            # },
            "agent_tools": {
                "command": "python",
                "args": ["/Users/mith/Desktop/project/llm_fast_api/src/llm/mcp/mcp_tools.py"],
                "transport": "stdio",
            },
            # "external_crm_tools": {
            #     "transport": "streamable_http",
            #     "url": "https://mcp.some-provider.com/v1/",  # <-- Replace with real URL
            #     "headers": {"Authorization": "Bearer YOUR_EXTERNAL_API_KEY"}  # <-- Replace with real key
            # }
        }
        mcp_server_manager = MCPToolManager(mcp_server_configs)
        await mcp_server_manager.load()
    except Exception as e:
        print(e)
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


app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, default_error_handler)

app.include_router(user_router.user_router)
app.include_router(llm_router.llm_router)
app.include_router(memory_router)
