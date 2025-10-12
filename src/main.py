import os
import threading
from contextlib import asynccontextmanager


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from watchdog.observers import Observer

from config.error_config import http_exception_handler, default_error_handler
from llm.mcp.mcp_tool_manager import MCPToolManager

from routers import user_router, llm_router

from routers.memory_router import memory_router
from utils.scheduler import scheduler_manager
from utils.util import set_api_key_environ, find_root_dir
from utils.watchdog import PersonalDocumentHandler

mcp_server_manager = None

# watchdog global config
watchdog_observer = Observer()
watchdog_watch_path = "./files/personal_docs"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_server_manager
    global watchdog_observer
    global watchdog_watch_path
    print("Application Startup")
    try:
        set_api_key_environ("./key.json")
        find_root_dir()
        # create_db_and_tables()

        # Start watchdog observer
        os.makedirs(watchdog_watch_path, exist_ok=True)
        event_handler = PersonalDocumentHandler()
        watchdog_observer.schedule(event_handler, watchdog_watch_path, recursive=True)
        watch_dog_observer_thread = threading.Thread(target=watchdog_observer.start)
        watch_dog_observer_thread.daemon = True
        watchdog_observer.start()
        print("Watchdog Started")

        # Start APScheduler
        scheduler_manager.start()

    except Exception as e:
        print("Starting application failed, error: ", e)
        raise e

    # Initialize LLM and Tooling Resources
    try:
        print("Loading mcp servers")
        # Configure MCP client
        mcp_server_configs = {

            "agent_tools": {
                "command": "python",
                "args": ["/app/src/llm/mcp/mcp_tools.py"],
                "transport": "stdio",
            }
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

    # Shutdown watchdog
    if watchdog_observer.is_alive():
        watchdog_observer.stop()
        watchdog_observer.join()
    print("Watchdog shutting down.")

    # Shutdown APScheduler
    scheduler_manager.stop()

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
