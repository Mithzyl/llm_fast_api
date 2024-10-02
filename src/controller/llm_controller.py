
from fastapi import FastAPI, APIRouter

from services import llm_service

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)

@router.get("/")
async def demo_response():
    return llm_service.get_response()

# Create
@router.post("/{session}")
async def demo_response(chat_session: str):