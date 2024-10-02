from fastapi import APIRouter, Depends
from sqlalchemy.testing.pickleable import User
from sqlmodel import Session
from starlette.responses import JSONResponse

from db.db import get_session
from models.dto.messgage_dto import Message
from models.model.user import UserLogin
from services import user_service

router = APIRouter(
    prefix="/users",
    tags=["user"],
)


@router.get("/")
async def get_users(session: Session = Depends(get_session)):
    return user_service.get_users(session)

@router.get("/{id}")
async def get_user_by_id(id: int, session: Session = Depends(get_session)):
    return user_service.get_user_by_id(id, session)

@router.post("/login", response_model=Message)
async def login(login_request: UserLogin, session: Session = Depends(get_session)):
    #TODO: Encrypt passwords
    return user_service.login(login_request, session)

