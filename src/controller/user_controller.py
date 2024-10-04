from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.testing.pickleable import User
from sqlmodel import Session
from starlette.responses import JSONResponse

from db.db import get_session
from models.dao.user_dao import UserLogin, UserRegister
from models.dto.messgage_dto import Message
from services import user_service
from services.user_service import UserService, get_user_service

router = APIRouter(
    prefix="/users",
    tags=["user"],
)

security = HTTPBearer()

@router.get("/")
async def get_users(user_service: UserService = Depends(get_user_service)):
    return user_service.get_users()

@router.get("/me")
async def get_me(token: HTTPAuthorizationCredentials = Depends(security), user_service: UserService = Depends(get_user_service)):
    return user_service.get_me(token)

@router.get("/{id}")
async def get_user_by_id(id: int, user_service: UserService = Depends(get_user_service)):
    return user_service.get_user_by_id(id)

@router.post("/login", response_model=Message)
async def login(login_request: UserLogin, user_service: UserService = Depends(get_user_service)):

    return user_service.login(login_request)


@router.post("/register", response_model=Message)
async def register(register_request: UserRegister, user_service: UserService = Depends(get_user_service)):
    return user_service.register(register_request)


