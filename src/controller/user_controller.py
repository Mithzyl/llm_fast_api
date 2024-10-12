from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.testing.pickleable import User
from sqlmodel import Session
from starlette.responses import JSONResponse

from models.dao.user_dao import UserLogin, UserRegister
from models.dto.messgage_dto import Response
from services import user_service
from services.user_service import UserService, get_user_service

user_router = APIRouter(
    prefix="/users",
    tags=["user"],
)

security = HTTPBearer()


@user_router.get("/")
async def get_users(user_service: UserService = Depends(get_user_service)):
    return user_service.get_users()


@user_router.get("/me")
async def get_me(token: HTTPAuthorizationCredentials = Depends(security),
                 user_service: UserService = Depends(get_user_service)):
    return user_service.get_me(token)


@user_router.post("/login", response_model=Response)
async def login(login_request: UserLogin, user_service: UserService = Depends(get_user_service)):
    return user_service.login(login_request)


@user_router.post("/register", response_model=Response)
async def register(register_request: UserRegister, user_service: UserService = Depends(get_user_service)):
    return user_service.register(register_request)


@user_router.get("/{id}")
async def get_user_by_id(id: int, user_service: UserService = Depends(get_user_service)):
    return user_service.get_user_by_id(id)
