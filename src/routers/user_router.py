from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from sqlalchemy.testing.pickleable import User
from sqlmodel import Session
from starlette.responses import JSONResponse

from dependencies.user_dependency import get_user_service
from models.param.user_param import UserLogin, UserRegister
from models.response.messgage_response import Response
from services import user_service
from services.user_service import UserService

user_router = APIRouter(
    prefix="/users",
    tags=["user"],
)

security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


@user_router.get("/")
async def get_users(user_service: UserService = Depends(get_user_service)):
    return user_service.get_users()


@user_router.get("/me")
async def get_me(token: str = Depends(oauth2_scheme),
                 user_service: UserService = Depends(get_user_service)):
    return user_service.get_me(token)


@user_router.post("/login", response_model=Response)
async def login(login_request: UserLogin,
                user_service: UserService = Depends(get_user_service)) -> str:
    return user_service.login(login_request)


@user_router.post("/register", response_model=Response)
async def register(register_request: UserRegister, user_service: UserService = Depends(get_user_service)):
    return user_service.register(register_request)


@user_router.get("/{id}")
async def get_user_by_id(id: int, user_service: UserService = Depends(get_user_service)):
    return user_service.get_user_by_id(id)
