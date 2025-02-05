from typing import Annotated

from fastapi import Depends
from requests import Session

from db.db import get_session
from models.model.user import User
from services.user_service import UserService


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(session)

# async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)],
#                            user_service: UserService = Depends(get_user_service)) -> User:
#     return user_service.get_me(token)