from fastapi import Depends
from requests import Session

from db.db import get_session
from services.user_service import UserService


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(session)