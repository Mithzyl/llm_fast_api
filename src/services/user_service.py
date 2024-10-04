import uuid
from typing import List

import jwt
from fastapi import Depends
from passlib.hash import bcrypt
from sqlmodel import Session, desc

from db.db import get_session
from models.dto.messgage_dto import Message
from models.dto.user_dto import UserDTO
from models.model.user import User
from utils.authenticate import authenticate_user, decode_token
from utils.jwt import encode_jwt, decode_jwt
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# temporary testing for bearer

class UserService:
    def __init__(self, session: Session):
        self.session = session

    def get_users(self) -> List[User]:
        return self.session.query(User).all()

    def get_user_by_id(self, id: int) -> User:
        return self.session.query(User).filter(User.id == id).first()

    def login(self, login_request: dict) -> Message:
        # query user, not exist return error
        user = self.session.query(User).filter(User.email == login_request.email).first()
        if not user:
            return Message(code="500", message="no such user")

        try:
            password = login_request.get(["password"])
            # TODO: Save token to redis
            access_token = authenticate_user(user, password)

            return Message(code="200", message=str(access_token))

        except Exception as e:
            return Message(code="200", message=e)

    def register(self, register_request: dict) -> Message:
        # 1. generate uuid
        # 2. encrypt password
        new_id = self.session.query(User).order_by(desc(User.id)).first().id + 1
        new_uuid = str(uuid.uuid4())
        print(register_request)
        role = 'user'
        hash_password = bcrypt.hash(register_request.password)
        new_user = User(new_id, new_uuid, register_request.email, register_request.name, hash_password, role)

        try:
            self.session.add(new_user)
            self.session.commit()

        except Exception as e:
            self.session.rollback()

            return Message(code="500", message=str(e))

        return Message(code="200", message="success")

        # 3. TODO: Auto login after register

    def get_me(self, token: HTTPAuthorizationCredentials) -> Message:
        try:
            decode_payload = decode_token(token)
            email = decode_payload.get('email', None)

            user = self.session.query(User).filter(User.email == email).first()
            dto = UserDTO.from_orm(user)
            # TODO: redirect

            return Message(code="200", message=dto)

        except Exception as e:
            # TODO: redirect
            return Message(code="500", message=str(e))


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(session)


