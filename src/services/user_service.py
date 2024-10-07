import uuid
from typing import List

import jwt
from fastapi import Depends
from passlib.hash import bcrypt
from sqlmodel import Session, desc, select

from db.db import get_session
from models.dao.user_dao import UserRegister, UserLogin
from models.dto.messgage_dto import Response
from models.dto.user_dto import UserDTO
from models.model.user import User
from utils.authenticate import authenticate_user, decode_token
from utils.jwt import encode_jwt, decode_jwt
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# temporary testing for bearer

class UserService:
    def __init__(self, session: Session):
        self.session = session

    def get_users(self) -> Response:
        users = self.session.exec(select(User)).all()
        return Response(code="200", message=users)

    def get_user_by_id(self, id: int) -> User:
        return self.session.exec(select(User).where(User.id == id)).first()

    def login(self, login_request: UserLogin) -> Response:
        # query user, not exist return error
        user = self.session.exec(select(User).where(User.email == login_request.email)).first()
        print(user)
        if not user:
            return Response(code="500", message="no such user")

        try:
            password = login_request.password
            # TODO: Save token to redis
            access_token = authenticate_user(user, password)

            return Response(code="200", message=str(access_token))

        except Exception as e:
            return Response(code="200", message=e)

    def register(self, register_request: UserRegister) -> Response:
        password = register_request.password
        name = register_request.name
        if password == '':
            return Response(code="200", message="password can not be empty")

        email = register_request.email
        exist_user = self.session.exec(select(User).where(User.email == email)).first()
        if not exist_user:
            # 1. generate uuid
            # 2. encrypt password
            new_id = self.session.exec(select(User).order_by(desc(User.id))).first().id + 1
            new_uuid = str(uuid.uuid4())
            print(register_request)
            role = 'user'
            hash_password = bcrypt.hash(password)
            new_user = User(new_id, new_uuid, email, name, hash_password, role)

            try:
                self.session.add(new_user)
                self.session.commit()
                self.session.refresh(new_user)

            except Exception as e:
                self.session.rollback()

                return Response(code="500", message=str(e))

            return Response(code="200", message="success")
        else:
            return Response(code="200", message="user already exist")

        # 3. TODO: Auto login after register

    def get_me(self, token: HTTPAuthorizationCredentials) -> Response:
        try:
            decode_payload = decode_token(token)
            email = decode_payload.get('email', None)

            user = self.session.exec(User.where(User.email == email)).first()
            dto = UserDTO.model_validate(user)
            # TODO: redirect

            return Response(code="200", message=dto)

        except Exception as e:
            # TODO: redirect
            return Response(code="500", message=str(e))


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(session)


