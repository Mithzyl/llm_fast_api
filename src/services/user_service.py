import uuid
from typing import List

import jwt
from fastapi import Depends, HTTPException, status
from jwt import InvalidTokenError
from passlib.hash import bcrypt
from requests import HTTPError
from sqlmodel import Session, desc, select

from db.db import get_session
from models.param.user_param import UserRegister, UserLogin
from models.response.messgage_response import Response
from models.response.user_response import UserDTO
from models.model.llm_model import LlmModel
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
        # print(user)
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

    def get_me(self, token: str) -> Response:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            decode_payload = decode_token(token)
            email = decode_payload.get('email', None)

            if not email:
                raise credentials_exception

            user = self.session.exec(select(User).where(User.email == email)).first()

            if user is None:
                raise credentials_exception
            user_response = UserDTO.model_validate(user)
            # TODO: redirect

            return Response(code="200", message=user_response)

        except InvalidTokenError as e:
            # TODO: redirect
            raise credentials_exception
            # return Response(code="500", message=str(e))

    def get_models(self) -> Response:
        models = self.session.exec(select(LlmModel)).all()
        print(models)
        return Response(code="200", message=models)






