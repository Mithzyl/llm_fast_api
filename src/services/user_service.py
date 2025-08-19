import uuid
from typing import List

import jwt
from fastapi import Depends, HTTPException, status
from jwt import InvalidTokenError
from passlib.hash import bcrypt
from redis import Redis
from requests import HTTPError
from sqlmodel import Session, desc, select
from starlette.responses import JSONResponse

from config.jwt_config import ACCESS_TOKEN_EXPIRE_MINUTES
from db.db import get_session
from fastapiredis.redis_client import RedisClient
from models.param.user_param import UserRegister, UserLogin
from models.response.user_response import UserDTO
from models.model.llm_model import LlmModel
from models.model.user import User
from utils.authenticate import authenticate_user, verify_token
from utils.jwt import encode_jwt, decode_jwt
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# temporary testing for bearer

class UserService:
    def __init__(self, session: Session):
        self.session = session

    def get_users(self) -> JSONResponse:
        users = self.session.exec(select(User)).all()
        return JSONResponse(status_code=200, content=users)

    def get_user_by_id(self, id: int) -> User:
        return self.session.exec(select(User).where(User.id == id)).first()

    def login(self, login_request: UserLogin, redis_client: RedisClient) -> JSONResponse:
        # query user, not exist return error
        user = self.session.exec(select(User).where(User.email == login_request.email)).first()
        # print(user)
        if not user:
            return JSONResponse(status_code=401, content="no such user")

        try:
            password = login_request.password

            access_token = authenticate_user(user, password)

            # set redis key for the user
            # TODO: set device type, web, mobile, etc.
            redis_key = f"auth:{user.userid}"

            redis_client.get_client().set(redis_key, access_token, ex=ACCESS_TOKEN_EXPIRE_MINUTES)

            return JSONResponse(status_code=200, content=str(access_token))

        except Exception as e:
            raise e

    def register(self, register_request: UserRegister) -> JSONResponse:
        password = register_request.password
        name = register_request.name
        if password == '':
            return JSONResponse(status_code=401, content="password can not be empty")

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

                return JSONResponse(status_code=500, content=str(e))

            return JSONResponse(status_code=200, content="success")
        else:
            return JSONResponse(status_code=401, content="user already exists")

        # 3. TODO: Auto login after register

    def get_me(self, token: str, redis_client: Redis) -> JSONResponse:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            decode_payload = verify_token(token, redis_client)

            email = decode_payload.get('email', None)

            if not email:
                raise credentials_exception

            user = self.session.exec(select(User).where(User.email == email)).first()

            if user is None:
                raise credentials_exception
            user_response = UserDTO.model_validate(user)
            # TODO: redirect

            return JSONResponse(status_code=200, content=user_response.model_dump())

        except InvalidTokenError as e:
            # TODO: redirect
            raise credentials_exception

    def get_models(self) -> JSONResponse:
        models = self.session.exec(select(LlmModel)).all()
        print(models)
        return JSONResponse(status_code=200, content=models)






