from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from passlib.exc import InvalidTokenError

from passlib.hash import bcrypt

from starlette import status

from utils.jwt import encode_jwt, decode_jwt


def authenticate_user(user, password) -> str:
    hashed_password = user.password
    if bcrypt.verify(password, hashed_password):
        # generate jwt
        token_payload = {
            'email': user.email,
            'sub': str(user.id)
        }
        access_token = encode_jwt(token_payload)

        return access_token
    else:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = 'Password incorrect'
        )


def decode_token(token: str) -> dict:
    decode_payload = decode_jwt(token)

    return decode_payload