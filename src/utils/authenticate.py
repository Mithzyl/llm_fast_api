from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from passlib.hash import bcrypt

from starlette import status

from utils.jwt import encode_jwt, decode_jwt


def authenticate_user(user, hashed_password) -> str:
    password = user.password
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


def decode_token(token: HTTPAuthorizationCredentials) -> dict:
    credential = token.credentials
    decode_payload = decode_jwt(credential)
    email = decode_payload.get('email', None)
    if email is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = 'Token incorrect'
        )

    return decode_payload