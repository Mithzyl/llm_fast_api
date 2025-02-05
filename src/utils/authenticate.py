from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from passlib.exc import InvalidTokenError

from passlib.hash import bcrypt
from redis import Redis

from starlette import status

from fastapiredis.redis_client import RedisClient
from utils.jwt import encode_jwt, decode_jwt


def authenticate_user(user, password) -> str:
    hashed_password = user.password
    if bcrypt.verify(password, hashed_password):
        # generate jwt
        token_payload = {
            'email': user.email,
            'sub': str(user.userid)
        }
        access_token = encode_jwt(token_payload)

        return access_token
    else:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = 'Password incorrect'
        )


def verify_token(token: str, redis_client: Redis) -> dict:
    try:
        decode_payload = decode_jwt(token)

        userid = decode_payload['sub']
        redis_key = f"auth:{userid}"
        token_status = redis_client.get(redis_key)
        
        if not token_status:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is no longer valid")

        return decode_payload

    except HTTPException:
        raise

    except Exception as e:
        raise e
