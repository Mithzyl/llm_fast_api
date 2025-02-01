from fastapi import HTTPException
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List

from config.jwt_middleware_config import EXCLUDE_PATHS
from fastapiredis.redis_client import get_custom_redis_client, get_redis
from utils.authenticate import verify_token


class JWTMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.exclude_paths = EXCLUDE_PATHS

    async def dispatch(self, request, call_next):
        # Check if the path should be excluded from JWT verification
        path = request.url.path
        if any(path.startswith(exclude_path) for exclude_path in self.exclude_paths):
            print("not protected path: ", path)
            return await call_next(request)

        try:
            print("protected path: ", path)
            # Verify JWT token for protected routes
            await self.verify_authorization(request, call_next)
            return await call_next(request)
        except HTTPException as e:
            return HTTPException(
                status_code=e.status_code,
                detail=e.detail,
                headers=e.headers
            )

        except Exception as e:
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e
            )

    async def verify_authorization(self, request, call_next,
                                   credential_exception=HTTPException(
                                        status_code=status.HTTP_401_UNAUTHORIZED,
                                        detail='Authorization header missing',
                                        headers={'WWW-Authenticate': 'Bearer'})):
        authorization = request.headers.get('Authorization')
        if not authorization:
            raise credential_exception

        token = authorization.split(' ')[1] if len(authorization.split(' ')) > 1 else None

        try:
            redis_client = get_custom_redis_client().get_client()
            if token:
                verify_token(token, redis_client)

            else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

            response = await call_next(request)

            return response

        except Exception:
            raise

