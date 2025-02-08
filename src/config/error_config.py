from fastapi import HTTPException
from starlette.responses import JSONResponse


async def http_exception_handler(request, exception: HTTPException) -> JSONResponse:
    """
    handles HTTP exceptions raised due to services such as authentication, API error
    Args:
        request:
        exception:

    Returns:

    """
    return JSONResponse(
        status_code=exception.status_code,
        content={"message: ": exception.detail}
    )

async def default_error_handler(request, exception):
    """
    handles common error such as code, implementation logic
    Args:
        request:
        exception:

    Returns:

    """
    return JSONResponse(
        status_code=500,
        content={"message": str(exception)}
    )


