from typing import Optional

from pydantic import BaseModel


class MemoryParam(BaseModel):
    message: str