from dataclasses import dataclass

from pydantic import BaseModel

class UserDTO(BaseModel):
    name: str
    email: str

    class Config:
        from_attributes = True