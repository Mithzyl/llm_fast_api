from dataclasses import dataclass

from pydantic import BaseModel

class UserDTO(BaseModel):
    name: str
    userid: str

    class Config:
        from_attributes = True