from pydantic import BaseModel
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    # __tablename__ = "User"

    id: int = Field(default=None, primary_key=True)
    email: str = Field(max_length=50)
    name: str = Field(max_length=50)
    password: str = Field(max_length=50)
    role : str = Field(max_length=50)


class UserLogin(BaseModel):
    email: str = Field(max_length=50)
    password: str = Field(max_length=50)