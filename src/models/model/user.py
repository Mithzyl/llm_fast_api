from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import DateTime
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    __tablename__ = "User"

    id: int = Field(default=None, primary_key=True)
    userid: str = Field(default=None, nullable=False)
    email: str = Field(max_length=50)
    name: str = Field(max_length=50)
    password: str = Field(max_length=50)
    role: str = Field(max_length=50)
    create_time: datetime = Field(nullable=False)
    update_time: datetime = Field(nullable=False)

    def __init__(self, id: int, userid: str, email: str, name: str, password: str, role: str):
        super().__init__()
        self.id = id
        self.userid = userid
        self.email = email
        self.name = name
        self.password = password
        self.role = role
        self.create_time = datetime.now()
        self.update_time = datetime.now()



