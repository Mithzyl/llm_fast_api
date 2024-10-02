
from datetime import datetime

from sqlmodel import SQLModel, Field


class llm_message(SQLModel, table=True):
    id: int = Field(primary_key=True)
    session_id: str = Field()
    user_id: str = Field()
    message: str = Field()
    create_time: datetime = Field()
    update_time: datetime = Field()
    role: str = Field()


class llm_session(SQLModel, table=True):
    id: int = Field(primary_key=True)
    session_id: str = Field()
    user_id: str = Field()
    title: str = Field()
    create_time: datetime = Field()
    update_time: datetime = Field()
