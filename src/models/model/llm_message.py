
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class llm_message(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    message_id: str = Field(index=True)
    session_id: str = Field(index=True)
    user_id: str = Field(index=True)
    message: str = Field()
    create_time: datetime = Field()
    update_time: datetime = Field()
    role: str = Field()
    parent_id: Optional[str] = Field(default=None, index=True)
    children_id: Optional[str] = Field(default=None, index=True)


class llm_session(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    session_id: str = Field(index=True)
    user_id: str = Field(index=True)
    title: str = Field()
    create_time: datetime = Field()
    update_time: datetime = Field()
    model: str = Field()
