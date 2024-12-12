from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field

@dataclass
class llm_message(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    message_id: str = Field(index=True)
    session_id: str = Field(index=True)
    user_id: str = Field(index=True)
    message: str = Field()
    create_time: int = Field()
    update_time: int = Field()
    role: str = Field()
    parent_id: Optional[str] = Field(default=None, index=True)
    children_id: Optional[str] = Field(default=None, index=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> object:
        return llm_message(**data)


@dataclass
class llm_session(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    session_id: str = Field(index=True)
    user_id: str = Field(index=True)
    title: str = Field()
    create_time: int = Field()
    update_time: int = Field()
    model: str = Field()

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> object:
        return llm_message(**data)
