
from datetime import datetime

from sqlmodel import SQLModel, Field


class LlmCost(SQLModel, table=True):
    __tablename__ = 'message_cost'

    id: int = Field(default=None, primary_key=True)
    message_id: str = Field(default=None)
    user_id: str = Field(default=None)
    prompt_token: str = Field(default=None)
    completion_token: str = Field(default=None)
    total_token: str = Field(default=None)
    cost: str = Field(default=None)
    create_time: datetime = Field(default=None)