from dataclasses import dataclass
from datetime import datetime
from typing import List

from pydantic import BaseModel


@dataclass
class ChatSessionDetail:
    """
    Chat history of a session
    """
    message_id: str
    message: str
    time: datetime
    role: str



class ChatSession(BaseModel):
    session_id: str
    title: str
    # content: List[ChatSessionDetail]
