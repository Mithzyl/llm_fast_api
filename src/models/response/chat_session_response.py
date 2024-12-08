from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class ChatSessionDetail:
    """
    Chat history of a session
    """
    message_id: str
    message: str
    time: datetime
    role: str


@dataclass
class ChatSession:
    session_id: str
    title: str
    # content: List[ChatSessionDetail]
