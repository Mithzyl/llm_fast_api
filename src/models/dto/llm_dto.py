from dataclasses import dataclass
from typing import List


@dataclass
class LlmDto:
    content: List | object
    conversation_id: str
    model: str


@dataclass
class Message:
    """
    Message to provide to the llm
    @ role: the role creating the message
    @ message: the message
    """
    message: str
    role: str

    @property
    def message(self) -> str:
        return self.message

    @property
    def role(self) -> str:
        return self.role