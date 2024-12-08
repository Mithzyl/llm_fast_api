from typing import Optional

from pydantic import BaseModel


class MessageDao(BaseModel):
    message: str
    conversation_id: Optional[str]
    model: Optional[str]

    def get_message(self):
        return self.message

    def get_conversation_id(self):
        return self.conversation_id

    def get_model(self):
        return self.model

class ChatCreateParam(MessageDao):
    temperature: float