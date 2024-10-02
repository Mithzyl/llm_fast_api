from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class MessageDao(BaseModel):
    message: str

    def get_message(self):
        return self.message