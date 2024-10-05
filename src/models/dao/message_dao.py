from dataclasses import dataclass

from pydantic import BaseModel


class MessageDao(BaseModel):
    message: str

    def get_message(self):
        return self.message