from dataclasses import dataclass

@dataclass
class Message:
    code: str
    message: dict | str