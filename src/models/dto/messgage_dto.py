from dataclasses import dataclass
from typing import List, Any


@dataclass
class Message:
    code: str
    message: List[dict | str] | Any
