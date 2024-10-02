from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    code: str
    message: List[dict | str]