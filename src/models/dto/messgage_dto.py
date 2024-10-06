from dataclasses import dataclass
from typing import List, Any


@dataclass
class Response:
    code: str
    message: List[dict | str] | Any

    def get_message(self) -> List | Any:
        return self.message

    def get_code(self) -> str:
        return self.code



