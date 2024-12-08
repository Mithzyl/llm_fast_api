from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple, Any

from openai import OpenAI

from utils.util import generate_md5_id


class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self) -> str:
        pass


class OpenAIProvider:
    def __init__(self, base_url: str, temperature: float = 0.8, api_key: str = None):
        self.completion = None
        self.base_url = base_url
        self.temperature = temperature
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def get_response(self, prompt: List[Any], model: str = 'qwen2:0.5b') -> dict[str, str | int | datetime | None]:

        try:
            self.completion = self.client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=self.temperature,
            )
            usage = self.completion.usage
            message_id = generate_md5_id()

            time = datetime.now()
            response = {'message_id': message_id,
                        'message': self.completion.choices[0].message.content,
                        'role': 'assistant',
                        'prompt_token': usage.prompt_tokens,
                        'completion_token': usage.completion_tokens,
                        'total_token': usage.total_tokens,
                        # 'cost': usage.total_cost,
                        'create_time': time,
                        'model': model
                        }
            return response
        except Exception as e:
            raise e





