from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import openai
from openai import OpenAI
from openai.types import CompletionUsage


class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self) -> str:
        pass


class OpenAIProvider:
    def __init__(self, base_url: str, temperature: float = 0.8):
        self.base_url = base_url
        self.temperature = temperature
        self.client = OpenAI(base_url=self.base_url, api_key='lm_studio')

    def get_response(self, prompt: List[Any]) -> str:
        self.completion = self.client.chat.completions.create(
            model="model-identifier",
            messages=prompt,
            temperature=self.temperature,
        )

        return self.completion.choices[0].message.content, self.completion.usage

