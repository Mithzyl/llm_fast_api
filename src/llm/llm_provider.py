from time import time
from typing import List, Any

from langsmith.wrappers import wrap_openai
from openai import OpenAI, Client

from utils.util import generate_md5_id


class OpenAIProvider:
    def __init__(self, base_url: str, temperature: float = 0.8, api_key: str = None):
        """
        Wrapper that interacts with OpenAI's API
        Args:
            base_url: api url
            temperature: controls the diversity and randomness
            api_key: the api key for different LLM api
        """
        self.completion = None
        self.base_url = base_url
        self.temperature = temperature
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.wrapped_client = wrap_openai(self.client)

    def get_response(self, prompt: List[Any], model: str = 'qwen2:0.5b') -> dict:
        """
        creates a completion interaction
        Args:
            prompt: prompt input
            model: model to call

        Returns:
            dict containing information of this round of interaction

        """

        try:
            self.completion = self.wrapped_client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=self.temperature,
            )
            usage = self.completion.usage
            message_id = generate_md5_id()

            create_time = int(time())
            response = {'message_id': message_id,
                        'message': self.completion.choices[0].message.content,
                        'role': 'assistant',
                        'prompt_token': usage.prompt_tokens,
                        'completion_token': usage.completion_tokens,
                        'total_token': usage.total_tokens,
                        # 'cost': usage.total_cost,
                        'create_time': create_time,
                        'model': model
                        }
            return response
        except Exception as e:
            raise e





