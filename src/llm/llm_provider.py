from time import time
from typing import List, Any

from humanfriendly.terminal import message
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.utils.config import ensure_config, get_callback_manager_for_config
from langsmith.wrappers import wrap_openai
from openai import OpenAI, Client, AsyncOpenAI

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
        self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self.wrapped_client = wrap_openai(self.async_client)

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

    async def stream_response(self, prompt: List[Any]):
        config = ensure_config(None | {"tags": ['agent_llm']})
        callback_manager = get_callback_manager_for_config(config)
        llm_run_manager = callback_manager.on_chat_model_start({}, [])[0]
        self.completion = await self.async_client.chat.completions.create(
            model='gpt-4o-mini-2024-07-18',
            messages=prompt,
            temperature=self.temperature,
            stream=True,
        )

        response_content = ""

        async for chunk in self.completion:
            delta = chunk.choices[0].delta
            print(delta)

            if delta.content:
                response_content += delta.content
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=delta.content)
                )

                llm_run_manager.on_llm_new_token(delta.content, chunk=chunk)


        return response_content




