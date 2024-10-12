from datetime import datetime
from typing import Dict, Optional

from fastapi import Depends, Body
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from models.dao.message_dao import MessageDao
from utils.util import generate_md5_id


class LlmApi:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def create_first_chat(self, message: str, model: Optional[str] = None) -> Dict[str, str]:
        model = model if model else self.model
        llm = ChatOpenAI(model=model, temperature=self.temperature, openai_api_base='https://api.deepseek.com')
        # llm = ChatOpenAI(
        #     model='deepseek-chat',
        #     openai_api_key='sk-3e80b5104a2749c5bb5c643aa7ef2d89',
        #     openai_api_base='https://api.deepseek.com',
        #     max_tokens=1024
        # )
        system_template = "You are an assistant that helps with daily questions, english teaching and coding"
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('user', '{message}')
        ])
        parser = StrOutputParser()
        chain = prompt_template | llm | parser
        try:
            with get_openai_callback() as cb:
                message_id = generate_md5_id()
                ai_msg = chain.invoke({"message": message})
                time = datetime.now()
                response = {'message_id': message_id,
                            'message': ai_msg,
                            'role': 'assistant',
                            'prompt_token': cb.prompt_tokens,
                            'completion_token': cb.completion_tokens,
                            'total_token': cb.total_tokens,
                            'cost': cb.total_cost,
                            'create_time': time,
                            'model': model
                            }
                return response
        except Exception as e:
            raise e

    def chat(self, message_history: list, new_message: str, model: Optional[str] = None) -> Dict[str, str]:
        model = self.model or model
        llm = ChatOpenAI(model=model, temperature=self.temperature)

        system_template = "You are an assistant that helps with daily questions, english teaching and coding"
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('placeholder', '{conversation}'),
            ('user', '{message}')
        ])
        parser = StrOutputParser()
        chain = prompt_template | llm | parser

        try:
            with get_openai_callback() as cb:
                message_id = generate_md5_id()
                ai_msg = chain.invoke(
                    {
                        "conversation": [(message.role, message.message) for message in message_history],
                        "message": new_message
                    }
                )
                time = datetime.now()
                response = {'message_id': message_id,
                            'message': ai_msg,
                            'role': 'assistant',
                            'prompt_token': cb.prompt_tokens,
                            'completion_token': cb.completion_tokens,
                            'total_token': cb.total_tokens,
                            'cost': cb.total_cost,
                            'create_time': time,
                            'model': model
                            }
                return response
        except Exception as e:
            raise e


def get_llm_api(model: str = 'gpt-4o-mini-2024-07-18', temperature: float = 0.9) -> LlmApi:
    return LlmApi(model=model, temperature=temperature)
