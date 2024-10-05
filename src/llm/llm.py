from datetime import datetime
from typing import Dict

from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.util import generate_md5_id


def create_first_chat(message: str) -> Dict[str, str]:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.9)
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
            response = {'message_id': message_id,
                        'message': ai_msg,
                        'role': 'assistant',
                        'prompt_token': cb.prompt_tokens,
                        'completion_token': cb.completion_tokens,
                        'total_token': cb.total_tokens,
                        'cost': cb.total_cost
                        }
            return response
    except Exception as e:
        raise e




