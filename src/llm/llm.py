from datetime import datetime
from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


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
        ai_msg = chain.invoke({"message": message})
    except Exception as e:
        assert e
        return str(e)

    response = {'message': ai_msg,
                'role': 'assistant'
                }

    return response
