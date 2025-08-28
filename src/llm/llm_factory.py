import json
import os
from os import environ
from typing import Optional

from langchain_openai import ChatOpenAI

def get_llm_provider(
    model: str,
    temperature: float = 0.7,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_config_file: Optional[str] = '/src/model_url_config.json',
) -> ChatOpenAI:
    """
    Factory function to create and configure a ChatOpenAI provider.
    """
    with open(environ["ROOT_DIR"] + model_config_file, 'r') as f:
        model_list = json.load(f)



    if model_list:
        for key in model_list.keys():
            if key in model:
                llm_base_url = model_list[key]['base_url']
                api_key = model_list[key]['api_key']
                break
            else:
                llm_base_url = model_list['llama']['base_url']
                api_key = model_list['llama']['api_key']


    return ChatOpenAI(
        base_url=llm_base_url,
        api_key=api_key,
        model=model,
        temperature=temperature
    )
