from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self) -> str:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name, prompt_template, parser):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.parser = parser
        self.model = openai.OpenAI(model_name)
