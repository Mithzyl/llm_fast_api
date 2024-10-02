from langchain_community.chat_models import ChatOpenAI

from models.dto.messgage_dto import Message


def get_response():
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.9)
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    try:

        ai_msg = llm.invoke(messages)
    except Exception as e:
        return Message(code="500", message=str(e))

    return Message(code="200", message=ai_msg.content)
