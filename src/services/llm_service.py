from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from sqlmodel import Session, desc

from llm.llm import create_first_chat
from models.dto.messgage_dto import Message
from models.dto.session_dto import ChatSession
from models.model.llm_message import llm_message, llm_session
from utils.util import generate_md5_id


class LlmService:
    def __init__(self, session: Session):
        self.session = session

    def get_messages_by_session(chat_session: str, session: Session):
        try:
            messages = (
                session.query(llm_message).filter(llm_message.session_id == chat_session)
                .order_by(desc(llm_message.create_time)).all()
            )
        except Exception as e:
            return Message(code="500", message=str(e))

        return Message(code="200", message=messages)


    def get_message_by_message_id(message_id, session):
        message = session.query(llm_message).filter(llm_message.id == message_id).first()

        return Message(code="200", message=message)


    def get_sessions_by_user_id(user_id, session):
        try:
            sessions = (session.query(llm_session).filter(llm_session.user_id == user_id)
                        .order_by(desc(llm_session.update_time)).all())

            session_dto = []
            for session in sessions:
                dto = ChatSession(session_id=session.session_id, title=session.title)
                session_dto.append(dto)

        except Exception as e:
            return Message(code="500", message=str(e))

        return Message(code="200", message=session_dto)


    def create_chat(message, session):
        # 1. generate a session_id
        # 2. Add initial system message of llm
        # 3. create langchain prompt template
        # 4. call LLM API
        # 5. save message and the session

        try:
            create_time = datetime.now()
            session_id = generate_md5_id()
            chat = create_first_chat(message.get_message())



        except Exception as e:
            return Message(code="500", message=str(e))

        return Message(code="200", message=chat)

