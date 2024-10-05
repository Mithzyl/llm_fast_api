from datetime import datetime

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials
from langchain_community.chat_models import ChatOpenAI
from sqlmodel import Session, desc, select

from db.db import get_session
from llm.llm import create_first_chat
from models.dao.message_dao import MessageDao
from models.dto.messgage_dto import Message
from models.dto.session_dto import ChatSession
from models.model.llm_message import llm_message, llm_session
from models.model.user import User
from utils.authenticate import decode_token
from utils.util import generate_md5_id


class LlmService:
    def __init__(self, session: Session):
        self.session = session

    def get_messages_by_session(self, chat_session: str):
        try:
            messages = (
                self.session.query(llm_message).filter(llm_message.session_id == chat_session)
                .order_by(desc(llm_message.create_time)).all()
            )
        except Exception as e:
            return Message(code="500", message=str(e))

        return Message(code="200", message=messages)


    def get_message_by_message_id(self, message_id, session):
        message = session.query(llm_message).filter(llm_message.id == message_id).first()

        return Message(code="200", message=message)


    def get_sessions_by_user_id(self, user_id):
        try:
            sessions = (self.session.query(llm_session).filter(llm_session.user_id == user_id)
                        .order_by(desc(llm_session.update_time)).all())

            session_dto = []
            for session in sessions:
                dto = ChatSession(session_id=session.session_id, title=session.title)
                session_dto.append(dto)

        except Exception as e:
            return Message(code="500", message=str(e))

        return Message(code="200", message=session_dto)


    def create_chat(self, message: MessageDao, token: HTTPAuthorizationCredentials):
        # 1. generate a session_id
        # 2. Add initial system message of llm
        # 3. create langchain prompt template
        # 4. call LLM API
        # 5. save message and the session

        try:
            # get current user TODO: get token from redis
            payload = decode_token(token)
            email = payload.get("email")
            user = self.session.exec(select(User).where(User.email == email)).first()
            create_time = datetime.now()

            session_id = generate_md5_id()
            chat = create_first_chat(message.get_message())
            update_time = datetime.now()
            title = chat[:10]  #

            session = llm_session(session_id=session_id, title=title, user_id=user.id, create_time=create_time, update_time=update_time)

            try:
                self.session.add(session)
                self.session.commit()
                self.session.refresh(session)
            except Exception as e:
                raise Exception(e)

            # TODO: handle regex to restructure output

        except Exception as e:
            return Message(code="500", message=str(e))

        return Message(code="200", message=chat)


def get_llm_service(session: Session = Depends(get_session)) -> LlmService:
    return LlmService(session)
