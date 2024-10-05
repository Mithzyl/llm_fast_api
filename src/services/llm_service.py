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
            title = message.get_message()[:len(message.get_message())//2]

            last_message_id = self.session.exec(select(llm_message).order_by(desc(llm_message.id))).first().id
            # message from the user
            user_message_id = last_message_id + 1
            user_message = llm_message(id=user_message_id,
                                       session_id=session_id,
                                       title=title,
                                       message=message.get_message(),
                                       user_id=user.userid,
                                       create_time=create_time,
                                       update_time=update_time,
                                       role='human')

            # message response from the model
            ai_message_id = user_message_id + 1
            ai_message = llm_message(id=ai_message_id,
                                     session_id=session_id,
                                     title=title,
                                     message=chat.get('message'),
                                     user_id=user.userid,
                                     create_time=create_time,
                                     update_time=update_time,
                                     role=chat.get('role'))

            # dialog session
            last_dialog_id = self.session.exec(select(llm_session).order_by(desc(llm_session.id))).first().id
            dialog_id = last_dialog_id + 1
            dialog = llm_session(
                id=dialog_id,
                session_id=session_id,
                title=title,
                user_id=user.userid,
                create_time=create_time,
                update_time=update_time)

            self.add_chat_session(user_message)
            self.add_chat_session(ai_message)
            self.add_chat_session(dialog)

            # TODO: handle regex to restructure output

        except Exception as e:
            raise e
            return Message(code="500", message=e)

        return Message(code="200", message=chat)

    def add_chat_session(self, dialog: llm_session | llm_message):
        try:
            self.session.add(dialog)
            self.session.commit()
            self.session.refresh(dialog)
        except Exception as e:
            self.session.rollback()
            raise Exception(e)


def get_llm_service(session: Session = Depends(get_session)) -> LlmService:
    return LlmService(session)
