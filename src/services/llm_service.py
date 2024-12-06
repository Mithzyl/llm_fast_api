from datetime import datetime
from typing import Any, Dict

from fastapi import Depends, HTTPException, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from langchain_community.chat_models import ChatOpenAI
from sqlmodel import Session, desc, select

from db.db import get_session
from llm.llm_api import LlmApi, get_llm_api
from models.dao.message_dao import MessageDao
from models.dto.llm_dto import LlmDto
from models.dto.messgage_dto import Response
from models.dto.session_dto import ChatSession
from models.model.llm_cost import LlmCost
from models.model.llm_message import llm_message, llm_session
from models.model.llm_model import LlmModel
from models.model.user import User
from utils.authenticate import decode_token
from utils.util import generate_md5_id


class LlmService:
    def __init__(self, session: Session, llm: LlmApi):
        self.llm = llm
        self.session = session

    def get_messages_by_conversation_id(self, conversation_id: str) -> Response:
        try:
            messages = (
                self.session.query(llm_message).filter(llm_message.session_id == conversation_id)
                .order_by(llm_message.create_time).all()
            )
        except Exception as e:
            return Response(code="500", message=str(e))

        return Response(code="200", message=messages)

    def get_message_by_message_id(self, message_id, session) -> Response:
        message = session.query(llm_message).filter(llm_message.id == message_id).first()

        return Response(code="200", message=message)

    def get_sessions_by_user_id(self, user_id) -> Response:
        try:
            sessions = (self.session.query(llm_session).filter(llm_session.user_id == user_id)
                        .order_by(desc(llm_session.update_time)).all())

            session_dto = []
            for session in sessions:
                dto = ChatSession(session_id=session.session_id, title=session.title)
                session_dto.append(dto)

        except Exception as e:
            return Response(code="500", message=str(e))

        return Response(code="200", message=session_dto)

    def create_chat(self, message: MessageDao, token: HTTPAuthorizationCredentials) -> Response:
        # 1. generate a session_id
        # 2. Add initial system message of llm
        # 3. create langchain prompt template
        # 4. call LLM API
        # 5. save message and the session

        payload = decode_token(token)
        email = payload.get("email")
        user = self.session.exec(
            select(User).where(User.email == email)).first()  # get current user TODO: get token from redis
        create_time = datetime.now()
        if message.get_model():
            model = message.get_model()


        if message.get_conversation_id():
            conversation_id = message.get_conversation_id()
            conversation = self.session.exec(select(llm_session)
                                             .where(llm_session.session_id == conversation_id)).first()
            if not conversation:
                raise HTTPException(status_code=404, detail="Resource not found")

            # model = conversation.model

            messages = self.get_messages_by_conversation_id(conversation_id).get_message()
            latest_message = self.session.exec(select(llm_message)
                                               .where(llm_message.session_id == conversation_id)
                                               .order_by(desc(llm_message.create_time))).first()

            new_message = message.get_message()

            last_message_primary_id = self.session.exec(select(llm_message).order_by(desc(llm_message.id))).first().id
            # message from the user
            update_time = datetime.now()
            user_message_primary_id = last_message_primary_id + 1
            user_message_id = generate_md5_id()
            user_message = llm_message(id=user_message_primary_id,
                                       message_id=user_message_id,
                                       session_id=conversation_id,
                                       message=message.get_message(),
                                       user_id=user.userid,
                                       create_time=create_time,
                                       update_time=update_time,
                                       role='user',
                                       parent_id=latest_message.message_id,
                                       children_id='')

            latest_message.children_id = user_message_id
            chat = self.llm.chat(messages, new_message, model)
            chat_id = chat.get('message_id')
            user_message.children_id = chat_id

            ai_primary_id = user_message.id + 1
            ai_message = llm_message(id=ai_primary_id,
                                     message_id=chat_id,
                                     session_id=conversation_id,
                                     message=chat.get('message'),
                                     user_id=user.userid,
                                     create_time=chat.get('create_time'),
                                     update_time=chat.get('create_time'),
                                     role='assistant',
                                     parent_id=user_message_id,
                                     children_id='')

            cost = self.cal_cost(chat, user.userid)

            conversation.update_time = chat.get('create_time')

            self.add_chat_session(conversation)
            self.add_chat_session(latest_message)
            self.add_chat_session(user_message)
            self.add_chat_session(ai_message)
            self.add_chat_session(cost)

        else:
            try:  # create new conversation
                conversation_id = generate_md5_id()
                update_time = datetime.now()
                title = message.get_message()[:len(message.get_message()) // 2]

                last_message_primary_id = self.session.exec(
                    select(llm_message).order_by(desc(llm_message.id))).first().id
                # message from the user
                user_message_primary_id = last_message_primary_id + 1
                user_message_id = generate_md5_id()
                user_message = llm_message(id=user_message_primary_id,
                                           message_id=user_message_id,
                                           session_id=conversation_id,
                                           message=message.get_message(),
                                           user_id=user.userid,
                                           create_time=create_time,
                                           update_time=update_time,
                                           role='human',
                                           parent_id='',
                                           children_id='')

                chat = self.llm.create_first_chat(message.get_message(), model)
                chat_id = chat.get('message_id')
                user_message.children_id = chat_id

                # message response from the model
                ai_message_id = user_message_primary_id + 1
                ai_message = llm_message(id=ai_message_id,
                                         message_id=chat.get('message_id'),
                                         session_id=conversation_id,
                                         title=title,
                                         message=chat.get('message'),
                                         user_id=user.userid,
                                         create_time=chat.get('create_time'),
                                         update_time=chat.get('create_time'),
                                         role=chat.get('role'),
                                         parent_id=user_message_id,
                                         children_id='')

                # conversation session
                last_conversation_id = self.session.exec(select(llm_session).order_by(desc(llm_session.id))).first().id
                conversation_primary_id = last_conversation_id + 1
                conversation = llm_session(
                    id=conversation_primary_id,
                    session_id=conversation_id,
                    title=title,
                    user_id=user.userid,
                    create_time=create_time,
                    update_time=update_time,
                    model=chat.get('model'))

                # cost
                message_cost = self.cal_cost(chat, user.userid)

                self.add_chat_session(user_message)
                self.add_chat_session(ai_message)
                self.add_chat_session(conversation)
                self.add_chat_session(message_cost)

            # TODO: handle regex to restructure output

            except Exception as e:
                return Response(code="500", message=e)

        conversation_response = LlmDto(conversation_id=conversation_id, content=chat, model=model)
        return Response(code="200", message=conversation_response)

    def post_message(self, message: MessageDao, token: HTTPAuthorizationCredentials):
        # 1. get messages from the session
        # 2. get parent message info
        # 3. get llm response and construct new message
        parent_message_id = message.get_message_id()

    def add_chat_session(self, record: Any) -> None:
        try:
            self.session.add(record)
            self.session.commit()
            self.session.refresh(record)
        except Exception as e:
            self.session.rollback()
            raise Exception(e)

    def commit_chat_record(self, conversation: Any, last_message: Any,
                           user_message: Any, ai_message: Any, cost: Any) -> None:
        """
        commit the records changes of one dialog to database
        """

        self.add_chat_session(conversation)
        self.add_chat_session(last_message)
        self.add_chat_session(user_message)
        self.add_chat_session(ai_message)
        self.add_chat_session(cost)

    def cal_cost(self, chat: Dict, user_id: str) -> LlmCost:
        create_time = chat.get('create_time')
        last_cost_id = self.session.exec(select(LlmCost).order_by(desc(LlmCost.id))).first().id
        cost_id = last_cost_id + 1
        message_id = chat.get('message_id')
        prompt_token = chat.get('prompt_token')
        complete_token = chat.get('completion_token')
        total_token = chat.get('total_token')
        cost = chat.get('cost')
        message_cost = LlmCost(id=cost_id,
                               user_id=user_id,
                               message_id=message_id,
                               prompt_token=prompt_token,
                               completion_token=complete_token,
                               total_token=total_token,
                               cost=cost,
                               create_time=create_time)

        return message_cost

    def get_model_list(self) -> Response:
        try:
            models = self.session.exec(select(LlmModel)).all()
            # print(models)
        except Exception as e:
            return Response(code="500", message=e)
        return Response(code="200", message=models)


def get_llm_service(session: Session = Depends(get_session), llm: LlmApi = Depends(get_llm_api)) -> LlmService:
    return LlmService(session, llm)
