from datetime import datetime
from typing import Any, Dict

from fastapi import Depends, HTTPException, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from langchain_community.chat_models import ChatOpenAI
from sqlmodel import Session, desc, select

from db.db import get_session
from llm.llm_api import LlmApi
from models.param.message_param import ChatCreateParam
from models.response.llm_response import LlmDto
from models.response.messgage_response import Response
from models.response.chat_session_response import ChatSession
from models.model.llm_cost import LlmCost
from models.model.llm_message import llm_message, llm_session
from models.model.llm_model import LlmModel
from models.model.user import User
from utils.authenticate import decode_token
from utils.util import generate_md5_id


class LlmService:
    session: Session

    def __init__(self, session: Session):
        self.session = session

    def get_messages_by_conversation_id(self, conversation_id: str) -> Response:
        try:
            messages = (
                self.session.exec(select(llm_message).filter(llm_message.session_id == conversation_id)
                .order_by(llm_message.create_time)).all()
            )
        except Exception as e:
            return Response(code="500", message=str(e))

        return Response(code="200", message=messages)

    def get_message_by_message_id(self, message_id: object) -> Response:
        message = self.session.exec(select(llm_message).filter(llm_message.id == message_id)).first()

        return Response(code="200", message=message)

    def get_sessions_by_user_id(self, user_id) -> Response:
        try:
            sessions = (self.session.exec(select(llm_session).filter(llm_session.user_id == user_id)
                        .order_by(desc(llm_session.update_time))).all())

            session_dto = []
            for session in sessions:
                dto = ChatSession(session_id=session.session_id, title=session.title)
                session_dto.append(dto)

        except Exception as e:
            return Response(code="500", message=str(e))

        return Response(code="200", message=session_dto)

    def create_chat(self, llm_param: ChatCreateParam, token: HTTPAuthorizationCredentials, llm: LlmApi) -> Response:
        # 1. generate a session_id
        # 2. Add initial system message of llm
        # 3. create langchain prompt template
        # 4. call LLM API
        # 5. save message and the session

        # llm = get_llm_api(message, message.temperature)
        payload = decode_token(token)
        email = payload.get("email")
        user = self.session.exec(
            select(User).where(User.email == email)).first()  # get current user TODO: get token from redis
        create_time = datetime.now()

        model = llm_param.get_model()


        if llm_param.get_conversation_id():
            conversation_id = llm_param.get_conversation_id()
            conversation = self.session.exec(select(llm_session)
                                             .where(llm_session.session_id == conversation_id)).first()
            if not conversation:
                raise HTTPException(status_code=404, detail="Resource not found")

            # model = conversation.model

            messages = self.get_messages_by_conversation_id(conversation_id).get_message()
            latest_message = self.session.exec(select(llm_message)
                                               .where(llm_message.session_id == conversation_id)
                                               .order_by(desc(llm_message.create_time))).first()

            new_message = llm_param.get_message()

            last_message_primary_id = self.session.exec(select(llm_message).order_by(desc(llm_message.id))).first().id
            # message from the user
            update_time = datetime.now()
            user_message_primary_id = last_message_primary_id + 1
            user_message_id = generate_md5_id()
            user_message = llm_message(id=user_message_primary_id,
                                       message_id=user_message_id,
                                       session_id=conversation_id,
                                       message=llm_param.get_message(),
                                       user_id=user.userid,
                                       create_time=create_time,
                                       update_time=update_time,
                                       role='user',
                                       parent_id=latest_message.message_id,
                                       children_id='')

            latest_message.children_id = user_message_id
            chat_state_response = llm.chat(messages, new_message, llm.model)
            chat_id = chat_state_response.get('message_id')
            user_message.children_id = chat_id

            ai_primary_id = user_message.id + 1
            ai_message = llm_message(id=ai_primary_id,
                                     message_id=chat_id,
                                     session_id=conversation_id,
                                     message=chat_state_response.get('message'),
                                     user_id=user.userid,
                                     create_time=chat_state_response.get('create_time'),
                                     update_time=chat_state_response.get('create_time'),
                                     role=chat_state_response.get('role'),
                                     parent_id=user_message_id,
                                     children_id='')

            cost = self.cal_cost(chat_state_response, user.userid)

            conversation.update_time = chat_state_response.get('create_time')

            self.add_chat_session(conversation)
            self.add_chat_session(latest_message)
            self.add_chat_session(user_message)
            self.add_chat_session(ai_message)
            self.add_chat_session(cost)

        else:
            try:
                
                # create new conversation
                conversation_id = generate_md5_id()
                update_time = datetime.now()
                # title = llm.generate_conversation_title(llm_param.get_message())
                #
                last_message_primary_id = self.session.exec(
                    select(llm_message).order_by(desc(llm_message.id))).first().id
                # message from the user
                user_message_primary_id = last_message_primary_id + 1
                user_message_id = generate_md5_id()
                user_message = llm_message(id=user_message_primary_id,
                                           message_id=user_message_id,
                                           session_id=conversation_id,
                                           message=llm_param.get_message(),
                                           user_id=user.userid,
                                           create_time=create_time,
                                           update_time=update_time,
                                           role='user',
                                           parent_id='',
                                           children_id='')

                # chat = llm.create_first_chat(llm_param.get_message(), model)
                chat_state = llm.run_workflow(llm_param.get_message())
                chat_state_response = chat_state['response']
                chat_id = chat_state_response.get('message_id')
                chat_title = chat_state['title']
                user_message.children_id = chat_id

                # message response from the model
                ai_message_id = user_message_primary_id + 1
                ai_message = llm_message(id=ai_message_id,
                                         message_id=chat_state_response.get('message_id'),
                                         session_id=conversation_id,
                                         title=chat_title,
                                         message=chat_state_response.get('message'),
                                         user_id=user.userid,
                                         create_time=chat_state_response.get('create_time'),
                                         update_time=chat_state_response.get('create_time'),
                                         role=chat_state_response.get('role'),
                                         parent_id=user_message_id,
                                         children_id='')

                # conversation session
                last_conversation_id = self.session.exec(select(llm_session).order_by(desc(llm_session.id))).first().id
                conversation_primary_id = last_conversation_id + 1
                conversation = llm_session(
                    id=conversation_primary_id,
                    session_id=conversation_id,
                    title=chat_title,
                    user_id=user.userid,
                    create_time=create_time,
                    update_time=update_time,
                    model=chat_state_response.get('model'))

                # cost
                message_cost = self.cal_cost(chat_state_response, user.userid)

                self.add_chat_session(user_message)
                self.add_chat_session(ai_message)
                self.add_chat_session(conversation)
                self.add_chat_session(message_cost)

            # TODO: handle regex to restructure output

            except Exception as e:
                return Response(code="500", message=e)

        conversation_response = LlmDto(conversation_id=conversation_id, content=chat_state_response, model=model)
        return Response(code="200", message=conversation_response)

    def post_message(self, message: ChatCreateParam, token: HTTPAuthorizationCredentials):
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

    def cal_cost(self, chat_state_response: dict, user_id: str) -> LlmCost:
        create_time = chat_state_response.get('create_time')
        last_cost_id = self.session.exec(select(LlmCost).order_by(desc(LlmCost.id))).first().id
        cost_id = last_cost_id + 1
        message_id = chat_state_response.get('message_id')
        prompt_token = chat_state_response.get('prompt_token')
        complete_token = chat_state_response.get('completion_token')
        total_token = chat_state_response.get('total_token')
        cost = chat_state_response.get('cost')
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

        except Exception as e:
            return Response(code="500", message=e)
        return Response(code="200", message=models)



