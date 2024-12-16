
import time
from typing import Any, Dict

from fastapi import HTTPException, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from sqlmodel import Session, desc, select

from fastapiredis.redis_client import RedisClient
from llm.llm_state import LlmGraph
from llm.mem0.mem0_client import CustomMemoryClient
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

    def get_messages_by_conversation_id(self, conversation_id: str, redis_client: RedisClient) -> Response:
        messages = []
        try:
            messages = redis_client.get_conversation_history(conversation_id)
            if not messages:
                messages = (
                    self.session.exec(select(llm_message).filter(llm_message.session_id == conversation_id)
                    .order_by(llm_message.create_time)).all()
                )

                # convert llm_message object to dict for redis storage
                redis_client.set_conversation_by_conversation_id(conversation_id, messages)

        except Exception as e:
            # return Response(code="500", message=str(e))
            print("get_messages_by_conversation_id error: ", e)

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

    def create_chat(self,
                    llm_param: ChatCreateParam,
                    token: str,
                    llm_graph: LlmGraph,
                    redis_client: RedisClient) -> Response:
        # 1. generate a session_id
        # 2. Add initial system message of llm
        # 3. create langchain prompt template
        # 4. call LLM API
        # 5. save message and the session
        # TODO: Add memory support

        # llm = get_llm_api(message, message.temperature)
        payload = decode_token(token)
        email = payload.get("email")
        user = self.session.exec(
            select(User).where(User.email == email)).first()  # get current user TODO: get token from redis
        create_time = int(time.time())

        model = llm_param.get_model()

        conversation_id = llm_param.get_conversation_id()
        try:
            if conversation_id:
                history_conversations = self.get_messages_by_conversation_id(conversation_id, redis_client).get_message()

                # get the conversation session
                conversation = self.session.exec(select(llm_session)
                                                 .where(llm_session.session_id == conversation_id)).first()
                if not history_conversations:
                    raise HTTPException(status_code=404, detail="Resource not found")

                # get the latest message from the conversation for primary id in the database
                conversation_latest_message = self.session.exec(select(llm_message)
                                               .where(llm_message.session_id == conversation_id)
                                               .order_by(desc(llm_message.create_time))).first()
            else:
                history_conversations = None
                conversation_latest_message = None
                conversation = None
                conversation_id = None

            # get the primary id of the last message in the database
            last_message_primary_id = self.session.exec(select(llm_message).order_by(desc(llm_message.id))).first().id
            # message from the user
            new_user_message = llm_param.get_message()
            update_time = int(time.time())
            user_message_primary_id = last_message_primary_id + 1
            user_message_id = generate_md5_id()
            user_message = llm_message(id=user_message_primary_id,
                                       message_id=user_message_id,
                                       session_id=conversation_id,
                                       message=new_user_message,
                                       user_id=user.userid,
                                       create_time=create_time,
                                       update_time=update_time,
                                       role='user',
                                       parent_id=conversation_latest_message.message_id if
                                                conversation_latest_message else '',
                                       children_id='')

            # llm api call
            chat_state = llm_graph.run_first_chat_workflow(new_user_message, history_conversations, user.userid)
            chat_state_response = chat_state['response']
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

            # if new conversation, create new llm_session object
            if not history_conversations:
                chat_title = chat_state['title']

                new_history_conversations = [user_message, ai_message]

                # get the last llm session in the database
                last_conversation = self.session.exec(select(llm_session).order_by(desc(llm_session.id))).first()
                if not last_conversation:
                    last_conversation_id = 0
                else:
                    last_conversation_id = last_conversation.id

                conversation_primary_id = last_conversation_id + 1
                conversation_id = generate_md5_id()
                user_message.session_id = conversation_id
                ai_message.session_id = conversation_id

                conversation = llm_session(
                    id=conversation_primary_id,
                    session_id=conversation_id,
                    title=chat_title,
                    user_id=user.userid,
                    create_time=create_time,
                    update_time=update_time,
                    model=chat_state_response.get('model'))
            else:
                conversation_latest_message.children_id = user_message_id
                conversation.update_time = chat_state_response.get('create_time')
                new_history_conversations = history_conversations
                new_history_conversations.append(user_message)
                new_history_conversations.append(ai_message)

            cost = self.cal_cost(chat_state_response, user.userid)

            # memory sql saving


            self.add_chat_session(conversation)
            if history_conversations:
                self.add_chat_session(conversation_latest_message)
            self.add_chat_session(user_message)
            self.add_chat_session(ai_message)
            self.add_chat_session(cost)

            redis_client.set_conversation_by_conversation_id(conversation_id, new_history_conversations)

        except Exception as e:
            print(e)
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

    def cal_cost(self, chat_state_response: dict, user_id: str) -> LlmCost:
        create_time = chat_state_response.get('create_time')
        last_cost = self.session.exec(select(LlmCost).order_by(desc(LlmCost.id))).first()
        if not last_cost:
            last_cost_id = 0
        else:
            last_cost_id = last_cost.id
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



