import uuid

import bcrypt
from sqlmodel import Session, desc

from db.db import get_session
from models.dto.messgage_dto import Message
from models.model.user import User


def get_users(session: Session):
    return session.query(User).all()


def get_user_by_id(id, session):
    return session.query(User).filter(User.id == id).first()


def login(login_request, session):
    # query user, not exist return error
    user = session.query(User).filter(User.email == login_request.email).first()
    if not user:
        return Message(code="200", message="no such user")

    if login_request.password != user.password:
        return Message(code="200", message="wrong password or email")

    return Message(code="200", message="success")



def register(register_request, session):
    # 1. generate uuid
    # 2. encrypt password
    new_id = session.query(User).order_by(desc(User.id)).first().id + 1
    new_uuid = str(uuid.uuid4())
    print(register_request)
    role = 'user'
    hash_password = bcrypt.hashpw(register_request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(new_id, new_uuid, register_request.email, register_request.name, hash_password, role)

    try:
        session.add(new_user)
        session.commit()
        return Message(code="200", message="success")
    except Exception as e:
        session.rollback()
        return Message(code="500", message=str(e))

    # 3. TODO: Auto login after register