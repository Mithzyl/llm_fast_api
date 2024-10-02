from sqlmodel import Session

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


    return None