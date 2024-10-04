from sqlmodel import Session, SQLModel, create_engine
from yaml import safe_load
import os

print(os.environ)
# TODO: Replace hard coding with env variable
# sql_url = f"mysql+pymysql://{os.environ['database_url']}:{os.environ['database_port']}/{os.environ['database']}?charset=utf8mb4"

sql_url = r"mysql+pymysql://mithzyl:8PjpYSJ3xdrzLJVo@sqlpub.com:3306/mithzyl_llm?charset=utf8mb4"
engine = create_engine(sql_url)

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    with Session(engine) as session:
        yield session

def create_db(db_name: str) -> None:
    engine = create_engine(f"sqlite:///{db_name}")
    create_db_and_tables()
