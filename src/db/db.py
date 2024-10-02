from sqlmodel import Session, SQLModel, create_engine
from yaml import safe_load

# TODO: Replace hard coding with env variable
sql_url = r"mysql+pymysql://root:huyuan1649@localhost:3306/llm_user?charset=utf8mb4"
engine = create_engine(sql_url)

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    with Session(engine) as session:
        yield session

def create_db(db_name: str) -> None:
    engine = create_engine(f"sqlite:///{db_name}")
    create_db_and_tables()
