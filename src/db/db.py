from sqlmodel import Session, SQLModel, create_engine
from yaml import safe_load
import os

from utils.util import read_yaml_config

db_config = read_yaml_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"))['database_config']
sql_url = f"mysql+pymysql://{db_config['mysql_url']}:{db_config['port']}/{db_config['database_name']}?charset=utf8mb4"

engine = create_engine(sql_url)

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    with Session(engine) as session:
        yield session

def create_db(db_name: str) -> None:
    engine = create_engine(f"sqlite:///{db_name}")
    create_db_and_tables()
