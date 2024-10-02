from dataclasses import Field
from sqlmodel import SQLModel


class Role(SQLModel, table=True):
    role = Field(str)