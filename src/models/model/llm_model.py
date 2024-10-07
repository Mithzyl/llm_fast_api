from sqlmodel import SQLModel, Field


class LlmModel(SQLModel, table=True):
    __tablename__ = "llm_model"

    id: int = Field(primary_key=True)
    name: str = Field(default=None)

