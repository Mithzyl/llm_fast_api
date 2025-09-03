from dataclasses import asdict, dataclass

from sqlmodel import SQLModel, Field

@dataclass
class LlmModel(SQLModel, table=True):
    __tablename__ = "llm_model"

    id: int = Field(primary_key=True)
    name: str = Field(default=None)

    def to_dict(self):
        return asdict(self)

