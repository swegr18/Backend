from sqlmodel import SQLModel, Field, Relationship
from typing import List
from uuid import UUID, uuid4
from datetime import datetime


class UserTable(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = {"schema": "public"}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    username: str
    hashed_password: str
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    audio_files: List["AudioFile"] = Relationship(back_populates="user")
