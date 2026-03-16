from datetime import datetime
from uuid import UUID, uuid4
from typing import Optional, List

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import Float


class AudioFile(SQLModel, table=True):
    """Class representing audio table"""
    __tablename__ = "audiofiles"
    __table_args__ = {"schema": "public"}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: Optional[UUID] = Field(default=None, foreign_key="public.users.id", index=True)
    filename: str
    content_type: str
    stored_filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration: Optional[float] = Field(default=None)
    avg_volume_dbfs: Optional[float] = Field(default=None)
    avg_pitch_hz: Optional[float] = Field(default=None)
    wpm: Optional[float] = Field(default=None)
    context_mode: Optional[str] = Field(default=None)
    graph_volume: Optional[List[float]] = Field(default=None,sa_column=Column(JSON))
    graph_freq: Optional[List[float]] = Field(default=None,sa_column=Column(JSON))
    user: Optional["UserTable"] = Relationship(back_populates="audio_files")