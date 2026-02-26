from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlmodel import SQLModel, Field


class AudioFile(SQLModel, table=True):
    """Class representing audio table"""
    __tablename__ = "audiofiles"
    __table_args__ = {"schema": "public"}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str
    content_type: str
    stored_filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration: Optional[float] = Field(default=None)
    avg_volume_dbfs: Optional[float] = Field(default=None)
    avg_pitch_hz: Optional[float] = Field(default=None)
    wpm: Optional[float] = Field(default=None)
    context_mode: Optional[str] = Field(default=None)  # "In-Person" or "Online"
