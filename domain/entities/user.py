from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


class User(BaseModel):
    """Domain entity representing a user"""
    id: Optional[str] = None
    email: str
    username: str
    hashed_password: str
    is_active: bool = True
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
