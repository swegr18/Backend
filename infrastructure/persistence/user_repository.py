from typing import Optional
from sqlmodel import Session, select
from domain.ports import UserRepository
from infrastructure.persistence.user_model import UserTable
from uuid import uuid4, UUID


class PostgresUserRepository(UserRepository):
    """Implements UserRepository port using SQLModel + PostgreSQL"""

    def __init__(self, engine):
        self.engine = engine

    def save_user(self, user_data: dict) -> UserTable:
        row = UserTable(
            id=uuid4(),
            email=user_data["email"],
            username=user_data["username"],
            hashed_password=user_data["hashed_password"],
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def find_by_email(self, email: str) -> Optional[UserTable]:
        with Session(self.engine) as session:
            statement = select(UserTable).where(UserTable.email == email)
            return session.exec(statement).first()

    def find_by_id(self, user_id: str) -> Optional[UserTable]:
        with Session(self.engine) as session:
            # Convert string to UUID if needed
            if isinstance(user_id, str):
                user_id = UUID(user_id)
            statement = select(UserTable).where(UserTable.id == user_id)
            return session.exec(statement).first()
