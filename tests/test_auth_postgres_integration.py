import os

import pytest
from sqlmodel import SQLModel, Session

from application.use_cases.auth import LoginUseCase
from infrastructure.persistence.database import engine
from infrastructure.persistence import audio_model
from infrastructure.persistence.user_model import UserTable
from infrastructure.persistence.user_repository import PostgresUserRepository
from infrastructure.security import hash_password


requires_postgres = pytest.mark.skipif(
    os.getenv("TEST_WITH_DB") != "1",
    reason="Postgres-backed tests disabled (set TEST_WITH_DB=1 to enable).",
)


@requires_postgres
def test_fr13_login_existing_user_with_postgres_tables():
    """
    FR1.3: The system shall allow a user to log in to an existing account
    (backed by real Postgres tables).

    This test verifies the login flow end-to-end across:
    - `UserTable` (SQLModel model mapped to the `users` table)
    - `PostgresUserRepository` (Postgres-backed repository)
    - `LoginUseCase` (application logic + token creation)
    """
    # Ensure tables exist for the test database
    SQLModel.metadata.create_all(engine)

    email = "pg-user@example.com"
    password = "pg-password"
    hashed = hash_password(password)

    # Seed a user row directly into the database using the SQLModel Session
    with Session(engine) as session:
        # Clean up any previous test data with the same email
        existing_users = session.query(UserTable).filter(UserTable.email == email).all()
        for user in existing_users:
            session.delete(user)
        session.commit()

        user_row = UserTable(
            email=email,
            username="pg-user",
            hashed_password=hashed,
        )
        session.add(user_row)
        session.commit()
        session.refresh(user_row)

    # Use the real Postgres-backed repository + login use case
    repo = PostgresUserRepository(engine)
    use_case = LoginUseCase(repo)

    result = use_case.execute(email=email, password=password)

    assert result["token_type"] == "bearer"
    assert isinstance(result["access_token"], str) and result["access_token"]
    assert isinstance(result["refresh_token"], str) and result["refresh_token"]

