import os
# Set environment variable BEFORE importing main to ensure database.py picks it up
os.environ["DATABASE_URL"] = "sqlite://"

from fastapi.testclient import TestClient
from sqlmodel import create_engine, SQLModel
from sqlalchemy.pool import StaticPool
from sqlalchemy import event

import main
import infrastructure.persistence.database
from infrastructure.container import container
from infrastructure.persistence.user_repository import PostgresUserRepository

# Setup in-memory SQLite for E2E tests to avoid needing a real Postgres DB
test_engine = create_engine(
    "sqlite://", 
    connect_args={"check_same_thread": False}, 
    poolclass=StaticPool
)

# Fix for "no such table: public.users" when models define schema="public"
# We attach a virtual 'public' database to SQLite so the generated SQL works.
@event.listens_for(test_engine, "connect")
def attach_public_schema(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("ATTACH DATABASE ':memory:' AS public")
    cursor.close()

# Patch the engine in the database module (in case it was already imported)
infrastructure.persistence.database.engine = test_engine

# Patch the engine in main so the startup event creates tables in SQLite
main.engine = test_engine

# Re-register the repository to use the SQLite engine
container.register("user_repository", PostgresUserRepository(test_engine))

# Ensure tables are created (TestClient without context manager doesn't run startup events reliably)
SQLModel.metadata.create_all(test_engine)

client = TestClient(main.app)


def test_health_endpoint():
    """
    Simple end-to-end test hitting the public health endpoint.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"


def test_auth_end_to_end_login_and_me():
    """
    End-to-end flow for auth:
    - register a user
    - log in
    - call /auth/me with the returned access token

    This uses the same FastAPI app instance (with its DI container)
    that is used in production.
    """
    email = "e2e_user@example.com"
    password = "e2e-password"
    username = "e2e-user"

    # Register
    register_resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "username": username, "password": password},
    )
    assert register_resp.status_code in (200, 201)
    user_data = register_resp.json()
    assert user_data["email"] == email
    assert user_data["username"] == username

    # Login
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login_resp.status_code == 200
    tokens = login_resp.json()
    access_token = tokens.get("access_token")
    assert tokens.get("token_type") == "bearer"
    assert access_token

    # Call /auth/me with bearer token
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["email"] == email
    assert me_data["username"] == username
