from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


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

