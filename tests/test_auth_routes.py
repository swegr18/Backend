import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.api.routes.auth import router
from infrastructure.container import container
from infrastructure.security import hash_password, create_access_token, create_refresh_token
from domain.ports import UserRepository


class FakeUser:
    def __init__(self, id, email, username, hashed_password, is_active=True):
        self.id = id
        self.email = email
        self.username = username
        self.hashed_password = hashed_password
        self.is_active = is_active


class InMemoryUserRepository(UserRepository):

    def __init__(self):
        self._by_id = {}
        self._by_email = {}

    def save_user(self, data: dict):
        uid = data.get("id") or str(uuid.uuid4())
        user = FakeUser(
            id=uid,
            email=data["email"],
            username=data["username"],
            hashed_password=data["hashed_password"],
            is_active=data.get("is_active", True),
        )
        self._by_id[uid] = user
        self._by_email[user.email] = user
        return user

    def find_by_email(self, email):
        return self._by_email.get(email)

    def find_by_id(self, user_id):
        return self._by_id.get(str(user_id))

    def update_email(self, user_id, new_email):
        user = self.find_by_id(user_id)
        if not user:
            return None
        self._by_email.pop(user.email, None)
        user.email = new_email
        self._by_email[new_email] = user
        return user

    def update_password(self, user_id, new_hashed_password):
        user = self.find_by_id(user_id)
        if not user:
            return None
        user.hashed_password = new_hashed_password
        return user


@pytest.fixture(autouse=True)
def _setup_container():
    repo = InMemoryUserRepository()
    container.register("user_repository", repo)
    yield repo


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _seed_user(repo, email="user@example.com", password="strongpassword", username="testuser"):
    hashed = hash_password(password)
    return repo.save_user({
        "email": email,
        "username": username,
        "hashed_password": hashed,
    })


def _auth_header(user_id: str) -> dict:
    token = create_access_token(data={"sub": user_id})
    return {"Authorization": f"Bearer {token}"}

def test_register_success(client, _setup_container):
    resp = client.post("/auth/register", json={
        "email": "new@example.com",
        "username": "newuser",
        "password": "securepass",
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "new@example.com"
    assert body["username"] == "newuser"
    assert body["is_active"] is True


def test_register_duplicate_email(client, _setup_container):
    _seed_user(_setup_container, email="dup@example.com")
    resp = client.post("/auth/register", json={
        "email": "dup@example.com",
        "username": "another",
        "password": "securepass",
    })
    assert resp.status_code == 409


def test_login_success(client, _setup_container):
    _seed_user(_setup_container, email="login@example.com", password="mypassword")
    resp = client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "mypassword",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert "access_token" in body
    assert "refresh_token" in body


def test_login_wrong_password(client, _setup_container):
    _seed_user(_setup_container, email="login@example.com", password="correct")
    resp = client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "wrong",
    })
    assert resp.status_code == 401


def test_login_nonexistent_user(client):
    resp = client.post("/auth/login", json={
        "email": "nobody@example.com",
        "password": "any",
    })
    assert resp.status_code == 401


def test_refresh_success(client, _setup_container):
    user = _seed_user(_setup_container)
    refresh = create_refresh_token(data={"sub": str(user.id)})
    resp = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert "access_token" in body


def test_refresh_with_access_token_rejected(client, _setup_container):
    user = _seed_user(_setup_container)
    access = create_access_token(data={"sub": str(user.id)})
    resp = client.post("/auth/refresh", json={"refresh_token": access})
    assert resp.status_code == 401


def test_refresh_invalid_token(client):
    resp = client.post("/auth/refresh", json={"refresh_token": "garbage"})
    assert resp.status_code == 401


def test_get_me_success(client, _setup_container):
    user = _seed_user(_setup_container)
    resp = client.get("/auth/me", headers=_auth_header(str(user.id)))
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "user@example.com"
    assert body["username"] == "testuser"


def test_get_me_no_token(client):
    resp = client.get("/auth/me")
    assert resp.status_code in (401, 403)


def test_get_me_user_not_found(client, _setup_container):
    fake_id = str(uuid.uuid4())
    resp = client.get("/auth/me", headers=_auth_header(fake_id))
    assert resp.status_code == 404


def test_change_email_success(client, _setup_container):
    user = _seed_user(_setup_container, email="old@example.com")
    resp = client.patch(
        "/auth/email",
        json={"new_email": "new@example.com"},
        headers=_auth_header(str(user.id)),
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "new@example.com"


def test_change_email_conflict(client, _setup_container):
    user = _seed_user(_setup_container, email="me@example.com")
    _seed_user(_setup_container, email="taken@example.com", username="other")
    resp = client.patch(
        "/auth/email",
        json={"new_email": "taken@example.com"},
        headers=_auth_header(str(user.id)),
    )
    assert resp.status_code == 409


def test_change_email_user_not_found(client, _setup_container):
    fake_id = str(uuid.uuid4())
    resp = client.patch(
        "/auth/email",
        json={"new_email": "x@example.com"},
        headers=_auth_header(fake_id),
    )
    assert resp.status_code == 404


def test_change_password_success(client, _setup_container):
    user = _seed_user(_setup_container, password="current")
    resp = client.patch(
        "/auth/password",
        json={"current_password": "current", "new_password": "newpass"},
        headers=_auth_header(str(user.id)),
    )
    assert resp.status_code == 204


def test_change_password_wrong_current(client, _setup_container):
    user = _seed_user(_setup_container, password="correct")
    resp = client.patch(
        "/auth/password",
        json={"current_password": "wrong", "new_password": "newpass"},
        headers=_auth_header(str(user.id)),
    )
    assert resp.status_code == 401


def test_change_password_user_not_found(client, _setup_container):
    fake_id = str(uuid.uuid4())
    resp = client.patch(
        "/auth/password",
        json={"current_password": "any", "new_password": "newpass"},
        headers=_auth_header(fake_id),
    )
    assert resp.status_code == 404
