"""Tests for health check routes."""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from infrastructure.api.routes.health import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_check_returns_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_returns_welcome(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API"}
