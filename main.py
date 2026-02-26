"""main module for FastAPI backend"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel

from infrastructure.config import settings
from infrastructure.container import container
from infrastructure.api.routes import health, audio, auth
from infrastructure.persistence.database import engine
from infrastructure.persistence.in_memory_repository import InMemoryRepository
from infrastructure.persistence.user_repository import PostgresUserRepository
# Import models so their tables are created at startup
from infrastructure.persistence.user_model import UserTable  # noqa: F401
from infrastructure.persistence.audio_model import AudioFile  # noqa: F401


# ── App creation ──────────────────────────────────────────────
app = FastAPI(
    title="Hexagonal API",
    description="FastAPI with Hexagonal Architecture",
    version="1.0.0",
    debug=settings.debug,
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    """ensure database tables are created at start"""
    SQLModel.metadata.create_all(engine)


# ── Dependency Injection ──────────────────────────────────────
def _setup_dependencies():
    """Setup dependency injection container"""
    container.register("repository", InMemoryRepository())
    container.register("user_repository", PostgresUserRepository(engine))

_setup_dependencies()


# ── Routes ────────────────────────────────────────────────────
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(audio.router, prefix=settings.api_prefix, tags=["audio"])
app.include_router(auth.router, prefix=settings.api_prefix, tags=["auth"])
