from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    api_prefix: str = "/api/v1"

    # Auth / JWT settings
    secret_key: str = "jwtsecret"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

