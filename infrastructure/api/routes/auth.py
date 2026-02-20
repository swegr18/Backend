from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from infrastructure.container import container
from infrastructure.security import decode_token, get_current_user_id
from application.use_cases.auth import RegisterUseCase, LoginUseCase, RefreshTokenUseCase

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str = None
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_active: bool

    class Config:
        from_attributes = True


@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    """Register a new user account"""
    repo = container.get("user_repository")
    use_case = RegisterUseCase(repo)
    try:
        user = use_case.execute(body.email, body.username, body.password)
        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            is_active=user.is_active,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """Log in and receive access + refresh tokens"""
    repo = container.get("user_repository")
    use_case = LoginUseCase(repo)
    try:
        tokens = use_case.execute(body.email, body.password)
        return tokens
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/auth/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest):
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user_id = payload.get("sub")
    repo = container.get("user_repository")
    use_case = RefreshTokenUseCase(repo)
    try:
        result = use_case.execute(user_id)
        return result
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")


@router.get("/auth/me", response_model=UserResponse)
def get_me(user_id: str = Depends(get_current_user_id)):
    repo = container.get("user_repository")
    user = repo.find_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        is_active=user.is_active,
    )
