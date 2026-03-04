from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from infrastructure.container import container
from infrastructure.security import decode_token, get_current_user_id
from application.use_cases.auth import (
    RegisterUseCase,
    LoginUseCase,
    RefreshTokenUseCase,
    ChangeEmailUseCase,
    ChangePasswordUseCase,
)

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


class ChangeEmailRequest(BaseModel):
    new_email: EmailStr


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


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


@router.patch("/auth/email", response_model=UserResponse)
def change_email(body: ChangeEmailRequest, user_id: str = Depends(get_current_user_id)):
    repo = container.get("user_repository")
    use_case = ChangeEmailUseCase(repo)
    try:
        user = use_case.execute(user_id=user_id, new_email=str(body.new_email))
        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            is_active=user.is_active,
        )
    except ValueError as e:
        detail = str(e)
        if detail == "A user with this email already exists":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
        if detail == "User not found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


@router.patch("/auth/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(body: ChangePasswordRequest, user_id: str = Depends(get_current_user_id)):
    repo = container.get("user_repository")
    use_case = ChangePasswordUseCase(repo)
    try:
        use_case.execute(
            user_id=user_id,
            current_password=body.current_password,
            new_password=body.new_password,
        )
        return None
    except ValueError as e:
        detail = str(e)
        if detail == "Current password is incorrect":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
        if detail == "User not found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
