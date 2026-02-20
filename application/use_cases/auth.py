from domain.ports import UserRepository
from infrastructure.security import hash_password, verify_password, create_access_token, create_refresh_token


class RegisterUseCase:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def execute(self, email: str, username: str, password: str):
        existing = self.user_repo.find_by_email(email)
        if existing:
            raise ValueError("A user with this email already exists")

        hashed = hash_password(password)
        user = self.user_repo.save_user({
            "email": email,
            "username": username,
            "hashed_password": hashed,
        })
        return user


class LoginUseCase:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def execute(self, email: str, password: str):
        user = self.user_repo.find_by_email(email)
        if not user:
            raise ValueError("Invalid email or password")

        if not verify_password(password, user.hashed_password):
            raise ValueError("Invalid email or password")

        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }


class RefreshTokenUseCase:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def execute(self, user_id: str):
        user = self.user_repo.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        access_token = create_access_token(data={"sub": str(user.id)})
        return {
            "access_token": access_token,
            "token_type": "bearer",
        }
