import uuid

from application.use_cases.auth import LoginUseCase, ChangeEmailUseCase, ChangePasswordUseCase
from domain.ports import UserRepository
from infrastructure.security import hash_password, verify_password


class FakeUser:
    def __init__(self, id: str, email: str, username: str, hashed_password: str, is_active: bool = True):
        self.id = id
        self.email = email
        self.username = username
        self.hashed_password = hashed_password
        self.is_active = is_active


class InMemoryUserRepository(UserRepository):
    """Simple in-memory implementation of UserRepository for unit tests."""

    def __init__(self):
        self._users_by_id = {}
        self._users_by_email = {}

    def save_user(self, user_data: dict):
        user_id = user_data.get("id") or str(uuid.uuid4())
        user = FakeUser(
            id=user_id,
            email=user_data["email"],
            username=user_data["username"],
            hashed_password=user_data["hashed_password"],
            is_active=user_data.get("is_active", True),
        )
        self._users_by_id[user_id] = user
        self._users_by_email[user.email] = user
        return user

    def find_by_email(self, email: str):
        return self._users_by_email.get(email)

    def find_by_id(self, user_id: str):
        return self._users_by_id.get(str(user_id))

    def update_email(self, user_id: str, new_email: str):
        user = self.find_by_id(user_id)
        if not user:
            return None
        self._users_by_email.pop(user.email, None)
        user.email = new_email
        self._users_by_email[new_email] = user
        return user

    def update_password(self, user_id: str, new_hashed_password: str):
        user = self.find_by_id(user_id)
        if not user:
            return None
        user.hashed_password = new_hashed_password
        return user


def _create_login_use_case_with_user(email: str, password: str) -> LoginUseCase:
    repo = InMemoryUserRepository()
    hashed = hash_password(password)
    repo.save_user(
        {
            "email": email,
            "username": "test-user",
            "hashed_password": hashed,
        }
    )
    return LoginUseCase(repo)


def test_fr13_login_existing_user_success():
    """
    FR1.3: The system shall allow a user to log in to an existing account.

    This test covers the successful login path at the application (use-case) layer.
    """
    email = "user@example.com"
    password = "strong-password"
    use_case = _create_login_use_case_with_user(email, password)

    result = use_case.execute(email=email, password=password)

    assert result["token_type"] == "bearer"
    assert isinstance(result["access_token"], str) and result["access_token"]
    assert isinstance(result["refresh_token"], str) and result["refresh_token"]


def test_fr13_login_fails_with_wrong_password():
    """FR1.3 negative case: login should fail with an incorrect password."""
    email = "user@example.com"
    correct_password = "correct-password"
    wrong_password = "wrong-password"
    use_case = _create_login_use_case_with_user(email, correct_password)

    try:
        use_case.execute(email=email, password=wrong_password)
        assert False, "Expected ValueError for invalid credentials"
    except ValueError as exc:
        assert "Invalid email or password" in str(exc)


def test_fr13_login_fails_for_nonexistent_user():
    """FR1.3 negative case: login should fail if the user does not exist."""
    repo = InMemoryUserRepository()
    use_case = LoginUseCase(repo)

    try:
        use_case.execute(email="missing@example.com", password="any-password")
        assert False, "Expected ValueError for invalid credentials"
    except ValueError as exc:
        assert "Invalid email or password" in str(exc)


def _create_user(repo: InMemoryUserRepository, email: str, password: str, username: str = "test-user"):
    return repo.save_user(
        {
            "email": email,
            "username": username,
            "hashed_password": hash_password(password),
        }
    )


def test_change_email_success():
    repo = InMemoryUserRepository()
    user = _create_user(repo, "old@example.com", "old-password")
    use_case = ChangeEmailUseCase(repo)

    updated = use_case.execute(user_id=user.id, new_email="new@example.com")

    assert updated.email == "new@example.com"
    assert repo.find_by_email("new@example.com") is not None
    assert repo.find_by_email("old@example.com") is None


def test_change_email_conflict():
    repo = InMemoryUserRepository()
    user = _create_user(repo, "first@example.com", "password-1")
    _create_user(repo, "taken@example.com", "password-2", username="another")
    use_case = ChangeEmailUseCase(repo)

    try:
        use_case.execute(user_id=user.id, new_email="taken@example.com")
        assert False, "Expected ValueError for duplicate email"
    except ValueError as exc:
        assert "A user with this email already exists" in str(exc)


def test_change_password_success():
    repo = InMemoryUserRepository()
    user = _create_user(repo, "user@example.com", "current-password")
    previous_hash = user.hashed_password
    use_case = ChangePasswordUseCase(repo)

    updated = use_case.execute(
        user_id=user.id,
        current_password="current-password",
        new_password="new-password",
    )

    assert updated.hashed_password != previous_hash
    assert verify_password("new-password", updated.hashed_password)


def test_change_password_fails_with_wrong_current_password():
    repo = InMemoryUserRepository()
    user = _create_user(repo, "user@example.com", "current-password")
    use_case = ChangePasswordUseCase(repo)

    try:
        use_case.execute(
            user_id=user.id,
            current_password="wrong-password",
            new_password="new-password",
        )
        assert False, "Expected ValueError for incorrect current password"
    except ValueError as exc:
        assert "Current password is incorrect" in str(exc)

