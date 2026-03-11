from domain.entities.user import User
import uuid

def test_user_model_creation():
    """Test that the User domain entity can be instantiated correctly."""
    u_id = str(uuid.uuid4())
    user = User(
        id=u_id,
        email="test@test.com",
        username="tester",
        hashed_password="hash"
    )
    assert user.email == "test@test.com"
    assert user.username == "tester"
    assert user.is_active is True
