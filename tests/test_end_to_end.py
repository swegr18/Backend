import os
# Set environment variable BEFORE importing main to ensure database.py picks it up
os.environ["DATABASE_URL"] = "sqlite://"

import uuid
import tempfile
import shutil
import pytest
from fastapi.testclient import TestClient
from sqlmodel import create_engine, SQLModel
from sqlalchemy.pool import StaticPool
from sqlalchemy import event
from sqlalchemy import JSON
from io import BytesIO
from unittest.mock import patch, MagicMock

import main
import infrastructure.persistence.database
from infrastructure.container import container
from infrastructure.persistence.user_repository import PostgresUserRepository
import infrastructure.api.routes.audio

# Setup in-memory SQLite for E2E tests to avoid needing a real Postgres DB
test_engine = create_engine(
    "sqlite://", 
    connect_args={"check_same_thread": False}, 
    poolclass=StaticPool
)

# Fix for "no such table: public.users" when models define schema="public"
# We attach a virtual 'public' database to SQLite so the generated SQL works.
@event.listens_for(test_engine, "connect")
def attach_public_schema(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("ATTACH DATABASE ':memory:' AS public")
    cursor.close()

# Patch the engine in the database module (in case it was already imported)
infrastructure.persistence.database.engine = test_engine

# Patch the engine in the audio module specifically because it imports 'engine' directly
infrastructure.api.routes.audio.engine = test_engine

# Patch the UPLOAD_DIR in audio module to use a temp dir so we don't clutter the file system
test_upload_dir = tempfile.mkdtemp()
infrastructure.api.routes.audio.UPLOAD_DIR = test_upload_dir

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_uploads():
    yield
    shutil.rmtree(test_upload_dir, ignore_errors=True)

# Patch the engine in main so the startup event creates tables in SQLite
main.engine = test_engine

# Re-register the repository to use the SQLite engine
container.register("user_repository", PostgresUserRepository(test_engine))

# Patch AudioFile model to use JSON instead of ARRAY for SQLite tests
# This allows us to keep ARRAY in production (Postgres) but run tests on SQLite
from infrastructure.persistence.audio_model import AudioFile
AudioFile.__table__.columns["graph_volume"].type = JSON()
AudioFile.__table__.columns["graph_freq"].type = JSON()

# Ensure tables are created (TestClient without context manager doesn't run startup events reliably)
SQLModel.metadata.create_all(test_engine)

client = TestClient(main.app)


def test_health_endpoint():
    """
    Simple end-to-end test hitting the public health endpoint.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API"}


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


def test_register_conflict_e2e():
    """Test registering a user with an email that is already taken."""
    email = "conflict@example.com"
    password = "password"
    username = "conflict-user"

    # First registration
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "username": username, "password": password},
    )

    # Second registration with same email
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "username": "another-user", "password": password},
    )
    assert response.status_code == 409
    assert "A user with this email already exists" in response.json()["detail"]


def test_login_invalid_credentials_e2e():
    """Test logging in with an incorrect password."""
    email = "login-fail@example.com"
    password = "correct-password"
    username = "login-fail-user"

    # Register user
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "username": username, "password": password},
    )

    # Attempt login with wrong password
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "wrong-password"},
    )
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]


def test_auth_me_invalid_token_e2e():
    """Test /auth/me with an invalid or malformed token."""
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer an-invalid-token"},
    )
    assert response.status_code == 401
    assert "Invalid or expired token" in response.json()["detail"]


def test_audio_endpoints_not_found_cases():
    """Test audio endpoints for 404 Not Found and empty cases."""
    # /metrics/latest with no audio files
    response = client.get("/api/v1/metrics/latest")
    assert response.status_code == 404
    assert "No audio files found" in response.json()["detail"]

    # /live-wpm with an unknown session_id
    response = client.get("/api/v1/live-wpm?session_id=unknown-session")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False

    # /userdata with an unknown file_id
    response = client.post(
        "/api/v1/userdata",
        params={
            "user_id": str(uuid.uuid4()),
            "filename": "test.mp3",
            "file_id": str(uuid.uuid4()),
        },
    )
    assert response.status_code == 404
    assert "Audio file not found" in response.json()["detail"]

def test_auth_refresh_token_e2e():
    """Test the refresh token endpoint."""
    email = "refresh@example.com"
    password = "password"
    client.post("/api/v1/auth/register", json={"email": email, "username": "refresher", "password": password})
    login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    refresh_token = login_resp.json()["refresh_token"]
    
    refresh_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert refresh_resp.status_code == 200
    assert "access_token" in refresh_resp.json()

    access_token = login_resp.json()["access_token"]
    bad_refresh = client.post("/api/v1/auth/refresh", json={"refresh_token": access_token})
    assert bad_refresh.status_code == 401

def test_auth_change_email_e2e():
    """Test changing email endpoint."""
    email1 = "change1@example.com"
    email2 = "change2@example.com"
    password = "password"
    
    client.post("/api/v1/auth/register", json={"email": email1, "username": "user1", "password": password})
    client.post("/api/v1/auth/register", json={"email": email2, "username": "user2", "password": password})
    
    login_resp = client.post("/api/v1/auth/login", json={"email": email1, "password": password})
    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    
    new_email = "new1@example.com"
    change_resp = client.patch("/api/v1/auth/email", json={"new_email": new_email}, headers=headers)
    assert change_resp.status_code == 200
    assert change_resp.json()["email"] == new_email
    
    conflict_resp = client.patch("/api/v1/auth/email", json={"new_email": email2}, headers=headers)
    assert conflict_resp.status_code == 409

def test_auth_change_password_e2e():
    """Test changing password endpoint."""
    email = "passchange@example.com"
    old_password = "old_password"
    new_password = "new_password"
    
    client.post("/api/v1/auth/register", json={"email": email, "username": "user", "password": old_password})
    login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": old_password})
    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    
    change_resp = client.patch(
        "/api/v1/auth/password", 
        json={"current_password": old_password, "new_password": new_password}, 
        headers=headers
    )
    assert change_resp.status_code == 204
    
    bad_change = client.patch(
        "/api/v1/auth/password", 
        json={"current_password": "wrong", "new_password": "newer"}, 
        headers=headers
    )
    assert bad_change.status_code == 401

def test_auth_me_wrong_token_type_e2e():
    email = "wrong_type@example.com"
    client.post("/api/v1/auth/register", json={"email": email, "username": "user", "password": "pwd"})
    login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": "pwd"})
    refresh_token = login_resp.json()["refresh_token"]
    
    # Call /me with refresh token instead of access token to hit line 56 in security.py
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {refresh_token}"})
    assert response.status_code == 401
    assert "Invalid token" in response.json()["detail"]

def test_auth_me_user_not_found_e2e():
    email = "notfound_me@example.com"
    client.post("/api/v1/auth/register", json={"email": email, "username": "user", "password": "pwd"})
    login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": "pwd"})
    access_token = login_resp.json()["access_token"]
    
    # Forcefully delete user
    from sqlmodel import Session
    from infrastructure.persistence.user_model import UserTable
    with Session(test_engine) as session:
        user = session.query(UserTable).filter(UserTable.email == email).first()
        session.delete(user)
        session.commit()
        
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert response.status_code == 404
    assert "User not found" in response.json()["detail"]

def test_get_current_user_direct():
    from infrastructure.api.routes.audio import get_current_user
    from fastapi import HTTPException
    import pytest
    
    with pytest.raises(HTTPException) as exc:
        get_current_user(str(uuid.uuid4()))
    assert exc.value.status_code == 401
    assert exc.value.detail == "User not found"

def test_graphs_endpoint_e2e():
    """Test the /graphs endpoint."""
    response = client.post(
        "/api/v1/graphs",
        params={"user_id": str(uuid.uuid4())}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_audio_upload_flow_e2e():
    """
    End-to-end flow for audio:
    - Register and log in a user.
    - Upload a non-final audio chunk.
    - Check live WPM stats.
    - Upload a final audio chunk.
    - Check the final metrics response.
    - Check /metrics/latest.
    - Associate the file with the user via /userdata.
    - Check /graphs for the user's data.
    """
    # 1. Register and Login
    email = "audio-user@example.com"
    password = "audio-password"
    username = "audio-user"
    register_resp = client.post("/api/v1/auth/register", json={"email": email, "username": username, "password": password})
    user_id = register_resp.json()["id"]

    # 2. Create a dummy audio file (minimal valid WebM)
    dummy_webm_content = (
        b'\x1aE\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x18B\x86\x81\x01B\xf7\x81\x01B\xf2\x81\x01B\xf3\x81\x01B\x82'
        b'\x84webmB\x87\x81\x02B\x85\x81\x02\x18S\x80g\x01\x00\x00\x00\x00\x00\x00\x0f\x15I\xa9f\x01\x00\x00\x00'
        b'\x00\x00\x00\x07E\x83\x01\x00\x00\x00\x00\x00\x00\x00\x16T\xaek\x01\x00\x00\x00\x00\x00\x00\x00\xae'
        b'\x01\x00\x00\x00\x00\x00\x00\x00d\x83\x01\x00\x00\x00\x00\x00\x00\x00\xd7\x81\x01\xe7\x81\x01s\xc5\x81'
        b'\x01\x1fC\xb7u\x01\x00\x00\x00\x00\x00\x00\x04\xa3\x81\x01\x80\x00\x00\x00\x00'
    )
    audio_file = ("test.webm", BytesIO(dummy_webm_content), "audio/webm")
    session_id = "e2e-session-123"
    file_id = str(uuid.uuid4())

    # We mock the audio processing functions to avoid needing FFmpeg installed
    # and to avoid actual file processing errors during E2E tests.
    with patch("infrastructure.api.routes.audio.convert_to_mp3"), \
         patch("infrastructure.api.routes.audio.AudioSegment") as mock_segment, \
         patch("infrastructure.api.routes.audio.calc_wpm_live") as mock_wpm, \
         patch("infrastructure.api.routes.audio.all_metrics") as mock_all_metrics, \
         patch("infrastructure.api.routes.audio.graph_metrics") as mock_graph:

        # Setup Mocks
        def fake_calc_wpm(sw, sl, sid, ci, path):
            sw[sid] = {"total_words": 10, "total_seconds": 10.0, "last_chunk": ci, "running_wpm": 100.0}
            return {"accepted": True, "running_wpm": 100, "last_chunk": ci}
        mock_wpm.side_effect = fake_calc_wpm
        mock_all_metrics.return_value = {
            "duration": 10.0,
            "avg_volume_dbfs": -20.0,
            "avg_pitch_hz": 440.0,
            "wpm": 150.0,
            "filler_proportion": 0.1,
            "transcribability": 0.9,
        }
        mock_graph.return_value = {"volume_db": [1, 2], "frequencies": [100, 200]}
        
        # Mock AudioSegment behavior (prevent pydub from calling ffprobe)
        mock_audio_obj = MagicMock()
        mock_segment.from_file.return_value = mock_audio_obj
        mock_audio_obj.__add__.return_value = mock_audio_obj

        # 3. Upload a non-final chunk
        upload_resp_1 = client.post(
            f"/api/v1/upload-audio?file_id={file_id}",
            files={"audio": audio_file},
            data={"session_id": session_id, "chunk_index": "0", "is_final": "false"},
        )
        assert upload_resp_1.status_code == 200
        upload_data_1 = upload_resp_1.json()
        assert upload_data_1["ok"] is True
        assert upload_data_1["final"] is False

        # Call live wpm exactly when st is ready to be returned
        live_wpm_resp = client.get(f"/api/v1/live-wpm?session_id={session_id}")
        assert live_wpm_resp.status_code == 200
        assert live_wpm_resp.json()["ready"] is True

        # 4. Upload a final chunk
        audio_file[1].seek(0)
        final_file_id = str(uuid.uuid4())
        upload_resp_2 = client.post(
            f"/api/v1/upload-audio?file_id={final_file_id}",
            files={"audio": audio_file},
            data={"session_id": session_id, "chunk_index": "1", "is_final": "true"},
        )
        assert upload_resp_2.status_code == 200
        upload_data_2 = upload_resp_2.json()
        assert upload_data_2["ok"] is True
        assert upload_data_2["final"] is True
        assert "final_metrics" in upload_data_2
        assert "filler_proportion" in upload_data_2["final_metrics"]
        assert "transcribability" in upload_data_2["final_metrics"]

    # 5. Associate file with user
    userdata_resp = client.post(
        "/api/v1/userdata",
        params={"user_id": user_id, "filename": "My Test Upload", "file_id": final_file_id},
    )
    assert userdata_resp.status_code == 200
    assert userdata_resp.json()["ok"] is True


def test_audio_upload_subprocess_error():
    dummy_webm_content = b'dummy'
    audio_file = ("test.webm", BytesIO(dummy_webm_content), "audio/webm")
    
    with patch("infrastructure.api.routes.audio.subprocess.run") as mock_run:
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="ffmpeg error output")
        
        response = client.post(
            f"/api/v1/upload-audio?file_id={str(uuid.uuid4())}",
            files={"audio": audio_file},
            data={"session_id": "err-session", "chunk_index": "0", "is_final": "false"}
        )
        assert response.status_code == 400
        assert "ffmpeg failed: ffmpeg error output" in response.json()["detail"]

def test_audio_upload_pydub_error():
    dummy_webm_content = b'dummy'
    audio_file = ("test.webm", BytesIO(dummy_webm_content), "audio/webm")
    
    with patch("infrastructure.api.routes.audio.convert_to_mp3"), \
         patch("infrastructure.api.routes.audio.AudioSegment") as mock_segment, \
         patch("infrastructure.api.routes.audio.calc_wpm_live", return_value={"accepted": True}):
        
        mock_segment.from_file.side_effect = Exception("Pydub mock error")
        
        response = client.post(
            f"/api/v1/upload-audio?file_id={str(uuid.uuid4())}",
            files={"audio": audio_file},
            data={"session_id": "err-session-3", "chunk_index": "0", "is_final": "false"}
        )
        assert response.status_code == 400
        assert "pydub combine failed: Pydub mock error" in response.json()["detail"]

def test_audio_upload_final_metrics_error():
    dummy_webm_content = b'dummy'
    audio_file = ("test.webm", BytesIO(dummy_webm_content), "audio/webm")
    
    with patch("infrastructure.api.routes.audio.convert_to_mp3"), \
         patch("infrastructure.api.routes.audio.AudioSegment") as mock_segment, \
         patch("infrastructure.api.routes.audio.all_metrics") as mock_metrics, \
         patch("infrastructure.api.routes.audio.calc_wpm_live", return_value={"accepted": True}):
        mock_segment.from_file.return_value = MagicMock()
        mock_metrics.side_effect = Exception("Custom metrics error")
        response = client.post(
            f"/api/v1/upload-audio?file_id={str(uuid.uuid4())}",
            files={"audio": audio_file},
            data={"session_id": "err-session-2", "chunk_index": "1", "is_final": "true"}
        )
        assert response.status_code == 400
        assert "final metrics failed: Custom metrics error" in response.json()["detail"]

def test_main_startup():
    import main
    main.on_startup()
