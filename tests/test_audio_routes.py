import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.api.routes import audio as audio_module
from infrastructure.api.routes.audio import router


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset module-level globals between tests."""
    audio_module.CURRENT_USER_ID = None
    audio_module.CURRENT_FILENAME = None
    audio_module.SESSION_WPM.clear()
    yield


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class FakeAudioRow:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid.uuid4())
        self.user_id = kwargs.get("user_id", uuid.uuid4())
        self.filename = kwargs.get("filename", "test.mp3")
        self.content_type = kwargs.get("content_type", "audio/mpeg")
        self.stored_filename = kwargs.get("stored_filename", "x.mp3")
        self.created_at = kwargs.get("created_at", datetime.utcnow())
        self.duration = kwargs.get("duration", 10.0)
        self.avg_volume_dbfs = kwargs.get("avg_volume_dbfs", -20.0)
        self.avg_pitch_hz = kwargs.get("avg_pitch_hz", 150.0)
        self.wpm = kwargs.get("wpm", 120.0)
        self.context_mode = kwargs.get("context_mode", "interview")
        self.graph_volume = kwargs.get("graph_volume", [-20.0, -18.0])
        self.graph_freq = kwargs.get("graph_freq", [150.0, 160.0])


def test_userdata_sets_globals(client):
    uid = str(uuid.uuid4())
    resp = client.post("/userdata", params={"user_id": uid, "filename": "demo.webm"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["user_id"] == uid
    assert body["filename"] == "demo.webm"


def test_live_wpm_session_not_found(client):
    resp = client.get("/live-wpm", params={"session_id": "nosession"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False


def test_live_wpm_session_found(client):
    audio_module.SESSION_WPM["sess1"] = {
        "running_wpm": 130.0,
        "last_chunk": 2,
        "total_words": 65,
        "total_seconds": 30.0,
    }
    resp = client.get("/live-wpm", params={"session_id": "sess1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["running_wpm"] == 130.0
    assert body["total_words"] == 65


def test_metrics_latest_no_data():
    fake_session = MagicMock()
    fake_exec = MagicMock()
    fake_exec.first.return_value = None
    fake_session.exec.return_value = fake_exec

    def override_session():
        yield fake_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[audio_module.get_session] = override_session
    c = TestClient(app)
    resp = c.get("/metrics/latest")
    assert resp.status_code == 404


def test_metrics_latest_success():
    row = FakeAudioRow()
    fake_session = MagicMock()
    fake_exec = MagicMock()
    fake_exec.first.return_value = row
    fake_session.exec.return_value = fake_exec

    def override_session():
        yield fake_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[audio_module.get_session] = override_session
    c = TestClient(app)

    resp = c.get("/metrics/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["duration"] == 10.0
    assert body["wpm"] == 120.0
    assert body["context_mode"] == "interview"


    
def test_graphs_returns_data():
    """Should return graph data for a user."""
    uid = uuid.uuid4()
    rows = [FakeAudioRow(user_id=uid), FakeAudioRow(user_id=uid)]

    fake_session = MagicMock()
    fake_exec = MagicMock()
    fake_exec.all.return_value = rows
    fake_session.exec.return_value = fake_exec

    def override_session():
        yield fake_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[audio_module.get_session] = override_session
    c = TestClient(app)

    resp = c.post("/graphs", params={"user_id": str(uid)})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert "graph_volume" in body[0]
    assert "graph_freq" in body[0]


def test_upload_audio_empty(client):
    """Empty upload should return 400."""
    resp = client.post(
        "/upload-audio",
        files={"audio": ("empty.webm", b"", "audio/webm")},
    )
    assert resp.status_code == 400
    assert "Empty upload" in resp.json()["detail"]


@patch("infrastructure.api.routes.audio.AudioSegment")
@patch("infrastructure.api.routes.audio.calc_wpm_live")
@patch("infrastructure.api.routes.audio.convert_to_mp3")
def test_upload_audio_non_final_chunk(mock_convert, mock_wpm, mock_pydub, client):
    mock_wpm.return_value = {"accepted": True, "running_wpm": 100.0, "chunk_words": 10,
                             "chunk_seconds": 3.0, "total_words": 10, "total_seconds": 3.0}
    mock_segment = MagicMock()
    mock_pydub.from_file.return_value = mock_segment

    resp = client.post(
        "/upload-audio",
        files={"audio": ("chunk.webm", b"fake-audio-bytes", "audio/webm")},
        data={"session_id": "s1", "chunk_index": "0", "is_final": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["final"] is False
    assert body["ok"] is True


@patch("infrastructure.api.routes.audio.Session")
@patch("infrastructure.api.routes.audio.graph_metrics")
@patch("infrastructure.api.routes.audio.all_metrics")
@patch("infrastructure.api.routes.audio.AudioSegment")
@patch("infrastructure.api.routes.audio.calc_wpm_live")
@patch("infrastructure.api.routes.audio.convert_to_mp3")
def test_upload_audio_final_chunk_success(
    mock_convert, mock_wpm, mock_pydub, mock_all_metrics, mock_graph, mock_db_session, client
):
    audio_module.CURRENT_USER_ID = uuid.uuid4()

    mock_wpm.return_value = {"accepted": True, "running_wpm": 120.0, "chunk_words": 20,
                             "chunk_seconds": 5.0, "total_words": 20, "total_seconds": 5.0}
    mock_segment = MagicMock()
    mock_pydub.from_file.return_value = mock_segment

    mock_all_metrics.return_value = {
        "duration": 10.0, "avg_volume_dbfs": -20.0, "avg_pitch_hz": 150.0, "wpm": 120.0,
    }
    mock_graph.return_value = {"volume_db": [-20.0], "frequencies": [150.0]}

    # Mock the DB Session context manager
    session_instance = MagicMock()
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session_instance)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

    resp = client.post(
        "/upload-audio",
        files={"audio": ("chunk.webm", b"fake-audio-bytes", "audio/webm")},
        data={"session_id": "s2", "chunk_index": "0", "is_final": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["final"] is True
    assert body["final_metrics"]["wpm"] == 120.0


@patch("infrastructure.api.routes.audio.AudioSegment")
@patch("infrastructure.api.routes.audio.calc_wpm_live")
@patch("infrastructure.api.routes.audio.convert_to_mp3")
def test_upload_audio_final_no_user_set(mock_convert, mock_wpm, mock_pydub, client):
    mock_wpm.return_value = {"accepted": True, "running_wpm": 100.0, "chunk_words": 10,
                             "chunk_seconds": 3.0, "total_words": 10, "total_seconds": 3.0}
    mock_segment = MagicMock()
    mock_pydub.from_file.return_value = mock_segment

    mock_all_metrics_patch = patch(
        "infrastructure.api.routes.audio.all_metrics",
        return_value={"duration": 5, "avg_volume_dbfs": -20, "avg_pitch_hz": 150, "wpm": 100},
    )
    mock_graph_patch = patch(
        "infrastructure.api.routes.audio.graph_metrics",
        return_value={"volume_db": [], "frequencies": []},
    )
    with mock_all_metrics_patch, mock_graph_patch:
        resp = client.post(
            "/upload-audio",
            files={"audio": ("chunk.webm", b"fake-audio-bytes", "audio/webm")},
            data={"session_id": "s3", "chunk_index": "0", "is_final": "true"},
        )
    assert resp.status_code == 400
    assert "User not set" in resp.json()["detail"]
