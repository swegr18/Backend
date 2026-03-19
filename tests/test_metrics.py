import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from metrics import all_metrics, graph_metrics, calc_wpm_live, calculate_transcribability


@pytest.fixture
def mock_transcription():
    """Fixture to mock the transcription function."""
    mock_output = {
        "text": "This is a test transcription",
        "segments": [
            {"avg_logprob": -0.5},
            {"avg_logprob": -0.3},
        ]
    }
    with patch("metrics.transcription", return_value=mock_output) as mock:
        yield mock


@pytest.fixture
def mock_librosa():
    """Fixture to mock librosa functions."""
    with patch("metrics.librosa") as mock:
        mock.load.return_value = (np.zeros(16000), 16000)  # 1 second audio
        mock.get_duration.return_value = 1.0
        mock.feature.rms.return_value = np.array([[0.1]])
        mock.amplitude_to_db.return_value = np.array([-20.0])
        mock.yin.return_value = np.array([150.0, 155.0, np.nan])
        yield mock


def test_all_metrics(mock_transcription, mock_librosa):
    """Test the all_metrics function."""
    path = "dummy/path.mp3"
    metrics = all_metrics(path)

    assert "duration" in metrics
    assert "avg_volume_dbfs" in metrics
    assert "avg_pitch_hz" in metrics
    assert "wpm" in metrics
    assert "filler_proportion" in metrics
    assert "transcribability" in metrics
    assert metrics["wpm"] == 300.0  # 5 words / 1 second * 60
    assert metrics["filler_proportion"] == 0.0
    assert metrics["transcribability"] == pytest.approx(0.449329, abs=1e-6)


def test_graph_metrics(mock_librosa):
    """Test the graph_metrics function."""
    # Further mocking for graph-specific functions
    mock_librosa.effects.trim.return_value = (np.zeros(16000), 16000)
    mock_librosa.effects.hpss.return_value = (np.zeros(16000), np.zeros(16000))
    mock_librosa.frames_to_time.side_effect = [np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1])]
    mock_librosa.feature.rms.return_value = np.array([[0.1, 0.2]])
    mock_librosa.amplitude_to_db.return_value = np.array([-20.0, -15.0])

    path = "dummy/path.mp3"
    graph_data = graph_metrics(path)

    assert "frequencies" in graph_data
    assert "volume_db" in graph_data
    assert isinstance(graph_data["frequencies"], list)
    assert isinstance(graph_data["volume_db"], list)


def test_calculate_transcribability():
    """Test the calculate_transcribability helper."""
    whisper_output = {"segments": [{"avg_logprob": -0.693147}]}  # approx log(0.5)
    # math.exp(-0.693147) is approx 0.5
    assert calculate_transcribability(whisper_output) == pytest.approx(0.5, abs=1e-4)


@patch("metrics.AudioSegment")
def test_calc_wpm_live(mock_audio_segment):
    """Test the live WPM calculation logic."""
    # Mock pydub
    mock_seg = MagicMock()
    mock_seg.__len__.return_value = 2000  # 2 seconds
    mock_audio_segment.from_file.return_value = mock_seg

    # Mock transcription
    mock_transcription_output = {"text": "one two three four"}
    with patch("metrics.transcription", return_value=mock_transcription_output):
        session_wpm = {}
        session_lock = MagicMock()
        session_id = "test-session"

        result1 = calc_wpm_live(session_wpm, session_lock, session_id, 0, "chunk1.mp3")
        assert result1["accepted"] is True
        assert result1["running_wpm"] == 120.0

        result2 = calc_wpm_live(session_wpm, session_lock, session_id, 1, "chunk2.mp3")
        assert result2["accepted"] is True
        assert result2["running_wpm"] == 120.0
