import pytest
from unittest.mock import patch
from transcribe import transcription


def test_transcription_file_not_found():
    """Test that transcription returns empty string if file does not exist."""
    with patch("os.path.exists", return_value=False):
        assert transcription("nonexistent.mp3") == ""


def test_transcription_empty_file():
    """Test that transcription returns empty string if file is empty."""
    with patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=0):
        assert transcription("empty.mp3") == ""


def test_transcription_os_error():
    """Test that OSError during file checks is handled."""
    with patch("os.path.exists", side_effect=OSError("Disk error")):
        assert transcription("error.mp3") == ""


@patch("transcribe.model")
def test_transcription_success(mock_model):
    """Test successful transcription call."""
    with patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000):
        
        mock_model.transcribe.return_value = {"text": "Success"}
        result = transcription("audio.mp3")
        assert result == {"text": "Success"}


@patch("transcribe.model")
def test_transcription_runtime_error_failed_load(mock_model):
    """Test RuntimeError related to loading audio is handled."""
    with patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000):
        
        mock_model.transcribe.side_effect = RuntimeError("Failed to load audio")
        assert transcription("corrupt.mp3") == ""


@patch("transcribe.model")
def test_transcription_runtime_error_other(mock_model):
    """Test that unrelated RuntimeErrors are raised."""
    with patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000):
        
        mock_model.transcribe.side_effect = RuntimeError("Other error")
        with pytest.raises(RuntimeError):
            transcription("audio.mp3")