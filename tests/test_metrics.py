from types import SimpleNamespace

import numpy as np

import metrics


def test_all_metrics_uses_calculated_helpers(monkeypatch):
    """
    Unit test for metrics.all_metrics(), focused on wiring:
    - stubs out heavy dependencies (transcription, librosa, filler)
    - verifies the returned dict contains expected keys and values.
    """

    def fake_transcription(path):
        return {
            "text": "hello world from test",
            "segments": [
                {"avg_logprob": -0.1},
                {"avg_logprob": -0.2},
            ],
        }

    def fake_librosa_load(path):
        # Simple 1D signal and sample rate
        y = np.ones(16000, dtype=float)
        sr = 16000
        return y, sr

    def fake_get_duration(y, sr):
        return 10.0  # seconds

    def fake_calculate_filler_proportion(text):
        return 0.1

    # Patch external functions used by metrics.all_metrics
    monkeypatch.setattr(metrics, "transcription", fake_transcription)
    monkeypatch.setattr(metrics.librosa, "load", fake_librosa_load)
    monkeypatch.setattr(metrics.librosa, "get_duration", fake_get_duration)
    monkeypatch.setattr(metrics, "calculate_filler_proportion", fake_calculate_filler_proportion, raising=False)

    # Also patch helpers to deterministic values so we don't depend on DSP behavior
    monkeypatch.setattr(metrics, "calculate_average_volume", lambda y: -20.0)
    monkeypatch.setattr(metrics, "calculate_wpm", lambda text, duration: 180.0)
    monkeypatch.setattr(metrics, "calculate_avg_pitch", lambda y: 150.0)
    monkeypatch.setattr(metrics, "calculate_transcribability", lambda output: 0.9)

    result = metrics.all_metrics("dummy-path.wav")

    assert set(result.keys()) == {
        "duration",
        "avg_volume_dbfs",
        "avg_pitch_hz",
        "wpm",
    }
    assert result["duration"] == 10.0
    assert result["avg_volume_dbfs"] == -20.0
    assert result["avg_pitch_hz"] == 150.0
    assert result["wpm"] == 180.0


