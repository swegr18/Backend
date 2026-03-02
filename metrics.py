"""metric calculations"""

import logging
import numpy as np
import librosa
from pydub import AudioSegment
import filler
from transcribe import transcription



logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

#calculate numerical metrics
def all_metrics(path):
    """metric calculations for post recording display"""
    y, sr = librosa.load(path,sr=None)
    duration = len(y) / sr
    #trim background noise
    y, _ = librosa.effects.trim(y)
    #volume calculations
    rms = librosa.feature.rms(y=y)
    db = librosa.amplitude_to_db(rms, ref=1.0)
    average_db = float(np.mean(db))
    average_db = np.round(average_db,1)
    #wpm
    text=transcription(path)
    word_count = len(text.split())
    wpm = word_count/(duration/60)
    wpm = np.round(wpm,1)
    #pitch calculations
    f0 = librosa.yin(y,fmin=80,fmax=400)
    avg_freq = float(np.round(np.nanmean(f0), 1))
    #filler words
    filler_proportion = filler.calculateFillerProportion(text)
    #logging
    logger.info("METRICS SHOWN HERE")
    logger.info("duration:%s",duration)
    logger.info("average volume:%s",average_db)
    logger.info("average freq:%s",avg_freq)
    logger.info("wpm:%s",wpm)
    logger.info("filler proportion:%s",filler_proportion)
    return {"duration":duration,"avg_volume_dbfs":average_db,"avg_pitch_hz":avg_freq,"wpm":wpm}

def graph_metrics(path):
    """calculates arrays of data needed for graphs in stats page)"""
    y, sr = librosa.load(path)
    y, _ = librosa.effects.trim(y)
    y_harmonic, _ = librosa.effects.hpss(y)

    hop_length = 512
    f0 = librosa.yin(y_harmonic, fmin=80, fmax=400, sr=sr, hop_length=hop_length)

    times = librosa.frames_to_time(
        np.arange(len(f0)), 
        sr=sr, 
        hop_length=hop_length
    )

    valid = ~np.isnan(f0)
    f0 = f0[valid]
    times = times[valid]

    seconds = np.floor(times).astype(int)

    avg_freq_second = [
        float(np.mean(f0[seconds == sec]))
        for sec in range(seconds.max() + 1)
    ]

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0)

    volume_times = librosa.frames_to_time(
        np.arange(len(db)),
        sr=sr,
        hop_length=hop_length
    )

    volume_seconds = np.floor(volume_times).astype(int)

    avg_db_second = [
        float(np.mean(db[volume_seconds == sec]))
        for sec in range(volume_seconds.max() + 1)
    ]

    return {"frequencies": avg_freq_second,"volume_db": avg_db_second}

def calc_wpm_live(session_wpm,session_lock,session_id: str, chunk_index: int, chunk_mp3_path: str):
    """calculates averaged wpm for each session"""
    # chunk duration
    try:
        seg = AudioSegment.from_file(chunk_mp3_path)
        duration_s = len(seg) / 1000.0
    except Exception:
        duration_s = 0.0

    # transcription -> word count
    text = transcription(chunk_mp3_path) or ""
    words = len(text.split())

    with session_lock:
        st = session_wpm.get(session_id)
        if st is None:
            st = {"total_words": 0, "total_seconds": 0.0, "last_chunk": -1, "running_wpm": None}
            session_wpm[session_id] = st

        expected = st["last_chunk"] + 1

        # reject out-of-order chunks (your uploads are currently out-of-order)
        if chunk_index != expected:
            return {
                "accepted": False,
                "expected_chunk": expected,
                "last_chunk": st["last_chunk"],
                "running_wpm": st["running_wpm"],
            }

        st["total_words"] += words
        st["total_seconds"] += max(duration_s, 0.0)
        st["last_chunk"] = chunk_index

        if st["total_seconds"] > 0:
            st["running_wpm"] = float(np.round(st["total_words"] / (st["total_seconds"] / 60.0), 0))
        else:
            st["running_wpm"] = None

        return {
            "accepted": True,
            "chunk_words": words,
            "chunk_seconds": float(np.round(duration_s, 3)),
            "total_words": st["total_words"],
            "total_seconds": float(np.round(st["total_seconds"], 3)),
            "running_wpm": st["running_wpm"],
        }
