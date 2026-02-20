import librosa
import logging
import whisper
import numpy as np
import os
from pydub import AudioSegment


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
model = whisper.load_model("tiny")

#use whisper to get text transcription
def transcription(path):
    try:
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            return ""
    except OSError:
        return ""

    try:
        result = model.transcribe(path, fp16=False)
        return (result.get("text"))
    except RuntimeError as e:
        msg = str(e)
        if "Failed to load audio" in msg:
            logger.warning("Whisper couldn't decode %s (treating as empty). %s", path, msg)
            return ""
        raise
    
    
#calculate numerical metrics
def all_metrics(path):
    y, sr = librosa.load(path)
    duration = librosa.get_duration(y=y,sr=sr)
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
    #pitch calculations and background noise trimming
    y, _ = librosa.effects.trim(y)
    y_harmonic, _ = librosa.effects.hpss(y)
    f0 = librosa.yin(y_harmonic,fmin=80,fmax=400)
    f0 = f0[~np.isnan(f0)]
    avg_freq = float(np.mean(f0))
    avg_freq = np.round(avg_freq,1)
    frames_per_second = int(len(f0) / librosa.get_duration(y=y, sr=sr))
    chunks = np.array_split(f0, frames_per_second)
    avg_freq_second = [np.mean(chunk) for chunk in chunks]
    
    #logging
    logger.info("METRICS SHOWN HERE")
    logger.info(f"duration:{duration}")
    logger.info(f"average volume:{average_db}")
    logger.info(f"average freq:{avg_freq}")
    logger.info(f"wpm{wpm}")
    logger.info(f"frequencies:{avg_freq_second}")
    return {"duration":duration,"avg_volume_dbfs":average_db,"avg_pitch_hz":avg_freq,"wpm":wpm}    
    
    

def calc_wpm_live(session_wpm,session_lock,session_id: str, chunk_index: int, chunk_mp3_path: str):
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
            st["running_wpm"] = float(np.round(st["total_words"] / (st["total_seconds"] / 60.0), 2))
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

