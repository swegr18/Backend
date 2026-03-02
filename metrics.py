import librosa
import logging
import filler
import numpy as np
import math

from transcribe import transcription
from pydub import AudioSegment


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

#calculate numerical metrics
def all_metrics(path):

    whisper_output = transcription(path)
    text = whisper_output.get("text")
    y, sr = librosa.load(path)
    duration = librosa.get_duration(y=y,sr=sr)

    #volume calculations
    average_db = calculate_average_volume(y)

    #wpm
    wpm = calculate_wpm(text, duration)

    #pitch calculations and background noise trimming
    avg_freq, avg_freq_second = calculate_avg_pitch(y, sr)

    #filler words
    filler_proportion = filler.calculate_filler_proportion(text)

    # Proportion of transcribability
    transcribability = calculate_transcribability(whisper_output)

    #logging
    logger.info("METRICS SHOWN HERE")
    logger.info(f"duration:{duration}")
    logger.info(f"average volume:{average_db}")
    logger.info(f"average freq:{avg_freq}")
    logger.info(f"wpm{wpm}")
    logger.info(f"frequencies:{avg_freq_second}")
    logger.info(f"filler proportion:{filler_proportion}")
    logger.info(f"proportion of transcribability:{transcribability}")
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

# Calculate proportion of transcribable text (Whisper's confidence in the transcription)
#
# Input: Dict returned by model.transcribe()
# Output: Float Proportion of transcribable text
#
def calculate_transcribability(whisper_output):
    segments = whisper_output.get("segments")

    totalLog = 0

    for seg in segments:
        totalLog += seg.get("avg_logprob")

    return math.exp(totalLog)
        
# Calculate average volume in deciBels
#
# Input: Librosa audio time series (np.ndarray)
# Output: Float, in deciBels
#
def calculate_average_volume(y):
    rms = librosa.feature.rms(y=y)
    db = librosa.amplitude_to_db(rms, ref=1.0)
    average_db = float(np.mean(db))
    return np.round(average_db,1)

# Calculate words per minute
#
# Input: text string, duration as float
# Output: Float, words per minute
#
def calculate_wpm(text, duration):
    word_count = len(text.split())
    wpm = word_count/(duration/60)
    return np.round(wpm,1)

# Calculate average pitch
#
# Input: Librosa time series, sample rate
# Output: Tuple of average pitch frequency and average frequency per second?
#
def calculate_avg_pitch(y, sr):
    y, _ = librosa.effects.trim(y)
    y_harmonic, _ = librosa.effects.hpss(y)
    f0 = librosa.yin(y_harmonic,fmin=80,fmax=400)
    f0 = f0[~np.isnan(f0)]
    avg_freq = float(np.mean(f0))
    avg_freq = np.round(avg_freq,1)
    frames_per_second = int(len(f0) / librosa.get_duration(y=y, sr=sr))
    chunks = np.array_split(f0, frames_per_second)
    avg_freq_second = [np.mean(chunk) for chunk in chunks]

    return avg_freq, avg_freq_second
    