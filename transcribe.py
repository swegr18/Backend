import os
import logging
import whisper
import filler

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
        result = model.transcribe(path, fp16=False, initial_prompt=" ".join(filler.VOCABLE_FILLERS))
        return (result.get("text"))
    except RuntimeError as e:
        msg = str(e)
        if "Failed to load audio" in msg:
            logger.warning("Whisper couldn't decode %s (treating as empty). %s", path, msg)
            return ""
        raise
    
    