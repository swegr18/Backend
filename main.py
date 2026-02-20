from fastapi import FastAPI,UploadFile,File,HTTPException,Depends,Request
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session,select
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from pydub import AudioSegment
from infrastructure.config import settings
from infrastructure.container import container
from infrastructure.api.routes import health, audio, auth
from infrastructure.persistence.user_model import UserTable
from infrastructure.persistence.user_repository import PostgresUserRepository
from infrastructure.persistence.in_memory_repository import InMemoryRepository
import logging
import os,time
import subprocess
import librosa
import numpy as np
import whisper
import filler

#audio table initialisation
class AudioFile(SQLModel, table=True):
    __tablename__ = "audiofiles"
    __table_args__ = {"schema": "public"}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str
    content_type: str
    stored_filename: str 
    created_at:  datetime = Field(default_factory=datetime.utcnow)
    duration: Optional[float] = Field(default=None)
    avg_volume_dbfs: Optional[float] = Field(default=None)
    avg_pitch_hz: Optional[float] = Field(default=None)
    wpm: Optional[float] = Field(default=None)
    
#database linking    
DATABASE_URL = "postgresql+psycopg://postgres:postgres@db:5432/audiodb"
engine = create_engine(DATABASE_URL, echo=True)
def get_session():
    with Session(engine) as session:
        yield session    
    
app = FastAPI(
    title = "Hexagonal API",
    description = "FastAPI with Hexagonal Architecture",
    version = "1.0.0",
    debug = settings.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:8081", "http://127.0.0.1:8081", 
    "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
#logger to aid debugging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
model = whisper.load_model("tiny")

# Register routes
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#ensure database created 
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)
    
#audio receieved from react frontend--> store,convert,calculate,store
@app.post("/api/v1/upload-audio")
async def upload_and_store(request: Request, audio: UploadFile = File(...)):
    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")

    logger.info("Received upload filename=%s type=%s bytes=%d",
                audio.filename, audio.content_type, len(contents))
    
    
    form = await request.form()
    session_id = form.get("session_id") or "default"
    chunk_index = form.get("chunk_index") or "0"
    is_final = (form.get("is_final") == "true")
    session_dir = os.path.join(UPLOAD_DIR, "sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)
    combined_path = os.path.join(session_dir, "combined.webm")
    
    #  Save upload to disk
    file_id = uuid4()
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    with open(input_path, "wb") as f:
        f.write(contents)
    chunk_mp3_path = os.path.join(session_dir, f"chunk_{int(chunk_index):06d}.mp3")
    
    try:
        convert_to_mp3(input_path, chunk_mp3_path)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e.stderr[-500:]}")    
        
    try:
        chunk_audio = AudioSegment.from_file(input_path, format="webm")
        if os.path.exists(combined_path):
            combined_audio = AudioSegment.from_file(combined_path, format="webm")
            combined_audio = combined_audio + chunk_audio
        else:
            combined_audio = chunk_audio

        combined_audio.export(combined_path, format="webm")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"pydub combine failed: {e}")
   

    output_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
    
    if is_final:
        try:
            convert_to_mp3(combined_path, output_path)
            metrics = all_metrics(output_path)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e.stderr[-500:]}")

        #  Store in Postgres via SQLModel
        row = AudioFile(
            id = file_id,
            filename = audio.filename or "upload",
            content_type = "audio/mpeg",
            stored_filename = f"{file_id}.mp3",
            duration = metrics["duration"],
            avg_volume_dbfs = metrics["avg_volume_dbfs"],
            avg_pitch_hz = metrics["avg_pitch_hz"],
            wpm = metrics["wpm"]
            )

        with Session(engine) as session:
            session.add(row)
            session.commit()
        return {"ok": True, "final": True}
    return {"ok": True, "final": False}

def wait_for_file(path: str, min_bytes: int = 1024, timeout: float = 0.4) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
                return True
        except OSError:
            pass
        time.sleep(0.02)
    return False

#send metrics from database to frontend
@app.get("/api/v1/metrics/latest")
async def get_metrics(db: Session = Depends(get_session)):
    newest = db.exec(
        select(AudioFile).order_by(AudioFile.created_at.desc())
    ).first()

    if not newest:
        raise HTTPException(status_code=404, detail="No audio files found")
    return{"duration":newest.duration,"avg_volume_dbfs":newest.avg_volume_dbfs,"avg_pitch_hz":newest.avg_pitch_hz,"wpm":newest.wpm}

#send live wpm to frontend 
@app.get("/api/v1/live-wpm")
async def get_live_wpm(session_id: str, chunk_index: int):
    chunk_mp3_path = os.path.join(
        UPLOAD_DIR, "sessions", session_id, f"chunk_{chunk_index:06d}.mp3"
    )

    if not wait_for_file(chunk_mp3_path):
        logger.info("live-wpm not ready: %s", chunk_mp3_path)
        return {"wpm": None, "ready": False, "chunk_index": chunk_index}

    wpm_live = calc_wpm_live(chunk_mp3_path)
    wpm_live["ready"] = True
    wpm_live["chunk_index"] = chunk_index
    return wpm_live

#convert webm (audiovisual) to mp3(audio)
def convert_to_mp3(input_path: str, output_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-b:a", "192k", output_path],
        check = True,
        capture_output = True,
        text = True,
    )
    
def calc_wpm_live(path):
    text=transcription(path)
    word_count = len(text.split())
    wpm = word_count/(2/60)
    wpm=np.round(wpm,2)
    logger.info(f"wpm{wpm}")
    return {"wpm": wpm}

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

    # Filler proportion
    fillerProportion = filler.calculateFillerProportion(text)
    
    #logging
    logger.info("METRICS SHOWN HERE")
    logger.info(f"duration:{duration}")
    logger.info(f"average volume:{average_db}")
    logger.info(f"average freq:{avg_freq}")
    logger.info(f"wpm{wpm}")
    logger.info(f"frequencies:{avg_freq_second}")
    logger.info(f"filler proportion:{fillerProportion}")
    return {"duration":duration,"avg_volume_dbfs":average_db,"avg_pitch_hz":avg_freq,"wpm":wpm}    
    
# Initialize dependencies (Dependency Injection)
def _setup_dependencies():
    """Setup dependency injection container"""
    # Register repositories
    container.register("repository", InMemoryRepository())
    container.register("user_repository", PostgresUserRepository(engine))

_setup_dependencies()

app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(audio.router, prefix=settings.api_prefix, tags=["audio"])
app.include_router(auth.router, prefix=settings.api_prefix, tags=["auth"])
