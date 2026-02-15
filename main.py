from fastapi import FastAPI,UploadFile,File,HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session,select
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime

from infrastructure.config import settings
from infrastructure.container import container
from infrastructure.api.routes import health, audio
from infrastructure.persistence.in_memory_repository import InMemoryRepository
import logging
import os
import subprocess
import librosa
import numpy as np
import whisper

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
    title="Hexagonal API",
    description="FastAPI with Hexagonal Architecture",
    version="1.0.0",
    debug=settings.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#logger to aid debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = whisper.load_model("base")

# Register routes
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#ensure database created 
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)
    
#audio receieved from react frontend--> store,convert,calculate,store
@app.post("/api/v1/upload-audio")
async def upload_and_store(audio: UploadFile = File(...)):
    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")

    logger.info("Received upload filename=%s type=%s bytes=%d",
                audio.filename, audio.content_type, len(contents))

    # 1) Save upload to disk
    file_id = uuid4()
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    with open(input_path, "wb") as f:
        f.write(contents)

    # 2) Convert to MP3 & calculate metrics
    output_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
    try:
        convert_to_mp3(input_path, output_path)
        metrics=all_metrics(output_path)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e.stderr[-500:]}")



    # 3) Store in Postgres via SQLModel
    row = AudioFile(
        id=file_id,
        filename=audio.filename or "upload",
        content_type="audio/mpeg",
        stored_filename=f"{file_id}.mp3",
        duration=metrics["duration"],
        avg_volume_dbfs=metrics["avg_volume_dbfs"],
        avg_pitch_hz=metrics["avg_pitch_hz"],
        wpm=metrics["wpm"]
        )

    with Session(engine) as session:
        session.add(row)
        session.commit()

#send wpm from database to frontend
@app.get("/api/v1/metrics/latest")
async def get_metrics(db: Session = Depends(get_session)):
    newest = db.exec(
        select(AudioFile).order_by(AudioFile.created_at.desc())
    ).first()

    if not newest:
        raise HTTPException(status_code=404, detail="No audio files found")
    return{"duration":newest.duration,"avg_volume_dbfs":newest.avg_volume_dbfs,"avg_pitch_hz":newest.avg_pitch_hz,"wpm":newest.wpm}

#convert webm (audiovisual) to mp3(audio)
def convert_to_mp3(input_path: str, output_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-b:a", "192k", output_path],
        check=True,
        capture_output=True,
        text=True,
    )
    
def calc_wpm(path,text):
    
    y, sr = librosa.load(path)
    duration=librosa.get_duration(y=y,sr=sr)
    word_count = len(text.split())
    wpm=word_count/(duration/60)
    return {"wpm": wpm}

#use whisper to get text transcription
def transcription(path):
    result = model.transcribe(path)
    text = result["text"]  
    return text

#calculate numerical metrics
def all_metrics(path):
    y, sr = librosa.load(path)
    duration=librosa.get_duration(y=y,sr=sr)
    #volume calculations
    rms = librosa.feature.rms(y=y)
    db = librosa.amplitude_to_db(rms, ref=1.0)
    average_db = float(np.mean(db))
    #wpm 
    text=transcription(path)
    word_count = len(text.split())
    wpm=word_count/(duration/60)
    #pitch calculations and background noise trimming
    y, _ = librosa.effects.trim(y)
    y_harmonic, _ = librosa.effects.hpss(y)
    f0 = librosa.yin(y_harmonic,fmin=80,fmax=400)
    f0 = f0[~np.isnan(f0)]
    avg_freq = float(np.mean(f0))
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
    
# Initialize dependencies (Dependency Injection)
def _setup_dependencies():
    """Setup dependency injection container"""
    # Register repositories
    container.register("repository", InMemoryRepository())

_setup_dependencies()

app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(audio.router, prefix=settings.api_prefix, tags=["audio"])
