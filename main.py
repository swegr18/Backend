"""main module for FastAPI backend"""
import os
import subprocess

from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
import os
import subprocess
import threading
import logging
from pydub import AudioSegment
from fastapi import FastAPI,UploadFile,File,HTTPException,Depends,Request
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session,select

from infrastructure.config import settings
from infrastructure.container import container
from infrastructure.api.routes import health, audio
from infrastructure.persistence.in_memory_repository import InMemoryRepository
import logging

from metrics import calc_wpm_live,all_metrics

#audio table initialisation
class AudioFile(SQLModel, table=True):
    """Class representing audio table"""
    __tablename__ = "audiofiles"
    __table_args__ = {"schema": "public"}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str
    content_type: str
    stored_filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration: Optional[float] = Field(default=None)
    avg_volume_dbfs: Optional[float] = Field(default=None)
    avg_pitch_hz: Optional[float] = Field(default=None)
    wpm: Optional[float] = Field(default=None)

#database linking
DATABASE_URL = "postgresql+psycopg://postgres:postgres@db:5432/audiodb"
engine = create_engine(DATABASE_URL, echo=True)

def get_session():
    """subroutine to define session"""
    with Session(engine) as session:
        yield session

app = FastAPI(
    title="Hexagonal API",
    description="FastAPI with Hexagonal Architecture",
    version="1.0.0",
    debug=settings.debug
)
SESSION_WPM = {}
SESSION_LOCK = threading.Lock()

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

# Register routes
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def on_startup():
    """ensure database created at start"""
    SQLModel.metadata.create_all(engine)


@app.post("/api/v1/upload-audio")
async def upload_and_store(request: Request, audio: UploadFile = File(...)):
    """audio receieved from react frontend--> store,convert,calculate,store"""
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
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e.stderr[-500:]}") from e
    running = calc_wpm_live(SESSION_WPM,SESSION_LOCK,session_id, int(chunk_index), chunk_mp3_path)
    try:
        chunk_audio = AudioSegment.from_file(input_path, format="webm")
        if os.path.exists(combined_path):
            combined_audio = AudioSegment.from_file(combined_path, format="webm")
            combined_audio = combined_audio + chunk_audio
        else:
            combined_audio = chunk_audio
        combined_audio.export(combined_path, format="webm")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"pydub combine failed: {e}") from e


    # 2) Convert to MP3 & calculate metrics
    output_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")

    if not is_final:
        return {"ok": True, "final": False, "chunk_index": chunk_index, **running}

    # FINAL: compute full metrics on combined audio and commit ONE DB row 
    try:
        output_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
        convert_to_mp3(combined_path, output_path)
        metrics = all_metrics(output_path)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e.stderr[-500:]}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"final metrics failed: {e}")from e

    row = AudioFile(
        id=file_id,
        filename=audio.filename or "upload",
        content_type="audio/mpeg",
        stored_filename=f"{file_id}.mp3",
        duration=metrics["duration"],
        avg_volume_dbfs=metrics["avg_volume_dbfs"],
        avg_pitch_hz=metrics["avg_pitch_hz"],
        wpm=metrics["wpm"],
    )

    with Session(engine) as session:
        session.add(row)
        session.commit()


    with SESSION_LOCK:
        SESSION_WPM.pop(session_id, None)

    return {"ok": True, "final": True, "chunk_index": chunk_index,
            "running": running, "final_metrics": metrics}


@app.get("/api/v1/metrics/latest")
async def get_metrics(db: Session = Depends(get_session)):
    """send metrics from database to frontend"""
    newest = db.exec(
        select(AudioFile).order_by(AudioFile.created_at.desc())
    ).first()

    if not newest:
        raise HTTPException(status_code=404, detail="No audio files found")
    return{"duration":newest.duration,"avg_volume_dbfs":newest.avg_volume_dbfs,
           "avg_pitch_hz":newest.avg_pitch_hz,"wpm":newest.wpm}


@app.get("/api/v1/live-wpm")
async def get_live_wpm(session_id: str):
    """send live wpm to frontend """
    with SESSION_LOCK:
        st = SESSION_WPM.get(session_id)

    if not st:
        return {"ready": False, "running_wpm": None, "last_chunk": None}

    return {
        "ready": True,
        "running_wpm": st["running_wpm"],
        "last_chunk": st["last_chunk"],
        "total_words": st["total_words"],
        "total_seconds": round(st["total_seconds"], 2),
    }

def convert_to_mp3(input_path: str, output_path: str) -> None:
    """convert webm (audiovisual) to mp3(audio)"""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-b:a", "192k", output_path],
        check=True,
        capture_output=True,
        text=True,
    )


def _setup_dependencies():
    """Setup dependency injection container"""
    # Register repositories
    container.register("repository", InMemoryRepository())

_setup_dependencies()

app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(audio.router, prefix=settings.api_prefix, tags=["audio"])
