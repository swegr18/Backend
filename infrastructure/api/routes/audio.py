"""audio routes"""
import os
import subprocess
import logging
import threading
from uuid import uuid4,UUID

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, status
from pydub import AudioSegment
from sqlmodel import Session, select
from infrastructure.security import get_current_user_id
from infrastructure.persistence.user_repository import PostgresUserRepository
from infrastructure.persistence.user_model import UserTable
from infrastructure.persistence.database import get_session, engine
from infrastructure.persistence.audio_model import AudioFile
from metrics import calc_wpm_live, all_metrics, graph_metrics

router = APIRouter()
logger = logging.getLogger(__name__)
CURRENT_USER_ID = None
CURRENT_FILENAME = None
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_WPM = {}
SESSION_LOCK = threading.Lock()


def convert_to_mp3(input_path: str, output_path: str) -> None:
    """convert webm (audiovisual) to mp3(audio)"""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-b:a", "192k", output_path],
        check=True,
        capture_output=True,
        text=True,
    )
def get_current_user(
    user_id: str = Depends(get_current_user_id),
) -> UserTable:
    repo = PostgresUserRepository(engine)
    user = repo.find_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
@router.post("/upload-audio")
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
    context_mode = form.get("context_mode")  

    session_dir = os.path.join(UPLOAD_DIR, "sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)
    combined_path = os.path.join(session_dir, "combined.webm")

    #  save upload to disk
    file_id = uuid4()
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    with open(input_path, "wb") as f:
        f.write(contents)

    chunk_mp3_path = os.path.join(session_dir, f"chunk_{int(chunk_index):06d}.mp3")

    try:
        convert_to_mp3(input_path, chunk_mp3_path)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e.stderr[-500:]}") from e
    running = calc_wpm_live(SESSION_WPM, SESSION_LOCK, session_id, int(chunk_index), chunk_mp3_path)
    # combine chunks
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


    #  convert to MP3
    output_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")

    if not is_final:
        return {"ok": True, "final": False, "chunk_index": chunk_index, **running}

    # compute full metrics on combined audio and commit to db
    try:
        output_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
        convert_to_mp3(combined_path, output_path)
        metrics = all_metrics(output_path)
        graph_data=graph_metrics(output_path)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e.stderr[-500:]}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"final metrics failed: {e}") from e


    row = AudioFile(
        id=file_id,
        user_id=TEST_USER_ID,
        filename=audio.filename or "upload",
        content_type="audio/mpeg",
        stored_filename=f"{file_id}.mp3",
        duration=metrics["duration"],
        avg_volume_dbfs=metrics["avg_volume_dbfs"],
        avg_pitch_hz=metrics["avg_pitch_hz"],
        wpm=metrics["wpm"],
        context_mode=context_mode,
        graph_volume=graph_data["volume_db"],
        graph_freq=graph_data["frequencies"],
    )

    with Session(engine) as session:
        session.add(row)
        session.commit()


    with SESSION_LOCK:
        SESSION_WPM.pop(session_id, None)

    return {"ok": True, "final": True, "chunk_index": chunk_index,
            "running": running, "final_metrics": metrics}


@router.get("/metrics/latest")
async def get_metrics(db: Session = Depends(get_session)):
    """send metrics from database to frontend"""
    newest = db.exec(
        select(AudioFile).order_by(AudioFile.created_at.desc())
    ).first()

    if not newest:
        raise HTTPException(status_code=404, detail="No audio files found")
    return {"duration": newest.duration, "avg_volume_dbfs": newest.avg_volume_dbfs,
            "avg_pitch_hz": newest.avg_pitch_hz, "wpm": newest.wpm,
            "context_mode": newest.context_mode}


@router.get("/live-wpm")
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
    

@router.post("/graphs")
async def get_graph_data(
    
    user_id: UUID,
    db: Session = Depends(get_session),
):
    """send the graph data to frontend"""
    rows = db.exec(
        select(AudioFile)
        .where(AudioFile.user_id == user_id)
        .order_by(AudioFile.created_at.desc())
    ).all()

    return [
        {
            "audio_id": a.id,
            "created_at": a.created_at,
            "graph_volume": a.graph_volume or [],
            "graph_freq": a.graph_freq or [],
        }
        for a in rows
    ]
    
@router.post("/userdata")
async def send_user_data(
    user_id: UUID,
    filename: str,
    db: Session = Depends(get_session),
):
    """Receive user data from frontend and update audio row"""

    audio = db.exec(
        select(AudioFile).where(AudioFile.stored_filename == filename)
    ).first()

    if not audio:
        raise HTTPException(status_code=404, detail="Audio file not found")

    
    audio.user_id = user_id

    db.add(audio)
    db.commit()
    db.refresh(audio)

    return {
        "ok": True,
        "audio_id": str(audio.id),
        "user_id": str(user_id),
        "filename": filename
    }