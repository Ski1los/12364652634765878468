"""HTTP API and single-page UI for ClipLab Local."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .jobs import JobManager
from .pipeline import PipelineError, validate_source_url
from .settings import Settings, load_settings

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"


class CreateStory(BaseModel):
    topic: str = Field(min_length=10, max_length=900)
    story_kind: Literal["fictional", "inspiration", "curiosity"] = "fictional"
    language: Literal["ru", "en"] = "ru"
    duration_seconds: Literal[30, 45, 60, 90] = 45
    visual_style: Literal["cinematic", "comic", "minimal"] = "cinematic"
    render_height: Literal[720, 1080] = 720
    ai_disclosure_confirmed: bool = False

    @model_validator(mode="after")
    def check_disclosure(self) -> "CreateStory":
        self.topic = " ".join(self.topic.split())
        if not self.ai_disclosure_confirmed:
            raise ValueError("Подтвердите маркировку ролика как созданного с помощью ИИ.")
        return self


class CreateJob(BaseModel):
    source_url: str = Field(min_length=12, max_length=2000)
    clip_count: int = Field(default=3, ge=1, le=8)
    min_seconds: int = Field(default=20, ge=10, le=90)
    max_seconds: int = Field(default=50, ge=15, le=150)
    render_height: Literal[720, 1080] = 720
    rights_confirmed: bool = False

    @model_validator(mode="after")
    def check_duration_range(self) -> "CreateJob":
        if self.max_seconds <= self.min_seconds:
            raise ValueError("Максимальная длительность должна быть больше минимальной.")
        if not self.rights_confirmed:
            raise ValueError("Подтвердите, что у вас есть права на обработку видео.")
        return self


settings: Settings
manager: JobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, manager
    settings = load_settings()
    manager = JobManager(settings)
    yield


app = FastAPI(
    title="ClipLab Local",
    version="1.0.0",
    description="Free, self-hosted local video highlights with burned captions.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, object]:
    return {
        "whisper_model": settings.whisper_model,
        "local_llm_enabled": bool(settings.local_llm_url),
        "piper_voice_configured": bool(settings.piper_voice_model),
        "max_workers": settings.max_workers,
        "supported_sources": ["YouTube", "Twitch VOD / clips"],
    }


@app.post("/api/jobs", status_code=202)
def create_job(payload: CreateJob) -> dict:
    try:
        payload.source_url = validate_source_url(payload.source_url)
    except PipelineError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return manager.create(payload.model_dump(exclude={"rights_confirmed"}))


@app.post("/api/stories", status_code=202)
def create_story(payload: CreateStory) -> dict:
    return manager.create_story(payload.model_dump(exclude={"ai_disclosure_confirmed"}))


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return manager.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return manager.public_job(job)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, bool]:
    deleted = manager.delete(job_id)
    if not deleted:
        raise HTTPException(status_code=409, detail="Активную или отсутствующую задачу удалить нельзя.")
    return {"deleted": True}


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_file(job_id: str, filename: str) -> FileResponse:
    path = manager.file_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Файл не найден.")
    media_type = {
        ".mp4": "video/mp4", ".wav": "audio/wav", ".srt": "application/x-subrip",
        ".txt": "text/plain; charset=utf-8", ".json": "application/json",
    }.get(path.suffix.lower(), "application/octet-stream")
    # No Content-Disposition: attachment here: the same endpoint is used by <video>.
    # The UI's explicit download links carry the HTML `download` attribute.
    return FileResponse(path, media_type=media_type)
