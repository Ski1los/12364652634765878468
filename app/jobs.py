"""In-process background queue and durable job state for a single local instance."""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline import PipelineError, remove_job_files, run_pipeline
from .story import run_story_pipeline
from .settings import Settings
from .storage import read_json, write_json

ACTIVE_STATES = {"queued", "downloading", "transcribing", "selecting", "rendering", "drafting", "narrating", "composing"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="clipper")
        self._load_existing()

    def _job_path(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id / "job.json"

    def _save(self, job: dict[str, Any]) -> None:
        write_json(self._job_path(job["id"]), job)

    def _load_existing(self) -> None:
        for path in self.settings.jobs_dir.glob("*/job.json"):
            item = read_json(path)
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            if item.get("status") in ACTIVE_STATES:
                item["status"] = "failed"
                item["stage"] = "stopped"
                item["error"] = "Сервер был перезапущен во время обработки. Создайте задачу ещё раз."
                item.setdefault("logs", []).append({"at": now(), "message": item["error"]})
                item["updated_at"] = now()
                self._save(item)
            self._jobs[item["id"]] = item

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": "clip",
            "source_url": payload["source_url"],
            "clip_count": payload["clip_count"],
            "min_seconds": payload["min_seconds"],
            "max_seconds": payload["max_seconds"],
            "render_height": payload["render_height"],
            "cloud_processing_allowed": payload.get("cloud_processing_allowed", False),
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "error": None,
            "clips": [],
            "logs": [{"at": now(), "message": "Задача добавлена в локальную очередь."}],
            "created_at": now(),
            "updated_at": now(),
        }
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._executor.submit(self._work, job_id)
        return self.public_job(job)

    def create_story(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": "story",
            "topic": payload["topic"],
            "story_kind": payload["story_kind"],
            "language": payload["language"],
            "duration_seconds": payload["duration_seconds"],
            "visual_style": payload["visual_style"],
            "render_height": payload["render_height"],
            "cloud_processing_allowed": payload.get("cloud_processing_allowed", False),
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "error": None,
            "story": None,
            "logs": [{"at": now(), "message": "Story Reel добавлен в локальную очередь."}],
            "created_at": now(),
            "updated_at": now(),
        }
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._executor.submit(self._work, job_id)
        return self.public_job(job)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes)
            job["updated_at"] = now()
            self._save(job)

    def _log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.setdefault("logs", []).append({"at": now(), "message": message[-1600:]})
            # Keep state files small while retaining enough diagnostics for one job.
            job["logs"] = job["logs"][-120:]
            job["updated_at"] = now()
            self._save(job)

    def _set_stage(self, job_id: str, stage: str, progress: int) -> None:
        self._update(job_id, status=stage, stage=stage, progress=progress)

    def _work(self, job_id: str) -> None:
        def log(message: str) -> None:
            # Pipeline milestones determine progress; it is deliberately not fake per-frame progress.
            lower = message.lower()
            if "загружаю источник" in lower:
                self._set_stage(job_id, "downloading", 10)
            elif "распознаю речь" in lower:
                self._set_stage(job_id, "transcribing", 35)
            elif "ищу потенциально" in lower or "языковая модель" in lower:
                self._set_stage(job_id, "selecting", 58)
            elif "локальная llm пишет" in lower or "облачная llm пишет" in lower:
                self._set_stage(job_id, "drafting", 22)
            elif "озвучиваю" in lower:
                self._set_stage(job_id, "narrating", 52)
            elif "сторис-карточки" in lower or "вшиваю субтитры" in lower:
                self._set_stage(job_id, "composing", 76)
            elif "рендер " in lower:
                self._set_stage(job_id, "rendering", 72)
            self._log(job_id, message)

        try:
            with self._lock:
                snapshot = dict(self._jobs[job_id])
            if snapshot.get("kind") == "story":
                story = run_story_pipeline(snapshot, self.settings, log)
                self._update(job_id, status="completed", stage="completed", progress=100, story=story, error=None)
            else:
                clips = run_pipeline(snapshot, self.settings, log)
                self._update(job_id, status="completed", stage="completed", progress=100, clips=clips, error=None)
        except PipelineError as error:
            self._log(job_id, str(error))
            self._update(job_id, status="failed", stage="failed", progress=100, error=str(error))
        except Exception as error:  # Avoid a stuck job and retain traceback only in server logs.
            traceback.print_exc()
            message = f"Непредвиденная ошибка обработки: {error}"
            self._log(job_id, message)
            self._update(job_id, status="failed", stage="failed", progress=100, error=message)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            items = [self.public_job(item) for item in self._jobs.values()]
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        result = dict(job)
        result.pop("source_url", None)  # Do not echo potentially private signed URLs to every browser client.
        return result

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.get("status") in ACTIVE_STATES:
                return False
            del self._jobs[job_id]
        remove_job_files(self.settings.jobs_dir / job_id)
        return True

    def file_path(self, job_id: str, filename: str) -> Path | None:
        # `filename` must be a basename and must resolve inside this job's output directory.
        if Path(filename).name != filename:
            return None
        with self._lock:
            job = self._jobs.get(job_id)
        if not job or job.get("status") != "completed":
            return None
        path = (self.settings.jobs_dir / job_id / "output" / filename).resolve()
        output_dir = (self.settings.jobs_dir / job_id / "output").resolve()
        if output_dir not in path.parents or not path.is_file():
            return None
        return path
