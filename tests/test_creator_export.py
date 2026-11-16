import zipfile

from app.jobs import JobManager
from app.settings import Settings


def _settings(tmp_path):
    return Settings(
        data_dir=tmp_path, whisper_model="base", whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=2, ffmpeg_threads=2, max_source_minutes=10, max_source_mb=10,
        local_llm_url="", local_llm_model="qwen2.5:3b", local_llm_api_key="",
        cloud_llm_url="", cloud_llm_model="", cloud_llm_api_key="", llm_mode="auto",
        max_workers=1, ytdlp_cookies_file="", piper_voice_model="", piper_speaker="",
    )


def test_completed_job_can_be_exported_as_a_creator_kit(tmp_path):
    settings = _settings(tmp_path)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    manager = JobManager(settings)
    job_id = "completed-job"
    output = settings.jobs_dir / job_id / "output"
    (output / "scenes").mkdir(parents=True)
    (output / "story-reel.mp4").write_bytes(b"video")
    (output / "story.srt").write_text("subtitle", encoding="utf-8")
    (output / "content-pack.txt").write_text("copy", encoding="utf-8")
    (output / "cover.png").write_bytes(b"image")
    (output / "scenes" / "scene-01.jpg").write_bytes(b"cache")
    manager._jobs[job_id] = {"id": job_id, "status": "completed"}
    archive = manager.archive_path(job_id)
    assert archive and archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"story-reel.mp4", "story.srt", "content-pack.txt", "cover.png"}
