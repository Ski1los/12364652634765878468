import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import PipelineError, Word, build_candidates, non_overlapping_best, validate_source_url, write_captions


def test_home_and_rejected_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "runtime"))
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "ClipLab Local" in page.text
        assert client.get("/api/config").status_code == 200

        rejected_domain = client.post(
            "/api/jobs",
            json={
                "source_url": "https://example.com/video",
                "clip_count": 3,
                "min_seconds": 20,
                "max_seconds": 50,
                "render_height": 720,
                "rights_confirmed": True,
            },
        )
        assert rejected_domain.status_code == 422
        assert "YouTube" in rejected_domain.json()["detail"]

        rights_required = client.post(
            "/api/jobs",
            json={
                "source_url": "https://www.youtube.com/watch?v=example",
                "clip_count": 3,
                "min_seconds": 20,
                "max_seconds": 50,
                "render_height": 720,
                "rights_confirmed": False,
            },
        )
        assert rights_required.status_code == 422

        missing_disclosure = client.post(
            "/api/stories",
            json={
                "topic": "Вымышленная история о старой библиотеке",
                "story_kind": "fictional",
                "language": "ru",
                "duration_seconds": 45,
                "visual_style": "cinematic",
                "render_height": 720,
                "ai_disclosure_confirmed": False,
            },
        )
        assert missing_disclosure.status_code == 422

        monkeypatch.setattr(
            "app.jobs.run_story_pipeline",
            lambda *_: {"title": "Шаблон", "hook": "Хук", "description": "Локальный черновик.", "hashtags": [],
                        "disclosure": "Шаблон.", "scenes": [], "video": "story.mp4", "subtitles": "story.srt",
                        "audio": "narration.wav", "content_pack": "content-pack.txt", "duration": 30,
                        "voice_engine": "test", "script_engine": "Локальный сценарный шаблон (без LLM)"},
        )
        no_llm = client.post(
            "/api/stories",
            json={
                "topic": "Вымышленная история о старой библиотеке и письме",
                "story_kind": "fictional",
                "language": "ru",
                "duration_seconds": 45,
                "visual_style": "cinematic",
                "render_height": 720,
                "cloud_processing_allowed": False,
                "ai_disclosure_confirmed": True,
            },
        )
        assert no_llm.status_code == 202
        config = client.get("/api/config").json()
        assert config["story_ready"] is True
        assert config["story_engine"] == "local_template"
        assert config["plan_ready"] is False


def test_url_allowlist_and_caption_generation(tmp_path):
    assert validate_source_url("https://youtu.be/abc123") == "https://youtu.be/abc123"
    assert validate_source_url("https://www.twitch.tv/videos/123") == "https://www.twitch.tv/videos/123"
    for unsafe in ("http://youtube.com/watch?v=a", "https://youtube.com.evil.test/a", "https://127.0.0.1/a"):
        try:
            validate_source_url(unsafe)
            raise AssertionError(f"unsafe URL was accepted: {unsafe}")
        except PipelineError:
            pass

    tokens = "Это было так неожиданно что все начали смеяться ха ха ха вот это поворот ну почему так смешно вообще".split()
    words = [Word(index * 0.48, index * 0.48 + 0.35, token) for index, token in enumerate(tokens)]
    candidates = build_candidates(words, 4, 12)
    assert candidates
    assert non_overlapping_best(candidates, 2)

    srt = tmp_path / "clip.srt"
    ass = tmp_path / "clip.ass"
    write_captions(words, 0, 720, 1280, srt, ass)
    assert "Это было так" in srt.read_text(encoding="utf-8")
    assert "[Events]" in ass.read_text(encoding="utf-8")


def test_story_request_enters_queue_and_completes_with_a_local_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("LOCAL_LLM_URL", "http://local-ollama.test/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(
        "app.jobs.run_story_pipeline",
        lambda *_: {
            "title": "Тест", "hook": "Хук", "description": "AI-assisted.", "hashtags": [],
            "disclosure": "Создано с помощью ИИ.", "scenes": [], "video": "story.mp4",
            "subtitles": "story.srt", "audio": "narration.wav", "content_pack": "content-pack.txt",
            "duration": 30, "voice_engine": "test",
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/stories",
            json={
                "topic": "Вымышленная история о старом письме и библиотеке",
                "story_kind": "fictional",
                "language": "ru",
                "duration_seconds": 30,
                "visual_style": "cinematic",
                "render_height": 720,
                "cloud_processing_allowed": False,
                "ai_disclosure_confirmed": True,
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        for _ in range(30):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(.02)
        assert job["status"] == "completed"
        assert job["story"]["video"] == "story.mp4"


def test_creator_plan_request_enters_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("LOCAL_LLM_URL", "http://local-ollama.test/v1")
    monkeypatch.setattr(
        "app.jobs.run_plan_pipeline",
        lambda *_: {"positioning": "Тест", "content_pillars": [], "schedule": [], "metrics_focus": [], "publish_checklist": [], "plan_days": 7, "file": "creator-plan.md"},
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/plans",
            json={
                "niche": "Оригинальные истории", "audience": "Зрители коротких видео",
                "content_focus": "stories", "platforms": ["youtube_shorts", "tiktok"],
                "plan_days": 7, "language": "ru", "cloud_processing_allowed": False,
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        for _ in range(30):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(.02)
        assert job["kind"] == "plan"
        assert job["status"] == "completed"
        assert job["plan"]["file"] == "creator-plan.md"
