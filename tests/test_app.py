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
