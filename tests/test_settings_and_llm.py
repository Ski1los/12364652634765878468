import pytest

from app.llm import chat_completion
from app.pipeline import PipelineError, downloader_network_error, download_source
from app.settings import LLMTarget, Settings, load_settings


def test_llm_selection_prefers_local_and_can_explicitly_use_cloud(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOCAL_LLM_URL", "http://ollama:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("CLOUD_LLM_URL", "https://llm.example/v1")
    monkeypatch.setenv("CLOUD_LLM_MODEL", "fast-model")
    monkeypatch.setenv("LLM_MODE", "auto")
    settings = load_settings()
    assert settings.active_llm and settings.active_llm.provider == "local"

    monkeypatch.setenv("LLM_MODE", "cloud")
    settings = load_settings()
    assert settings.active_llm and settings.active_llm.provider == "cloud"
    assert settings.active_llm.url == "https://llm.example/v1"


def test_cloud_endpoint_requires_https(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLOUD_LLM_URL", "http://insecure.example/v1")
    with pytest.raises(ValueError, match="CLOUD_LLM_URL"):
        load_settings()


def test_chat_completion_attaches_key_without_exposing_it(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "готовый текст"}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("app.llm.requests.post", fake_post)
    result = chat_completion(
        LLMTarget("https://llm.example/v1", "small-model", "secret-key", "cloud"),
        [{"role": "user", "content": "Привет"}],
        temperature=0.2,
        timeout=12,
    )
    assert result == "готовый текст"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["model"] == "small-model"


def test_download_source_has_size_and_duration_guardrails(monkeypatch, tmp_path):
    settings = Settings(
        data_dir=tmp_path, whisper_model="base", whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=6, ffmpeg_threads=6, max_source_minutes=90, max_source_mb=2500,
        local_llm_url="", local_llm_model="qwen2.5:3b", local_llm_api_key="",
        cloud_llm_url="", cloud_llm_model="", cloud_llm_api_key="", llm_mode="auto",
        max_workers=1, ytdlp_cookies_file="", ytdlp_proxy="", ytdlp_impersonate="chrome", ytdlp_force_ipv4=False, piper_voice_model="", piper_speaker="",
    )
    captured = []

    def fake_run(command, _log, **_kwargs):
        captured.extend(command)
        (tmp_path / "source.mp4").write_bytes(b"x" * 2048)

    monkeypatch.setattr("app.pipeline.run_command", fake_run)
    result = download_source("https://www.youtube.com/watch?v=abc", tmp_path, settings, lambda _: None)
    assert result.name == "source.mp4"
    assert "duration <= 5400" in captured
    assert "2500M" in captured
    assert "--impersonate" in captured and "chrome" in captured


def test_tls_downloader_error_has_an_actionable_message():
    assert "HTTPS-соединение" in downloader_network_error("TLS/SSL connection has been closed (EOF)")
    assert "HTTPS-соединение" in downloader_network_error("curl: (35) BoringSSL SSL_connect: Connection closed abruptly")
    assert "DNS-адрес" in downloader_network_error("curl: (6) Could not resolve host: www.youtube.com")
    assert downloader_network_error("video unavailable") is None
