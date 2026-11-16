"""Validated, environment-backed settings for local and optional cloud-assisted jobs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class LLMTarget:
    """One configured OpenAI-compatible chat endpoint, chosen server-side."""

    url: str
    model: str
    api_key: str
    provider: str  # local | cloud

    @property
    def is_cloud(self) -> bool:
        return self.provider == "cloud"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int
    ffmpeg_threads: int
    max_source_minutes: int
    max_source_mb: int
    local_llm_url: str
    local_llm_model: str
    local_llm_api_key: str
    cloud_llm_url: str
    cloud_llm_model: str
    cloud_llm_api_key: str
    llm_mode: str
    max_workers: int
    ytdlp_cookies_file: str
    piper_voice_model: str
    piper_speaker: str

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / ".cache" / "huggingface"

    @property
    def active_llm(self) -> LLMTarget | None:
        """Pick local first in auto mode; cloud is never silently selected over it."""
        local = (
            LLMTarget(self.local_llm_url, self.local_llm_model, self.local_llm_api_key, "local")
            if self.local_llm_url else None
        )
        cloud = (
            LLMTarget(self.cloud_llm_url, self.cloud_llm_model, self.cloud_llm_api_key, "cloud")
            if self.cloud_llm_url else None
        )
        if self.llm_mode == "local":
            return local
        if self.llm_mode == "cloud":
            return cloud
        return local or cloud


def _positive_int(name: str, default: int, *, maximum: int = 256) -> int:
    try:
        return min(maximum, max(1, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _nonnegative_int(name: str, default: int, *, maximum: int = 1_000_000) -> int:
    try:
        return min(maximum, max(0, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _endpoint(value: str, *, cloud: bool = False) -> str:
    url = value.strip().rstrip("/")
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in ({"https"} if cloud else {"http", "https"}) or not parsed.netloc:
        label = "CLOUD_LLM_URL" if cloud else "LOCAL_LLM_URL"
        raise ValueError(f"{label} должен быть полным {'HTTPS ' if cloud else ''}URL OpenAI-совместимого сервера.")
    return url


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    llm_mode = os.getenv("LLM_MODE", "auto").strip().lower()
    if llm_mode not in {"auto", "local", "cloud"}:
        raise ValueError("LLM_MODE может быть только auto, local или cloud.")
    settings = Settings(
        data_dir=data_dir,
        # `base` + int8 is the sensible default for a 6-core CPU such as Ryzen 5 5600.
        whisper_model=os.getenv("WHISPER_MODEL", "base").strip() or "base",
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip().lower() or "cpu",
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip().lower() or "int8",
        whisper_cpu_threads=_positive_int("WHISPER_CPU_THREADS", 6, maximum=64),
        ffmpeg_threads=_positive_int("FFMPEG_THREADS", 6, maximum=64),
        max_source_minutes=_nonnegative_int("MAX_SOURCE_MINUTES", 120, maximum=1_440),
        max_source_mb=_nonnegative_int("MAX_SOURCE_MB", 3_500, maximum=100_000),
        local_llm_url=_endpoint(os.getenv("LOCAL_LLM_URL", "")),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b").strip() or "qwen2.5:3b",
        local_llm_api_key=os.getenv("LOCAL_LLM_API_KEY", "").strip(),
        cloud_llm_url=_endpoint(os.getenv("CLOUD_LLM_URL", ""), cloud=True),
        cloud_llm_model=os.getenv("CLOUD_LLM_MODEL", "").strip(),
        cloud_llm_api_key=os.getenv("CLOUD_LLM_API_KEY", "").strip(),
        llm_mode=llm_mode,
        max_workers=_positive_int("MAX_WORKERS", 1, maximum=8),
        ytdlp_cookies_file=os.getenv("YTDLP_COOKIES_FILE", "").strip(),
        # Piper itself is packaged in requirements; a voice model is deliberately
        # supplied by the self-hoster so no hidden cloud TTS is ever used.
        piper_voice_model=os.getenv("PIPER_VOICE_MODEL", "").strip(),
        piper_speaker=os.getenv("PIPER_SPEAKER", "").strip(),
    )
    if settings.whisper_device != "cpu":
        raise ValueError(
            "В Docker-сборке поддерживается WHISPER_DEVICE=cpu. RX 580 не поддерживает CUDA; "
            "оставьте CPU/int8 или используйте отдельный совместимый worker."
        )
    if settings.cloud_llm_url and not settings.cloud_llm_model:
        raise ValueError("Для CLOUD_LLM_URL обязательно укажите CLOUD_LLM_MODEL.")
    data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
