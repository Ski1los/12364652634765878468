"""Environment-backed settings for the local-only video pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    whisper_model: str
    local_llm_url: str
    local_llm_model: str
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


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    settings = Settings(
        data_dir=data_dir,
        whisper_model=os.getenv("WHISPER_MODEL", "small").strip() or "small",
        local_llm_url=os.getenv("LOCAL_LLM_URL", "").rstrip("/"),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b").strip() or "qwen2.5:7b",
        max_workers=_positive_int("MAX_WORKERS", 1),
        ytdlp_cookies_file=os.getenv("YTDLP_COOKIES_FILE", "").strip(),
        # Piper itself is packaged in requirements; a voice model is deliberately
        # supplied by the self-hoster so no hidden cloud TTS is ever used.
        piper_voice_model=os.getenv("PIPER_VOICE_MODEL", "").strip(),
        piper_speaker=os.getenv("PIPER_SPEAKER", "").strip(),
    )
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
