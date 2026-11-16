"""Real FFmpeg smoke test using a bundled test binary, not a mocked renderer."""
from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg

from app.pipeline import Candidate, Word, render_clip
from app.settings import Settings
from app.story import render_story


def _settings(root: Path) -> Settings:
    return Settings(
        data_dir=root, whisper_model="base", whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=2, ffmpeg_threads=2, max_source_minutes=1, max_source_mb=10,
        local_llm_url="", local_llm_model="qwen2.5:3b", local_llm_api_key="",
        cloud_llm_url="", cloud_llm_model="", cloud_llm_api_key="", llm_mode="auto",
        max_workers=1, ytdlp_cookies_file="", ytdlp_proxy="", ytdlp_impersonate="chrome", ytdlp_force_ipv4=False, piper_voice_model="", piper_speaker="",
    )


def _create_fake_tools(bin_dir: Path) -> None:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    (bin_dir / "ffmpeg").symlink_to(ffmpeg)
    (bin_dir / "ffprobe").write_text("#!/bin/sh\necho 4.0\n", encoding="utf-8")
    # Docker supplies espeak-ng. This tiny deterministic substitute keeps the
    # integration test self-contained while exercising the non-Piper code path.
    (bin_dir / "espeak-ng").write_text(
        """#!/usr/bin/env python3
import sys, wave
from pathlib import Path
output = Path(sys.argv[sys.argv.index('-w') + 1])
output.parent.mkdir(parents=True, exist_ok=True)
with wave.open(str(output), 'wb') as wav:
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(22050)
    wav.writeframes(b'\\x00\\x00' * (22050 * 4))
""",
        encoding="utf-8",
    )
    for path in bin_dir.iterdir():
        path.chmod(0o755)


def test_real_ffmpeg_renders_a_clip_and_story_reel(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _create_fake_tools(bin_dir)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=800:sample_rate=44100", "-t", "4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    settings = _settings(tmp_path)
    words = [
        Word(.25, .6, "Это"), Word(.62, .95, "тестовый"), Word(.97, 1.3, "ролик."),
        Word(1.4, 1.7, "Субтитры"), Word(1.72, 2.05, "работают!"),
    ]
    clip = render_clip(
        source, Candidate(.1, 3.5, 2, "Тест", "интеграционный"), words, 1,
        tmp_path / "clip", 720, settings, lambda _: None,
    )
    assert (tmp_path / "clip" / clip["video"]).stat().st_size > 5_000
    story = {
        "title": "Тестовая история", "hook": "Короткий тест.", "description": "AI-assisted.",
        "hashtags": ["#тест"], "disclosure": "Создано с помощью ИИ; история вымышленная.",
        "scenes": [
            {"narration": "Первая короткая тестовая сцена для проверки сборки.", "on_screen_text": "Первая сцена", "visual_prompt": "Абстрактный свет."},
            {"narration": "Вторая короткая тестовая сцена продолжает эту историю.", "on_screen_text": "Вторая сцена", "visual_prompt": "Мягкий градиент."},
            {"narration": "Третья сцена проверяет титры в итоговом ролике.", "on_screen_text": "Третья сцена", "visual_prompt": "Редакционная карточка."},
            {"narration": "Финальная сцена завершает проверку локального ролика.", "on_screen_text": "Готово", "visual_prompt": "Спокойный финал."},
        ],
    }
    result = render_story(story, {"render_height": 720, "language": "ru"}, settings, tmp_path / "story", lambda _: None)
    assert (tmp_path / "story" / result["video"]).stat().st_size > 5_000
    assert (tmp_path / "story" / result["subtitles"]).is_file()
