"""Local Story Studio: script, narration and vertical story-reel generation.

The module deliberately creates original fictional/educational-style material and
labels it as AI-assisted. It does not scrape stories, clone voices or impersonate
real people. Everything except the user-selected local models stays on disk.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from .errors import PipelineError
from .llm import chat_completion
from .pipeline import Word, probe_duration, run_command, write_captions
from .settings import Settings
from .storage import write_json

Log = Callable[[str], None]

PALETTES = [
    ("#14213d", "#fca311", "#e5e5e5"),
    ("#2d1b36", "#f063a6", "#f7d9e5"),
    ("#073b4c", "#06d6a0", "#e8faf6"),
    ("#402218", "#ff9f1c", "#fff4df"),
    ("#202c39", "#d9e76c", "#eff4de"),
    ("#1b263b", "#79c2d0", "#e2f5f8"),
]


def _as_text(value: object, limit: int, fallback: str = "") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit] if text else fallback


def _extract_json(text: str) -> object | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _story_prompt(job: dict[str, Any]) -> str:
    language = "русском" if job["language"] == "ru" else "English"
    kind = {
        "fictional": "оригинальная художественная мини-история, прямо помеченная как вымышленная",
        "inspiration": "оригинальная вдохновляющая история без обещаний успеха, заработка или результата",
        "curiosity": "оригинальный развлекательный сценарий с безопасным вымышленным допущением; не выдавай его за новость или факт",
    }[job["story_kind"]]
    style = {"cinematic": "кинематографичный", "comic": "живой и ироничный", "minimal": "спокойный минималистичный"}[job["visual_style"]]
    return f"""Ты сценарист коротких вертикальных видео. Напиши на {language} {kind}.
Тема пользователя: {job['topic']!r}
Целевая длительность озвучки: около {job['duration_seconds']} секунд. Визуальный тон: {style}.

Правила безопасности и качества:
- Всё должно быть оригинальным. Не пересказывай чужие посты, фильмы, книги или ролики.
- Не используй реальных людей, бренды, обвинения, медицинские/финансовые/юридические советы и не обещай заработок.
- Не маскируй вымысел под реальную новость или личный опыт. Не используй манипулятивные «гарантированно вирусно» обещания.
- Начни с интригующей, но честной первой фразы. Закончи мягким призывом к комментарию, без давления.
- Текст должен быть готов для синтеза речи: короткие ясные предложения, без markdown и без эмодзи в narration.

Верни ТОЛЬКО валидный JSON-объект:
{{
 "title":"короткий заголовок до 70 символов",
 "hook":"первая фраза до 110 символов",
 "description":"описание для публикации до 280 символов с пометкой AI-assisted / создано с помощью ИИ",
 "hashtags":["#тег1","#тег2","#тег3"],
 "disclosure":"Создано с помощью ИИ; история вымышленная.",
 "scenes":[
   {{"narration":"текст для озвучки", "on_screen_text":"2–8 слов на экране", "visual_prompt":"описание абстрактной безопасной сцены без реальных людей"}}
 ]
}}
Нужно 5–8 сцен, суммарно примерно нужной длительности."""


def generate_script(job: dict[str, Any], settings: Settings, log: Log) -> dict[str, Any]:
    target = settings.active_llm
    if not target:
        raise PipelineError(
            "Story Studio требует локальную или явно настроенную облачную LLM. Для локального режима "
            "включите Ollama: LOCAL_LLM_URL=http://ollama:11434/v1 и скачайте qwen2.5:3b."
        )
    if target.is_cloud and not job.get("cloud_processing_allowed"):
        raise PipelineError(
            "Для облачного сценария нужно отдельное разрешение: тема и инструкция будут отправлены "
            "в настроенный внешний OpenAI-совместимый сервис."
        )
    location = "Облачная" if target.is_cloud else "Локальная"
    log(f"{location} LLM пишет оригинальный сценарий и пакет публикации…")
    raw = chat_completion(
        target,
        [
            {"role": "system", "content": "Отвечай только корректным JSON без markdown."},
            {"role": "user", "content": _story_prompt(job)},
        ],
        temperature=0.75,
        timeout=240,
    )
    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise PipelineError("Локальная LLM вернула сценарий не в JSON-формате. Повторите задачу или смените модель.")
    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list) or not 4 <= len(raw_scenes) <= 10:
        raise PipelineError("Сценарий должен содержать от 4 до 10 сцен. Модель вернула неполный ответ.")
    scenes: list[dict[str, str]] = []
    for item in raw_scenes:
        if not isinstance(item, dict):
            continue
        narration = _as_text(item.get("narration"), 600)
        if len(narration.split()) < 4:
            continue
        scenes.append({
            "narration": narration,
            "on_screen_text": _as_text(item.get("on_screen_text"), 90, "Продолжение истории"),
            "visual_prompt": _as_text(item.get("visual_prompt"), 240, "Абстрактная сюжетная карточка"),
        })
    if len(scenes) < 4:
        raise PipelineError("В сгенерированном сценарии слишком мало текста для полноценного ролика.")
    tags = data.get("hashtags", [])
    if not isinstance(tags, list):
        tags = []
    clean_tags = []
    for tag in tags:
        value = _as_text(tag, 40)
        if value:
            clean_tags.append(value if value.startswith("#") else "#" + value.replace(" ", ""))
    return {
        "title": _as_text(data.get("title"), 70, "Новая история"),
        "hook": _as_text(data.get("hook"), 110, scenes[0]["narration"][:110]),
        "description": _as_text(data.get("description"), 280, "Создано с помощью ИИ."),
        "hashtags": clean_tags[:8],
        "disclosure": _as_text(data.get("disclosure"), 160, "Создано с помощью ИИ; история вымышленная."),
        "scenes": scenes,
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
        if bold else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, maximum: int) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = (current + " " + word).strip()
        if current and draw.textbbox((0, 0), proposal, font=font)[2] > maximum:
            lines.append(current)
            current = word
        else:
            current = proposal
    if current:
        lines.append(current)
    return "\n".join(lines[:5])


def _draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str, spacing: int = 8) -> None:
    left, top, right, bottom = box
    wrapped = _wrap(draw, text, font, right - left)
    bounds = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align="center")
    x = left + ((right - left) - (bounds[2] - bounds[0])) / 2
    y = top + max(0, ((bottom - top) - (bounds[3] - bounds[1])) / 2)
    draw.multiline_text((x, y), wrapped, font=font, fill=fill, spacing=spacing, align="center")


def create_scene_cards(story: dict[str, Any], output_dir: Path, height: int) -> list[Path]:
    width = int(round(height * 9 / 16 / 2) * 2)
    slides_dir = output_dir / "scenes"
    slides_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, scene in enumerate(story["scenes"], start=1):
        dark, accent, pale = PALETTES[(index - 1) % len(PALETTES)]
        image = Image.new("RGB", (width, height), dark)
        draw = ImageDraw.Draw(image)
        # Simple editorial geometry has no copyright/personality rights burden and is fast locally.
        draw.ellipse((-width // 3, -height // 10, width // 2, height // 2), fill=accent)
        draw.ellipse((width // 2, height // 2, width + width // 3, height + height // 5), fill="#101820")
        margin = round(width * .09)
        draw.rounded_rectangle((margin, round(height * .09), width - margin, round(height * .15)), radius=12, outline=pale, width=2)
        label_font = _font(max(15, round(height * .017)), bold=True)
        draw.text((margin + 15, round(height * .105)), f"STORY REEL  /  {index:02}", font=label_font, fill=pale)
        _draw_centered(
            draw, (margin, round(height * .24), width - margin, round(height * .59)), scene["on_screen_text"],
            _font(max(30, round(height * .052)), bold=True), "#ffffff", spacing=round(height * .011),
        )
        draw.line((margin, round(height * .68), width - margin, round(height * .68)), fill=accent, width=max(4, round(height * .005)))
        _draw_centered(
            draw, (margin, round(height * .73), width - margin, round(height * .88)), scene["visual_prompt"],
            _font(max(16, round(height * .020))), pale, spacing=round(height * .006),
        )
        path = slides_dir / f"scene-{index:02}.jpg"
        image.save(path, quality=93, optimize=True)
        paths.append(path)
    return paths


def synthesize_narration(text: str, language: str, output_file: Path, settings: Settings, log: Log) -> str:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    text_file = output_file.with_suffix(".txt")
    text_file.write_text(text, encoding="utf-8")
    voice_path = Path(settings.piper_voice_model) if settings.piper_voice_model else None
    piper = shutil.which("piper")
    if voice_path and voice_path.is_file() and piper:
        command = [piper, "--model", str(voice_path), "--output_file", str(output_file)]
        if settings.piper_speaker:
            command.extend(["--speaker", settings.piper_speaker])
        log("Озвучиваю локальной Piper-моделью…")
        process = subprocess.run(command, input=text, text=True, capture_output=True, check=False)
        if process.returncode:
            raise PipelineError(f"Piper не смог создать озвучку: {(process.stderr or process.stdout)[-1000:]}")
        engine = "Piper (локальная модель)"
    else:
        # Included by the Docker image. It is a zero-setup offline fallback, not a cloud request.
        voice = "ru" if language == "ru" else "en-us"
        command = ["espeak-ng", "-v", voice, "-s", "155", "-w", str(output_file), "-f", str(text_file)]
        log("Озвучиваю через локальный eSpeak NG (для более естественного голоса подключите Piper)…")
        run_command(command, log)
        engine = "eSpeak NG (локальный fallback)"
    if not output_file.is_file() or output_file.stat().st_size < 300:
        raise PipelineError("Локальный синтез речи не создал аудиофайл.")
    return engine


def _story_words(story: dict[str, Any], duration: float) -> tuple[list[Word], list[float]]:
    scene_tokens = [scene["narration"].split() for scene in story["scenes"]]
    all_count = sum(len(tokens) for tokens in scene_tokens) or 1
    words: list[Word] = []
    durations: list[float] = []
    cursor = 0.0
    for tokens in scene_tokens:
        scene_duration = max(2.8, duration * len(tokens) / all_count)
        durations.append(scene_duration)
        step = scene_duration / max(1, len(tokens))
        for index, token in enumerate(tokens):
            words.append(Word(cursor + index * step, cursor + (index + 1) * step * .92, token))
        cursor += scene_duration
    return words, durations


def _concat_file(paths: list[Path], durations: list[float], output: Path) -> None:
    lines: list[str] = []
    for path, duration in zip(paths, durations):
        escaped = str(path.resolve()).replace("'", r"'\\''")
        lines.extend([f"file '{escaped}'", f"duration {duration:.3f}"])
    # ffmpeg concat applies the previous duration only when another input follows.
    escaped_last = str(paths[-1].resolve()).replace("'", r"'\\''")
    lines.append(f"file '{escaped_last}'")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_story(story: dict[str, Any], job: dict[str, Any], settings: Settings, output_dir: Path, log: Log) -> dict[str, Any]:
    height = job["render_height"]
    width = int(round(height * 9 / 16 / 2) * 2)
    narration = " ".join(scene["narration"] for scene in story["scenes"])
    voice_file = output_dir / "narration.wav"
    voice_engine = synthesize_narration(narration, job["language"], voice_file, settings, log)
    audio_duration = probe_duration(voice_file, log)
    words, scene_durations = _story_words(story, audio_duration)
    cards = create_scene_cards(story, output_dir, height)
    concat = output_dir / "scenes.txt"
    _concat_file(cards, scene_durations, concat)
    silent_video = output_dir / "visuals.mp4"
    log("Собираю вертикальные сторис-карточки…")
    run_command([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        # Subtle Ken Burns movement keeps original local story cards from feeling like a slideshow.
        "-vf", f"zoompan=z='min(max(zoom,pzoom)+0.0006,1.075)':d=1:s={width}x{height}:fps=30,format=yuv420p",
        "-threads", str(settings.ffmpeg_threads), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(silent_video),
    ], log)
    srt_file = output_dir / "story.srt"
    ass_file = output_dir / "story.ass"
    write_captions(words, 0, width, height, srt_file, ass_file)
    output_video = output_dir / "story-reel.mp4"
    subtitle = str(ass_file).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    log("Вшиваю субтитры и аудиодорожку в итоговый ролик…")
    run_command([
        "ffmpeg", "-y", "-i", str(silent_video), "-i", str(voice_file),
        "-vf", f"subtitles=filename='{subtitle}'", "-map", "0:v", "-map", "1:a", "-shortest",
        "-threads", str(settings.ffmpeg_threads), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(output_video),
    ], log)
    if not output_video.is_file() or output_video.stat().st_size < 1024:
        raise PipelineError("Не удалось собрать итоговый Story Reel.")
    return {
        "video": output_video.name,
        "subtitles": srt_file.name,
        "audio": voice_file.name,
        "duration": round(audio_duration, 2),
        "voice_engine": voice_engine,
    }


def write_content_pack(story: dict[str, Any], story_render: dict[str, Any], target: Path) -> None:
    hashtag_line = " ".join(story["hashtags"])
    body = f"""TITLE
{story['title']}

HOOK
{story['hook']}

DESCRIPTION
{story['description']}

HASHTAGS
{hashtag_line}

DISCLOSURE
{story['disclosure']}

VOICE
{story_render['voice_engine']}

SCENES
"""
    for index, scene in enumerate(story["scenes"], start=1):
        body += f"\n{index}. {scene['on_screen_text']}\n{scene['narration']}\n"
    target.write_text(body, encoding="utf-8")


def run_story_pipeline(job: dict[str, Any], settings: Settings, log: Log) -> dict[str, Any]:
    job_dir = settings.jobs_dir / job["id"]
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    script = generate_script(job, settings, log)
    write_json(job_dir / "story-script.json", script)
    render = render_story(script, job, settings, output_dir, log)
    pack_file = output_dir / "content-pack.txt"
    write_content_pack(script, render, pack_file)
    result = {**script, **render, "content_pack": pack_file.name}
    write_json(output_dir / "story.json", result)
    log("Story Reel готов: видео, SRT, аудио и пакет текста для публикации сохранены локально.")
    return result
