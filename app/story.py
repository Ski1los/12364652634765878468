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
from .pipeline import Word, ffmpeg_executable, probe_duration, run_command, write_captions
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
 "titles":["3 честных варианта заголовка до 70 символов"],
 "hook":"первая фраза до 110 символов",
 "description":"описание для публикации до 280 символов с пометкой AI-assisted / создано с помощью ИИ",
 "platform_captions":{{"youtube_shorts":"подпись до 180 символов","tiktok":"подпись до 180 символов","instagram_reels":"подпись до 180 символов"}},
 "hashtags":["#тег1","#тег2","#тег3"],
 "disclosure":"Создано с помощью ИИ; история вымышленная.",
 "scenes":[
   {{"narration":"текст для озвучки", "on_screen_text":"2–8 слов на экране", "visual_prompt":"описание абстрактной безопасной сцены без реальных людей"}}
 ]
}}
Нужно 5–8 сцен, суммарно примерно нужной длительности."""


def _local_template_script(job: dict[str, Any], log: Log) -> dict[str, Any]:
    """Build a transparent, offline draft when no LLM is configured.

    This is an editorial template rather than a hidden AI model: it keeps the
    feature operational on a clean modest PC and never transmits the topic.
    """
    topic = _as_text(job.get("topic"), 180, "a small change")
    is_russian = job.get("language") == "ru"
    duration = int(job.get("duration_seconds", 45))
    target_words = int(duration * (1.8 if is_russian else 2.0))
    if is_russian:
        kind = {"fictional": "вымышленная история", "inspiration": "вымышленная вдохновляющая история", "curiosity": "вымышленная лёгкая загадка"}.get(job.get("story_kind"), "вымышленная история")
        scenes = [
            {"narration": f"Однажды вечером герой подумал о теме «{topic}» и заметил маленькую деталь, которую раньше пропускал.", "on_screen_text": "Одна деталь", "visual_prompt": "Тихая абстрактная комната и луч света"},
            {"narration": "Он не стал искать готовый ответ. Вместо этого он остановился, сделал паузу и задал себе один честный вопрос.", "on_screen_text": "Пауза", "visual_prompt": "Минималистичная карточка с мягкими формами"},
            {"narration": "Ответ оказался не громким и не идеальным. Но он подсказал следующий спокойный шаг, который можно сделать уже сегодня.", "on_screen_text": "Следующий шаг", "visual_prompt": "Абстрактная дорога к тёплому горизонту"},
            {"narration": "Герой записал эту мысль, чтобы не потерять её в привычной спешке. И от этого история неожиданно стала яснее.", "on_screen_text": "Записать мысль", "visual_prompt": "Лист бумаги, свет и графические тени"},
            {"narration": "Эта короткая история вымышленная, но вопрос остаётся настоящим: какую маленькую деталь вы заметили бы в своей теме?", "on_screen_text": "А что заметили бы вы?", "visual_prompt": "Спокойная финальная карточка с мягким светом"},
            {"narration": "Иногда перемена начинается не с громкого решения, а с внимательного взгляда на то, что уже рядом.", "on_screen_text": "Внимательный взгляд", "visual_prompt": "Абстрактные круги и тонкая линия"},
            {"narration": "Герой улыбнулся и оставил себе время подумать. Никаких обещаний, только следующий аккуратный шаг.", "on_screen_text": "Без спешки", "visual_prompt": "Тёмный фон, акцентная линия и тёплое свечение"},
            {"narration": "Если эта вымышленная зарисовка отозвалась, сохраните идею для себя и поделитесь своей версией в комментарии.", "on_screen_text": "Ваша версия", "visual_prompt": "Чистая карточка для мягкого диалога"},
        ]
        title = _as_text(f"{topic}: маленький поворот", 70, "Маленький поворот")
        description = f"Оригинальная {kind} на тему «{topic}». Сценарий собран локально по прозрачному шаблону; проверьте его перед публикацией."
        tags = ["#оригинальнаяистория", "#shorts", "#локальныйконтент"]
        disclosure = "Оригинальная вымышленная история. Сценарий собран локально по шаблону; проверьте материал перед публикацией."
        filler = "Это не совет и не обещание результата. Просто спокойная пауза, чтобы выбрать свой следующий шаг."
        titles = [title, "Одна деталь меняет ход истории", "Маленький шаг без громких обещаний"]
    else:
        kind = {"fictional": "fictional story", "inspiration": "fictional reflective story", "curiosity": "fictional light mystery"}.get(job.get("story_kind"), "fictional story")
        scenes = [
            {"narration": f"One evening, the main character thought about {topic} and noticed a small detail that had always been there.", "on_screen_text": "One small detail", "visual_prompt": "Quiet abstract room with a beam of light"},
            {"narration": "Instead of chasing a perfect answer, they paused, took a breath, and wrote down one honest question.", "on_screen_text": "Take a pause", "visual_prompt": "Minimal graphic card with calm shapes"},
            {"narration": "The answer was not dramatic. It simply pointed to one thoughtful next step that could happen today.", "on_screen_text": "One next step", "visual_prompt": "Abstract path toward a warm horizon"},
            {"narration": "They kept that thought on paper, so the rush of an ordinary day would not erase it.", "on_screen_text": "Keep the thought", "visual_prompt": "Paper, light, and editorial shadows"},
            {"narration": "This short story is fictional, but the question is real: what small detail would you notice in your own life?", "on_screen_text": "What would you notice?", "visual_prompt": "Calm ending card with a soft glow"},
            {"narration": "Sometimes change begins without a grand announcement. It starts with looking carefully at what is already near.", "on_screen_text": "Look carefully", "visual_prompt": "Abstract circles connected by a fine line"},
            {"narration": "The character smiled and gave the idea more time. No promises, just one careful next step.", "on_screen_text": "No rush", "visual_prompt": "Dark background with a warm accent line"},
            {"narration": "If this fictional sketch stayed with you, save the idea and share your own version in the comments.", "on_screen_text": "Your version", "visual_prompt": "Clean closing card inviting a gentle conversation"},
        ]
        title = _as_text(f"{topic}: a small turn", 70, "A small turn")
        description = f"An original {kind} about {topic}. The script was assembled locally with a transparent template; review it before publishing."
        tags = ["#originalstory", "#shorts", "#localcreator"]
        disclosure = "Original fictional story. The script was assembled locally from a template; review it before publishing."
        filler = "This is not advice or a promise of results. It is simply a calm pause before choosing a next step."
        titles = [title, "One small detail changes the story", "A quiet next step"]

    scene_count = 5 if duration <= 45 else 6 if duration <= 60 else 8
    scenes = scenes[:scene_count]
    filler_index = 0
    while sum(len(scene["narration"].split()) for scene in scenes) < target_words:
        scenes[filler_index % len(scenes)]["narration"] += " " + filler
        filler_index += 1
    log("Локальный сценарный конструктор собрал оригинальный черновик без LLM и без передачи темы в сеть.")
    captions = {platform: description for platform in ("youtube_shorts", "tiktok", "instagram_reels")}
    return {
        "title": title, "titles": titles, "hook": scenes[0]["narration"][:110],
        "description": description, "platform_captions": captions, "hashtags": tags,
        "disclosure": disclosure, "scenes": scenes,
        "script_engine": "Локальный сценарный шаблон (без LLM)",
    }


def generate_script(job: dict[str, Any], settings: Settings, log: Log) -> dict[str, Any]:
    target = settings.active_llm
    if not target:
        return _local_template_script(job, log)
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
    title = _as_text(data.get("title"), 70, "Новая история")
    raw_titles = data.get("titles") if isinstance(data.get("titles"), list) else []
    title_variants: list[str] = []
    for item in [title, *raw_titles]:
        variant = _as_text(item, 70)
        if variant and variant not in title_variants:
            title_variants.append(variant)
    # A useful content kit should still have alternatives when a smaller LLM omits the new field.
    for fallback in (f"{title}: короткая история", f"Что произошло после этого? — {title}"):
        if len(title_variants) >= 3:
            break
        if fallback not in title_variants:
            title_variants.append(fallback[:70])
    raw_captions = data.get("platform_captions") if isinstance(data.get("platform_captions"), dict) else {}
    description = _as_text(data.get("description"), 280, "Создано с помощью ИИ.")
    captions = {
        platform: _as_text(raw_captions.get(platform), 220, description)
        for platform in ("youtube_shorts", "tiktok", "instagram_reels")
    }
    return {
        "title": title,
        "titles": title_variants[:3],
        "hook": _as_text(data.get("hook"), 110, scenes[0]["narration"][:110]),
        "description": description,
        "platform_captions": captions,
        "hashtags": clean_tags[:8],
        "disclosure": _as_text(data.get("disclosure"), 160, "Создано с помощью ИИ; история вымышленная."),
        "scenes": scenes,
        "script_engine": f"{location} LLM",
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


def create_cover_image(story: dict[str, Any], output_dir: Path, height: int) -> Path:
    """Generate an original square/vertical-safe cover without external imagery."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width = int(round(height * 9 / 16 / 2) * 2)
    dark, accent, pale = PALETTES[0]
    image = Image.new("RGB", (width, height), dark)
    draw = ImageDraw.Draw(image)
    draw.ellipse((-width // 2, -height // 6, width * 3 // 4, height * 3 // 5), fill=accent)
    draw.ellipse((width // 3, height // 2, width + width // 2, height + height // 4), fill="#07151a")
    margin = round(width * .085)
    draw.rounded_rectangle((margin, round(height * .08), width - margin, round(height * .16)), radius=12, fill="#101f27")
    draw.text((margin + 15, round(height * .103)), "CLIPLAB ORIGINAL STORY", font=_font(max(15, round(height * .017)), bold=True), fill=pale)
    _draw_centered(
        draw,
        (margin, round(height * .25), width - margin, round(height * .62)),
        story["title"],
        _font(max(34, round(height * .056)), bold=True),
        "#ffffff",
        spacing=round(height * .012),
    )
    draw.line((margin, round(height * .71), width - margin, round(height * .71)), fill="#dfff38", width=max(4, round(height * .005)))
    _draw_centered(
        draw,
        (margin, round(height * .75), width - margin, round(height * .88)),
        story["hook"],
        _font(max(17, round(height * .021))),
        pale,
        spacing=round(height * .006),
    )
    path = output_dir / "cover.png"
    image.save(path, optimize=True)
    return path


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
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak:
            voice = "ru" if language == "ru" else "en-us"
            command = [espeak, "-v", voice, "-s", "155", "-w", str(output_file), "-f", str(text_file)]
            log("Озвучиваю через локальный eSpeak NG (для более естественного голоса подключите Piper)…")
            run_command(command, log)
            engine = "eSpeak NG (локальный fallback)"
        else:
            # A bare pip installation may not ship a system TTS executable. Keep
            # the Story Reel usable as a subtitle-led video and disclose silence
            # honestly; Docker always includes eSpeak NG.
            seconds = max(2.0, len(text.split()) / 2.0)
            log("Piper/eSpeak не найдены: создаю тихую дорожку. Для речи используйте Docker или подключите Piper.")
            run_command([
                ffmpeg_executable(), "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
                "-t", f"{seconds:.2f}", "-c:a", "pcm_s16le", str(output_file),
            ], log)
            engine = "Тихая дорожка — eSpeak/Piper не найден"
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
    cover = create_cover_image(story, output_dir, height)
    cards = create_scene_cards(story, output_dir, height)
    concat = output_dir / "scenes.txt"
    _concat_file(cards, scene_durations, concat)
    silent_video = output_dir / "visuals.mp4"
    log("Собираю вертикальные сторис-карточки…")
    run_command([
        ffmpeg_executable(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        # The concat manifest owns every scene duration. A zoompan `d=1` here
        # would collapse a 45-second story into one frame per card, so retain
        # source timing and render it at a stable mobile-friendly frame rate.
        "-vf", "fps=30,format=yuv420p",
        "-threads", str(settings.ffmpeg_threads), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(silent_video),
    ], log)
    srt_file = output_dir / "story.srt"
    ass_file = output_dir / "story.ass"
    write_captions(words, 0, width, height, srt_file, ass_file)
    output_video = output_dir / "story-reel.mp4"
    subtitle = str(ass_file).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    log("Вшиваю субтитры и аудиодорожку в итоговый ролик…")
    run_command([
        ffmpeg_executable(), "-y", "-i", str(silent_video), "-i", str(voice_file),
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
        "thumbnail": cover.name,
        "duration": round(audio_duration, 2),
        "voice_engine": voice_engine,
    }


def write_content_pack(story: dict[str, Any], story_render: dict[str, Any], target: Path) -> None:
    hashtag_line = " ".join(story["hashtags"])
    titles = "\n".join(f"{index}. {item}" for index, item in enumerate(story.get("titles", [story["title"]]), start=1))
    captions = story.get("platform_captions", {})
    body = f"""TITLE OPTIONS
{titles}

HOOK
{story['hook']}

DESCRIPTION
{story['description']}

YOUTUBE SHORTS CAPTION
{captions.get('youtube_shorts', story['description'])}

TIKTOK CAPTION
{captions.get('tiktok', story['description'])}

INSTAGRAM REELS CAPTION
{captions.get('instagram_reels', story['description'])}

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
