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

from PIL import Image, ImageDraw, ImageFilter, ImageFont

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

# Story Reel motion defaults: a slow Ken Burns zoom plus soft crossfades turn the
# scene cards into an actual video instead of a static slideshow.
STORY_FPS = 30
CROSSFADE_SECONDS = 0.6
KEN_BURNS_ZOOM = 0.12
CARD_OVERSAMPLE = 2  # cards are rendered at 2x so the zoom stays sharp and smooth


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
Нужно 6–9 сцен. Каждая сцена — 2–4 коротких предложения (примерно 18–35 слов narration), чтобы суммарная озвучка попадала в целевую длительность."""


def _local_template_script(job: dict[str, Any], log: Log) -> dict[str, Any]:
    """Build a transparent, offline draft when no LLM is configured.

    This is an editorial template rather than a hidden AI model: it keeps the
    feature operational on a clean modest PC and never transmits the topic.
    """
    topic = _as_text(job.get("topic"), 180, "a small change")
    is_russian = job.get("language") == "ru"
    duration = int(job.get("duration_seconds", 45))
    # Local TTS reads at roughly two words per second; asking for enough words is
    # what actually makes the finished reel last as long as the user picked.
    target_words = int(duration * (2.2 if is_russian else 2.4))
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

    scene_count = 6 if duration <= 45 else 7 if duration <= 60 else 8
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


# --- card artwork -----------------------------------------------------------


def _hex(value: str) -> tuple[int, int, int]:
    raw = value.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _lerp_color(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * ratio)) for a, b in zip(start, end))


def _vertical_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        draw.line([(0, y), (width, y)], fill=_lerp_color(top, bottom, y / max(1, height - 1)))
    return image


def _glow_blobs(image: Image.Image, accent: str, pale: str) -> Image.Image:
    """Soft out-of-focus light spots make flat cards look lit, not clip-art flat."""
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse((-width * .45, -height * .16, width * .55, height * .38), fill=(*_hex(accent), 165))
    draw.ellipse((width * .55, height * .62, width * 1.45, height * 1.12), fill=(*_hex(accent), 70))
    draw.ellipse((width * .62, -height * .08, width * 1.15, height * .16), fill=(*_hex(pale), 46))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(12, height // 11)))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def _legibility_shade(image: Image.Image) -> Image.Image:
    """Darken the caption strip and the top edge so white text stays readable."""
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bottom_start = round(height * .60)
    for y in range(bottom_start, height):
        alpha = round(150 * (y - bottom_start) / max(1, height - bottom_start))
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    top_end = round(height * .14)
    for y in range(top_end):
        alpha = round(95 * (1 - y / max(1, top_end)))
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


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


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    spacing: int = 8,
    stroke: int = 0,
    stroke_fill: str = "#000000",
) -> None:
    left, top, right, bottom = box
    wrapped = _wrap(draw, text, font, right - left)
    bounds = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align="center")
    x = left + ((right - left) - (bounds[2] - bounds[0])) / 2
    y = top + max(0, ((bottom - top) - (bounds[3] - bounds[1])) / 2)
    draw.multiline_text(
        (x, y), wrapped, font=font, fill=fill, spacing=spacing, align="center",
        stroke_width=stroke, stroke_fill=stroke_fill,
    )


def _progress_dots(draw: ImageDraw.ImageDraw, size: tuple[int, int], index: int, total: int, accent: str, pale: str) -> None:
    width, height = size
    radius = max(3, round(height * .0042))
    gap = radius * 4
    row_width = (total - 1) * gap
    y = round(height * .945)
    start_x = (width - row_width) / 2
    for dot in range(total):
        x = start_x + dot * gap
        box = (x - radius, y - radius, x + radius, y + radius)
        if dot == index:
            draw.ellipse(box, fill=accent)
        else:
            draw.ellipse(box, outline=pale, width=max(2, radius // 3))


def create_cover_image(story: dict[str, Any], output_dir: Path, height: int) -> Path:
    """Generate an original square/vertical-safe cover without external imagery."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width = int(round(height * 9 / 16 / 2) * 2)
    dark, accent, pale = PALETTES[0]
    base = _vertical_gradient((width, height), _hex(dark), _lerp_color(_hex(dark), (0, 0, 0), .45))
    image = _glow_blobs(base, accent, pale)
    image = _legibility_shade(image)
    draw = ImageDraw.Draw(image)
    margin = round(width * .085)
    chip_top, chip_bottom = round(height * .08), round(height * .16)
    draw.rounded_rectangle((margin, chip_top, width - margin, chip_bottom), radius=14, fill=accent)
    chip_font = _font(max(15, round(height * .018)), bold=True)
    chip_label = "CLIPLAB ORIGINAL STORY"
    label_box = draw.textbbox((0, 0), chip_label, font=chip_font)
    draw.text(
        ((width - (label_box[2] - label_box[0])) / 2, (chip_top + chip_bottom - (label_box[3] - label_box[1])) / 2 - label_box[1]),
        chip_label, font=chip_font, fill=dark,
    )
    _draw_centered(
        draw,
        (margin, round(height * .24), width - margin, round(height * .60)),
        story["title"],
        _font(max(34, round(height * .058)), bold=True),
        "#ffffff",
        spacing=round(height * .012),
        stroke=max(2, round(height * .0025)),
        stroke_fill="#0b0f14",
    )
    bar_width = round(width * .32)
    draw.rounded_rectangle(
        ((width - bar_width) / 2, round(height * .665), (width + bar_width) / 2, round(height * .665) + max(5, round(height * .007))),
        radius=max(3, round(height * .003)), fill=accent,
    )
    _draw_centered(
        draw,
        (margin, round(height * .73), width - margin, round(height * .90)),
        story["hook"],
        _font(max(17, round(height * .022))),
        pale,
        spacing=round(height * .007),
        stroke=max(1, round(height * .0012)),
        stroke_fill="#0b0f14",
    )
    path = output_dir / "cover.png"
    image.save(path, optimize=True)
    return path


def create_scene_cards(story: dict[str, Any], output_dir: Path, height: int, *, oversample: int = CARD_OVERSAMPLE) -> list[Path]:
    """Render scene cards at 2x: ffmpeg zooms into them, so extra pixels keep it crisp."""
    unit_height = height * max(1, oversample)
    width = int(round(unit_height * 9 / 16 / 2) * 2)
    slides_dir = output_dir / "scenes"
    slides_dir.mkdir(parents=True, exist_ok=True)
    total = len(story["scenes"])
    paths: list[Path] = []
    for index, scene in enumerate(story["scenes"], start=1):
        dark, accent, pale = PALETTES[(index - 1) % len(PALETTES)]
        base = _vertical_gradient((width, unit_height), _hex(dark), _lerp_color(_hex(dark), (0, 0, 0), .5))
        image = _glow_blobs(base, accent, pale)
        image = _legibility_shade(image)
        draw = ImageDraw.Draw(image)
        margin = round(width * .09)
        # Accent chip with the scene counter anchors the editorial layout.
        chip_top, chip_bottom = round(unit_height * .085), round(unit_height * .145)
        chip_font = _font(max(16, round(unit_height * .018)), bold=True)
        chip_label = f"STORY  ·  {index:02}/{total:02}"
        label_box = draw.textbbox((0, 0), chip_label, font=chip_font)
        chip_width = (label_box[2] - label_box[0]) + round(width * .07)
        draw.rounded_rectangle(
            ((width - chip_width) / 2, chip_top, (width + chip_width) / 2, chip_bottom),
            radius=round((chip_bottom - chip_top) / 2), fill=accent,
        )
        draw.text(
            ((width - (label_box[2] - label_box[0])) / 2, (chip_top + chip_bottom - (label_box[3] - label_box[1])) / 2 - label_box[1]),
            chip_label, font=chip_font, fill=dark,
        )
        _draw_centered(
            draw, (margin, round(unit_height * .26), width - margin, round(unit_height * .58)), scene["on_screen_text"],
            _font(max(30, round(unit_height * .056)), bold=True), "#ffffff", spacing=round(unit_height * .011),
            stroke=max(2, round(unit_height * .0022)), stroke_fill="#0b0f14",
        )
        bar_width = round(width * .30)
        draw.rounded_rectangle(
            ((width - bar_width) / 2, round(unit_height * .63), (width + bar_width) / 2, round(unit_height * .63) + max(5, round(unit_height * .006))),
            radius=max(3, round(unit_height * .003)), fill=accent,
        )
        _draw_centered(
            draw, (margin, round(unit_height * .68), width - margin, round(unit_height * .80)), scene["visual_prompt"],
            _font(max(16, round(unit_height * .019))), pale, spacing=round(unit_height * .006),
            stroke=max(1, round(unit_height * .0011)), stroke_fill="#0b0f14",
        )
        _progress_dots(draw, (width, unit_height), index - 1, total, accent, pale)
        path = slides_dir / f"scene-{index:02}.jpg"
        image.save(path, quality=92, optimize=True)
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
            # A slower rate and small word gaps keep the synthetic fallback understandable.
            command = [espeak, "-v", voice, "-s", "140", "-g", "6", "-w", str(output_file), "-f", str(text_file)]
            log("Озвучиваю через локальный eSpeak NG. Для естественного голоса установите Piper-модель: python -m app.download_voice dmitri")
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


# --- animated assembly ------------------------------------------------------


def _scene_timeline(story: dict[str, Any], duration: float, fade: float, fps: int) -> tuple[list[Word], list[float]]:
    """Word timings plus per-scene lengths on the finished crossfaded timeline.

    Each scene is lengthened by its share of the crossfade overlap, so the final
    video lasts about as long as the narration instead of shrinking by
    (scenes - 1) * fade. Word cursors advance by scene_length - fade to match.
    """
    scene_tokens = [scene["narration"].split() for scene in story["scenes"]]
    all_count = sum(len(tokens) for tokens in scene_tokens) or 1
    count = len(scene_tokens)
    fade_share = fade * (count - 1) / count if count else 0.0
    lengths: list[float] = []
    for tokens in scene_tokens:
        raw = max(2.8, duration * len(tokens) / all_count) + fade_share
        lengths.append(round(raw * fps) / fps)
    words: list[Word] = []
    cursor = 0.0
    for tokens, scene_length in zip(scene_tokens, lengths):
        step = scene_length / max(1, len(tokens))
        for index, token in enumerate(tokens):
            words.append(Word(cursor + index * step, cursor + (index + 1) * step * .92, token))
        cursor += scene_length - fade
    return words, lengths


def build_story_filtergraph(
    scene_lengths: list[float],
    width: int,
    height: int,
    *,
    fade: float = CROSSFADE_SECONDS,
    fps: int = STORY_FPS,
    zoom: float = KEN_BURNS_ZOOM,
    subtitle_path: str | None = None,
) -> str:
    """Ken Burns zoom per card, crossfades between scenes, optional burned captions.

    Scene cards are rendered at 2x size; zoompan crops the output window from the
    oversized source, which is the standard trick against zoompan pixel jitter.
    """
    branches: list[str] = []
    for index, length in enumerate(scene_lengths):
        frames = max(fps, int(round(length * fps)))
        if index % 2 == 0:
            zoom_expr = f"1+{zoom}*on/{frames}"
        else:
            zoom_expr = f"1+{zoom}-{zoom}*on/{frames}"
        branches.append(
            f"[{index}:v]zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={width}x{height}:fps={fps},format=yuv420p,setsar=1,settb=AVTB[v{index}]"
        )
    current = "v0"
    elapsed = scene_lengths[0] if scene_lengths else 0.0
    for index in range(1, len(scene_lengths)):
        offset = elapsed - fade
        branches.append(
            f"[{current}][v{index}]xfade=transition=fade:duration={fade:.3f}:offset={max(0.0, offset):.3f}[x{index}]"
        )
        current = f"x{index}"
        elapsed = offset + scene_lengths[index]
    if subtitle_path:
        branches.append(f"[{current}]subtitles=filename='{subtitle_path}'[vout]")
    else:
        branches.append(f"[{current}]format=yuv420p[vout]")
    return ";".join(branches)


def render_story(story: dict[str, Any], job: dict[str, Any], settings: Settings, output_dir: Path, log: Log) -> dict[str, Any]:
    height = job["render_height"]
    width = int(round(height * 9 / 16 / 2) * 2)
    narration = " ".join(scene["narration"] for scene in story["scenes"])
    voice_file = output_dir / "narration.wav"
    voice_engine = synthesize_narration(narration, job["language"], voice_file, settings, log)
    audio_duration = probe_duration(voice_file, log)
    words, scene_lengths = _scene_timeline(story, audio_duration, CROSSFADE_SECONDS, STORY_FPS)
    cover = create_cover_image(story, output_dir, height)
    cards = create_scene_cards(story, output_dir, height)
    srt_file = output_dir / "story.srt"
    ass_file = output_dir / "story.ass"
    write_captions(words, 0, width, height, srt_file, ass_file)
    subtitle = str(ass_file).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    graph = build_story_filtergraph(scene_lengths, width, height, subtitle_path=subtitle)
    output_video = output_dir / "story-reel.mp4"
    audio_index = len(cards)
    fade_out_start = max(0.0, audio_duration - 0.8)
    graph += (
        f";[{audio_index}:a]aresample=44100,afade=t=in:st=0:d=0.25,"
        f"afade=t=out:st={fade_out_start:.3f}:d=0.8[aout]"
    )
    command = [ffmpeg_executable(), "-y"]
    for card in cards:
        command.extend(["-i", str(card)])
    command.extend(["-i", str(voice_file)])
    command.extend([
        "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]", "-shortest",
        "-threads", str(settings.ffmpeg_threads), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output_video),
    ])
    log("Собираю анимированный Story Reel: плавный зум карточек, переходы и вшитые субтитры…")
    run_command(command, log)
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
