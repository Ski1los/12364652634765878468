"""Creator-plan generation: a local-first editorial calendar, not a growth guarantee."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .errors import PipelineError
from .llm import chat_completion
from .settings import Settings
from .storage import write_json

Log = Callable[[str], None]


def _text(value: object, limit: int, fallback: str = "") -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value[:limit] if value else fallback


def _json_object(content: str) -> dict[str, Any] | None:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return result if isinstance(result, dict) else None


def _prompt(job: dict[str, Any]) -> str:
    language = "русском" if job["language"] == "ru" else "English"
    platforms = ", ".join(job["platforms"])
    style = {
        "stories": "оригинальные вымышленные истории",
        "education": "короткие объясняющие ролики без медицинских, финансовых или юридических советов",
        "commentary": "безопасный развлекательный комментарий без нападок на людей и бренды",
    }[job["content_focus"]]
    return f"""Ты контент-стратег для начинающего автора коротких вертикальных роликов.
Составь реалистичный {job['plan_days']}-дневный план на {language}.
Ниша: {job['niche']!r}
Аудитория: {job['audience']!r}
Площадки: {platforms}.
Фокус: {style}.

Правила:
- Идеи должны быть оригинальными, реализуемыми одним человеком и подходить для Story Reel или нарезок автора.
- Не обещай доход, просмотры, монетизацию или вирусность; не советуй накрутки, массовый спам или обход правил площадок.
- Не используй чужие истории, реальных людей, неподтверждённые факты или вводящие в заблуждение заголовки.
- Дай разнообразные, но тематически связанные идеи. Каждый хук должен быть интригующим и честным.
- Для каждого дня укажи простое действие, которое можно выполнить за одну сессию.

Верни ТОЛЬКО JSON:
{{
 "positioning":"одно предложение о позиции канала",
 "content_pillars":["рубрика 1","рубрика 2","рубрика 3"],
 "schedule":[
   {{"day":1,"title":"до 70 символов","idea":"конкретная оригинальная идея","hook":"первая честная фраза","format":"Story Reel или Clip Finder","cta":"мягкий вопрос зрителю","production_note":"что подготовить"}}
 ],
 "metrics_focus":["что смотреть после публикации без накруток"],
 "publish_checklist":["проверка фактов", "AI-маркировка при необходимости"]
}}
В schedule должно быть ровно {job['plan_days']} элементов."""


def _normalise(job: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    schedule_raw = data.get("schedule")
    if not isinstance(schedule_raw, list):
        raise PipelineError("LLM вернула контент-план без списка дней.")
    schedule: list[dict[str, Any]] = []
    for position, raw in enumerate(schedule_raw[: job["plan_days"]], start=1):
        if not isinstance(raw, dict):
            continue
        title = _text(raw.get("title"), 70)
        idea = _text(raw.get("idea"), 320)
        hook = _text(raw.get("hook"), 150)
        if not title or not idea or not hook:
            continue
        schedule.append({
            "day": position,
            "title": title,
            "idea": idea,
            "hook": hook,
            "format": _text(raw.get("format"), 40, "Story Reel"),
            "cta": _text(raw.get("cta"), 120, "Как бы вы поступили?"),
            "production_note": _text(raw.get("production_note"), 160, "Соберите черновик и проверьте перед публикацией."),
        })
    if len(schedule) < max(3, job["plan_days"] // 2):
        raise PipelineError("LLM вернула слишком мало пригодных идей. Повторите генерацию или смените модель.")
    pillars = data.get("content_pillars") if isinstance(data.get("content_pillars"), list) else []
    metrics = data.get("metrics_focus") if isinstance(data.get("metrics_focus"), list) else []
    checklist = data.get("publish_checklist") if isinstance(data.get("publish_checklist"), list) else []
    return {
        "positioning": _text(data.get("positioning"), 220, "Оригинальный короткий контент для выбранной аудитории."),
        "content_pillars": [_text(item, 80) for item in pillars if _text(item, 80)][:5],
        "schedule": schedule,
        "metrics_focus": [_text(item, 140) for item in metrics if _text(item, 140)][:5],
        "publish_checklist": [_text(item, 140) for item in checklist if _text(item, 140)][:8],
        "platforms": job["platforms"],
        "plan_days": job["plan_days"],
    }


def write_plan_markdown(plan: dict[str, Any], target: Path) -> None:
    lines = ["# Creator Plan", "", f"> {plan['positioning']}", "", "## Рубрики"]
    lines.extend(f"- {item}" for item in plan["content_pillars"])
    lines.extend(["", "## Календарь"])
    for item in plan["schedule"]:
        lines.extend([
            f"### День {item['day']} — {item['title']}",
            f"**Формат:** {item['format']}",
            f"**Идея:** {item['idea']}",
            f"**Хук:** {item['hook']}",
            f"**CTA:** {item['cta']}",
            f"**Подготовка:** {item['production_note']}",
            "",
        ])
    lines.extend(["## На что смотреть", *[f"- {item}" for item in plan["metrics_focus"]], "", "## Перед публикацией"])
    lines.extend(f"- {item}" for item in plan["publish_checklist"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_plan_pipeline(job: dict[str, Any], settings: Settings, log: Log) -> dict[str, Any]:
    target = settings.active_llm
    if not target:
        raise PipelineError("Контент-план требует локальную LLM или настроенный cloud endpoint.")
    if target.is_cloud and not job.get("cloud_processing_allowed"):
        raise PipelineError("Для cloud-плана нужно разрешение на передачу ниши, аудитории и инструкции в выбранный LLM-сервис.")
    location = "Облачная" if target.is_cloud else "Локальная"
    log(f"{location} LLM составляет реалистичный контент-план…")
    raw = chat_completion(
        target,
        [
            {"role": "system", "content": "Отвечай только валидным JSON без markdown."},
            {"role": "user", "content": _prompt(job)},
        ],
        temperature=0.7,
        timeout=240,
    )
    data = _json_object(raw)
    if not data:
        raise PipelineError("LLM вернула контент-план не в JSON-формате.")
    plan = _normalise(job, data)
    job_dir = settings.jobs_dir / job["id"]
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(job_dir / "creator-plan.json", plan)
    markdown = output_dir / "creator-plan.md"
    write_plan_markdown(plan, markdown)
    plan["file"] = markdown.name
    write_json(output_dir / "plan.json", plan)
    log("Контент-план готов: календарь идей и чек-лист сохранены локально.")
    return plan
