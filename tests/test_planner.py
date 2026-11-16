from app.planner import run_plan_pipeline
from app.settings import Settings


def _settings(tmp_path):
    return Settings(
        data_dir=tmp_path, whisper_model="base", whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=6, ffmpeg_threads=6, max_source_minutes=120, max_source_mb=3500,
        local_llm_url="http://ollama:11434/v1", local_llm_model="qwen2.5:3b", local_llm_api_key="",
        cloud_llm_url="", cloud_llm_model="", cloud_llm_api_key="", llm_mode="auto",
        max_workers=1, ytdlp_cookies_file="", ytdlp_proxy="", ytdlp_impersonate="chrome", ytdlp_force_ipv4=False, piper_voice_model="", piper_speaker="",
    )


def test_creator_plan_is_normalised_and_written(monkeypatch, tmp_path):
    schedule = [
        {
            "day": index, "title": f"Идея {index}", "idea": f"Оригинальная идея для дня {index}",
            "hook": f"Честный хук для дня {index}", "format": "Story Reel",
            "cta": "Что бы выбрали вы?", "production_note": "Проверьте сценарий.",
        }
        for index in range(1, 8)
    ]
    monkeypatch.setattr(
        "app.planner.chat_completion",
        lambda *_args, **_kwargs: __import__("json").dumps({
            "positioning": "Оригинальные истории для спокойного просмотра.",
            "content_pillars": ["находки", "выбор", "финал"], "schedule": schedule,
            "metrics_focus": ["удержание первых секунд"], "publish_checklist": ["проверьте факты"],
        }, ensure_ascii=False),
    )
    job = {
        "id": "plan-test", "niche": "Вымышленные истории", "audience": "Зрители 18–30",
        "content_focus": "stories", "platforms": ["youtube_shorts", "tiktok"], "plan_days": 7,
        "language": "ru", "cloud_processing_allowed": False,
    }
    plan = run_plan_pipeline(job, _settings(tmp_path), lambda _: None)
    assert len(plan["schedule"]) == 7
    assert plan["file"] == "creator-plan.md"
    assert (tmp_path / "jobs" / "plan-test" / "output" / "creator-plan.md").is_file()
