from pathlib import Path

from app.settings import Settings
from app.story import create_scene_cards, generate_script, render_story, write_content_pack


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": """{
              \"title\": \"Старое письмо\",
              \"hook\": \"В книге лежало письмо без адреса.\",
              \"description\": \"Создано с помощью ИИ.\",
              \"hashtags\": [\"история\", \"shorts\"],
              \"disclosure\": \"Создано с помощью ИИ; история вымышленная.\",
              \"scenes\": [
                {\"narration\":\"В старой библиотеке герой заметил книгу без названия.\",\"on_screen_text\":\"Книга без названия\",\"visual_prompt\":\"Тихая библиотека\"},
                {\"narration\":\"Между страницами лежало письмо, адресованное никому.\",\"on_screen_text\":\"Письмо\",\"visual_prompt\":\"Конверт на столе\"},
                {\"narration\":\"Внутри было всего два слова: не бойся начинать.\",\"on_screen_text\":\"Не бойся начинать\",\"visual_prompt\":\"Луч света\"},
                {\"narration\":\"Он улыбнулся и решил наконец сделать первый шаг.\",\"on_screen_text\":\"Первый шаг\",\"visual_prompt\":\"Абстрактная дорога\"}
              ]
            }"""}}]
        }


def test_local_story_script_is_normalised_and_cards_are_created(monkeypatch, tmp_path):
    settings = Settings(
        data_dir=tmp_path, whisper_model="base", whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=6, ffmpeg_threads=6, max_source_minutes=120, max_source_mb=3500,
        local_llm_url="http://ollama:11434/v1", local_llm_model="qwen2.5:3b", local_llm_api_key="",
        cloud_llm_url="", cloud_llm_model="", cloud_llm_api_key="", llm_mode="auto",
        max_workers=1, ytdlp_cookies_file="", ytdlp_proxy="", ytdlp_impersonate="chrome", ytdlp_force_ipv4=False, piper_voice_model="", piper_speaker="",
    )
    monkeypatch.setattr(
        "app.story.chat_completion",
        lambda *args, **kwargs: FakeResponse().json()["choices"][0]["message"]["content"],
    )
    job = {
        "topic": "Вымышленная история о старом письме",
        "story_kind": "fictional",
        "language": "ru",
        "duration_seconds": 45,
        "visual_style": "cinematic",
    }
    story = generate_script(job, settings, lambda _: None)
    assert story["hashtags"] == ["#история", "#shorts"]
    assert len(story["scenes"]) == 4
    cards = create_scene_cards(story, tmp_path, 720)
    assert len(cards) == 4 and all(path.is_file() for path in cards)
    pack = tmp_path / "content-pack.txt"
    write_content_pack(story, {"voice_engine": "test"}, pack)
    assert "DISCLOSURE" in pack.read_text(encoding="utf-8")


def test_story_uses_a_transparent_local_template_when_llm_is_missing(tmp_path):
    settings = Settings(
        data_dir=tmp_path, whisper_model="base", whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=6, ffmpeg_threads=6, max_source_minutes=120, max_source_mb=3500,
        local_llm_url="", local_llm_model="qwen2.5:3b", local_llm_api_key="",
        cloud_llm_url="", cloud_llm_model="", cloud_llm_api_key="", llm_mode="auto",
        max_workers=1, ytdlp_cookies_file="", ytdlp_proxy="", ytdlp_impersonate="chrome", ytdlp_force_ipv4=False, piper_voice_model="", piper_speaker="",
    )
    story = generate_script({
        "topic": "Вымышленная история о письме в старой книге", "story_kind": "fictional",
        "language": "ru", "duration_seconds": 45, "visual_style": "cinematic",
    }, settings, lambda _: None)
    assert story["script_engine"] == "Локальный сценарный шаблон (без LLM)"
    assert len(story["scenes"]) == 5
    assert story["platform_captions"]["tiktok"] == story["description"]
    assert "LLM" not in story["disclosure"]


def test_story_render_assembles_local_assets_without_cloud(monkeypatch, tmp_path):
    story = {
        "title": "Тестовая история", "hook": "Письмо ждало героя.", "description": "AI-assisted.",
        "hashtags": ["#история"], "disclosure": "Создано с помощью ИИ; история вымышленная.",
        "scenes": [
            {"narration": "Герой вошёл в тихую библиотеку и увидел старую книгу.", "on_screen_text": "Тихая библиотека", "visual_prompt": "Тёплый свет"},
            {"narration": "Внутри оказалось письмо, которое меняло всё вокруг.", "on_screen_text": "Старое письмо", "visual_prompt": "Конверт"},
            {"narration": "Он прочитал первую строчку и улыбнулся от неожиданности.", "on_screen_text": "Первая строчка", "visual_prompt": "Луч света"},
            {"narration": "Теперь он знал, что сделает первый шаг сегодня.", "on_screen_text": "Первый шаг", "visual_prompt": "Абстрактная дорога"},
        ],
    }
    settings = Settings(
        data_dir=tmp_path, whisper_model="base", whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpu_threads=6, ffmpeg_threads=6, max_source_minutes=120, max_source_mb=3500,
        local_llm_url="", local_llm_model="qwen2.5:3b", local_llm_api_key="",
        cloud_llm_url="", cloud_llm_model="", cloud_llm_api_key="", llm_mode="auto",
        max_workers=1, ytdlp_cookies_file="", ytdlp_proxy="", ytdlp_impersonate="chrome", ytdlp_force_ipv4=False, piper_voice_model="", piper_speaker="",
    )
    commands = []

    def fake_speech(text, language, output, _settings, _log):
        output.write_bytes(b"voice")
        return "test voice"

    def fake_command(command, _log, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"x" * 2048)

    monkeypatch.setattr("app.story.synthesize_narration", fake_speech)
    monkeypatch.setattr("app.story.probe_duration", lambda *_: 32.0)
    monkeypatch.setattr("app.story.run_command", fake_command)
    result = render_story(story, {"render_height": 720, "language": "ru"}, settings, tmp_path, lambda _: None)
    assert result["video"] == "story-reel.mp4"
    assert (tmp_path / result["subtitles"]).is_file()
    assert (tmp_path / result["thumbnail"]).is_file()
    assert any("fps=30" in command[command.index("-vf") + 1] for command in commands if "-vf" in command)
