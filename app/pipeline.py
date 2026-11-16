"""Offline-first ingest, transcription, moment selection and rendering.

No cloud credential is used here.  Whisper downloads its open model on first
use; an optional Ollama endpoint may improve semantic selection but is not
required for a completed job.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from .errors import PipelineError
from .llm import chat_completion
from .settings import Settings
from .storage import write_json

Log = Callable[[str], None]

ALLOWED_HOSTS = ("youtube.com", "youtu.be", "twitch.tv")
LAUGHTER = re.compile(
    r"(?:ха(?:ха|х){1,}|хе(?:хе|х){1,}|ахах|смех|сме[ёе]т|рж[её]|угар|рофл|лол|"
    r"haha+|heh+|laugh(?:ing|ter)?|lmao|lol)",
    re.IGNORECASE,
)
REACTIONS = re.compile(
    r"(?:что|чего|как так|не может быть|блин|офиг|жесть|вау|господи|почему|"
    r"what|no way|wait|oh my|really|damn|wtf|why)",
    re.IGNORECASE,
)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    start: float
    end: float
    score: float
    title: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "score": round(self.score, 2),
            "title": self.title,
            "reason": self.reason,
        }


def validate_source_url(url: str) -> str:
    """Restrict downloader input to the promised services, avoiding SSRF."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not host:
        raise PipelineError("Укажите полную HTTPS-ссылку на YouTube или Twitch.")
    if not any(host == domain or host.endswith("." + domain) for domain in ALLOWED_HOSTS):
        raise PipelineError("Поддерживаются только YouTube и Twitch ссылки (VOD или клип).")
    return parsed.geturl()


def run_command(command: list[str], log: Log, *, cwd: Path | None = None) -> None:
    """Run a command and stream a bounded, useful copy of its output to the job."""
    printable = " ".join(command[:3])
    log(f"Запуск: {printable} …")
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    tail: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        tail.append(line)
        tail = tail[-12:]
        # yt-dlp and ffmpeg can flood the UI with progress bars; milestones still help.
        if any(marker in line.lower() for marker in ("error", "destination", "merging", "frame=", "download")):
            log(line[-300:])
    status = process.wait()
    if status:
        details = "\n".join(tail)[-1600:]
        raise PipelineError(f"Команда не выполнилась (код {status}). {details}")


def ffmpeg_executable() -> str:
    """Prefer the system FFmpeg, with a wheel fallback for a bare pip install."""
    if binary := shutil.which("ffmpeg"):
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        # Keep the ordinary command name in the error if a deployment omitted both.
        return "ffmpeg"


def probe_duration(source: Path, log: Log) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(source),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        value = result.stdout.strip()
    else:
        # imageio-ffmpeg intentionally bundles ffmpeg but not ffprobe. `-i`
        # prints a reliable Duration line without decoding the entire media file.
        result = subprocess.run([ffmpeg_executable(), "-i", str(source)], capture_output=True, text=True, check=False)
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
        value = str(int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))) if match else ""
    try:
        duration = float(value)
    except ValueError as error:
        raise PipelineError("Не удалось прочитать длительность загруженного видео.") from error
    if not math.isfinite(duration) or duration <= 0:
        raise PipelineError("Источник не содержит корректного видео.")
    log(f"Длительность источника: {duration / 60:.1f} мин.")
    return duration


def downloader_network_error(message: str) -> str | None:
    """Turn noisy yt-dlp transport output into an actionable user-facing diagnostic."""
    lowered = message.lower()
    if any(marker in lowered for marker in (
        "could not resolve host", "dns server returned", "name or service not known",
        "temporary failure in name resolution",
    )):
        return (
            "Сервер не может разрешить DNS-адрес YouTube. Это проблема DNS/интернета у сервера "
            "или Docker, а не видео и не Whisper. В preview-песочнице внешние видеохостинги могут "
            "быть недоступны. На своём Windows-ПК проверьте, что YouTube открывается в браузере, "
            "перезапустите Docker Desktop и контейнер. Если ошибка остаётся только в Docker — "
            "настройте разрешённый DNS/прокси вашей сети; не добавляйте случайный proxy."
        )
    if any(marker in lowered for marker in (
        "tls/ssl connection", "sslerror", "ssl_error_syscall", "certificate verify",
        "connection has been closed", "connection closed abruptly",
    )):
        return (
            "Сервер не смог установить HTTPS-соединение с YouTube. Это проблема сети/прокси "
            "текущего сервера, а не видео или Whisper. В этой preview-песочнице внешние "
            "видеохостинги могут быть заблокированы. Для своего сервера: пересоберите образ с "
            "актуальным yt-dlp, оставьте YTDLP_IMPERSONATE=chrome; при корпоративной сети "
            "укажите YTDLP_PROXY. Видео и cookies никуда дополнительно не отправляются."
        )
    return None


def download_source(url: str, source_dir: Path, settings: Settings, log: Log) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    template = str(source_dir / "source.%(ext)s")
    command = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist", "--no-progress", "--restrict-filenames",
        "--retries", "3", "--extractor-retries", "3", "--socket-timeout", "30",
        "--merge-output-format", "mp4", "--remux-video", "mp4",
        "-f", "bv*+ba/b", "-o", template,
    ]
    if settings.ytdlp_impersonate:
        command.extend(["--impersonate", settings.ytdlp_impersonate])
    if settings.ytdlp_proxy:
        command.extend(["--proxy", settings.ytdlp_proxy])
    if settings.ytdlp_force_ipv4:
        command.append("--force-ipv4")
    # A weak CPU can be made unusable by an accidental multi-hour VOD or 40 GB source.
    # Set either value to 0 in .env to explicitly remove that guard.
    if settings.max_source_minutes:
        command.extend(["--match-filter", f"duration <= {settings.max_source_minutes * 60}"])
    if settings.max_source_mb:
        command.extend(["--max-filesize", f"{settings.max_source_mb}M"])
    command.append(url)
    if settings.ytdlp_cookies_file:
        cookie_path = Path(settings.ytdlp_cookies_file)
        if cookie_path.is_file():
            command[3:3] = ["--cookies", str(cookie_path)]
        else:
            log("Файл cookies из YTDLP_COOKIES_FILE не найден; пробую без него.")
    try:
        run_command(command, log)
    except PipelineError as error:
        message = str(error)
        lowered = message.lower()
        friendly = downloader_network_error(message)
        if friendly:
            raise PipelineError(friendly) from error
        if "Sign in to confirm" in message or "bot" in lowered:
            raise PipelineError(
                "YouTube запросил проверку доступа. Обновите yt-dlp и, если у вас есть право "
                "просмотра, укажите Netscape cookies через YTDLP_COOKIES_FILE."
            ) from error
        if "does not pass" in message or "max-filesize" in lowered:
            raise PipelineError(
                "Источник превышает безопасный лимит этого профиля. Измените MAX_SOURCE_MINUTES "
                "или MAX_SOURCE_MB в .env, если на диске достаточно места."
            ) from error
        raise
    files = [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not files:
        raise PipelineError("yt-dlp завершился, но видеофайл не был создан.")
    return max(files, key=lambda path: path.stat().st_size)


def transcribe(source: Path, settings: Settings, log: Log) -> list[Word]:
    """Use faster-whisper locally, keeping word timestamps for accurate captions."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # Defensive: it is a declared dependency.
        raise PipelineError("Не найден faster-whisper. Пересоберите контейнер.") from error

    log(
        f"Загружаю локальную Whisper-модель {settings.whisper_model} "
        f"({settings.whisper_cpu_threads} CPU-потоков, {settings.whisper_compute_type}; первый запуск может занять время)…"
    )
    try:
        model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads,
            num_workers=1,
            download_root=str(settings.model_cache_dir),
        )
        segments, info = model.transcribe(
            str(source),
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        log(f"Распознан язык: {info.language} ({info.language_probability:.0%} уверенности).")
        words: list[Word] = []
        for segment in segments:
            segment_words = list(segment.words or [])
            if segment_words:
                for item in segment_words:
                    text = (item.word or "").strip()
                    if text:
                        words.append(Word(float(item.start), float(item.end), text))
            else:
                # Some media has segment timings but no word alignment. Keep it usable.
                tokens = (segment.text or "").strip().split()
                if tokens:
                    step = (float(segment.end) - float(segment.start)) / len(tokens)
                    words.extend(
                        Word(float(segment.start) + index * step, float(segment.start) + (index + 1) * step, token)
                        for index, token in enumerate(tokens)
                    )
        if len(words) < 8:
            raise PipelineError("В речи слишком мало распознанных слов для выбора моментов.")
        return words
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(f"Локальная расшифровка Whisper не удалась: {error}") from error


def _sentences(words: list[Word], maximum: float) -> list[list[Word]]:
    result: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        sentence_length = word.end - current[0].start
        terminal = bool(re.search(r"[.!?…]$", word.text))
        if terminal or sentence_length >= maximum:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def _title(words: Iterable[Word]) -> str:
    text = " ".join(word.text for word in words).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 72:
        text = text[:69].rstrip(" ,.;:!?") + "…"
    return text or "Яркий момент"


def build_candidates(words: list[Word], minimum: int, maximum: int) -> list[Candidate]:
    """Build coherent speech windows, then score reactions without a paid model."""
    sentences = _sentences(words, maximum)
    candidates: list[Candidate] = []
    for first in range(len(sentences)):
        block: list[Word] = []
        for last in range(first, len(sentences)):
            proposed = block + sentences[last]
            duration = proposed[-1].end - proposed[0].start
            if duration > maximum + 1.5:
                break
            block = proposed
            if duration < minimum:
                continue
            text = " ".join(word.text for word in block)
            laughter = len(LAUGHTER.findall(text))
            reactions = len(REACTIONS.findall(text))
            punctuation = sum(word.text.count("!") + word.text.count("?") for word in block)
            lexical = len({re.sub(r"\W", "", word.text.lower()) for word in block if len(word.text) > 3})
            words_per_second = len(block) / max(duration, 1)
            pace_bonus = max(0, 1.4 - abs(words_per_second - 2.6)) * 0.6
            score = 1.0 + laughter * 4.4 + reactions * 1.8 + min(punctuation, 5) * 0.3 + min(lexical / 40, 1.2) + pace_bonus
            reason_bits = []
            if laughter:
                reason_bits.append("смех / шутливая реакция")
            if reactions:
                reason_bits.append("эмоциональная реакция")
            if punctuation >= 2:
                reason_bits.append("экспрессивная речь")
            reason = ", ".join(reason_bits) or "цельный разговорный фрагмент"
            candidates.append(Candidate(block[0].start, block[-1].end, score, _title(block), reason))
            # A second window at the same sentence start adds little value.
            if duration >= minimum + 12:
                break
    return candidates


def _extract_json(text: str) -> object | None:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", clean)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def choose_with_configured_llm(
    candidates: list[Candidate],
    wanted: int,
    settings: Settings,
    log: Log,
    *,
    allow_cloud: bool,
) -> list[Candidate] | None:
    """Rank candidates with the configured model, preserving local-only as default."""
    target = settings.active_llm
    if not target:
        return None
    if target.is_cloud and not allow_cloud:
        log("Облачный смысловой выбор отключён: разрешение на передачу краткого текста не дано; использую локальный скоринг.")
        return None
    shortlist = sorted(candidates, key=lambda item: item.score, reverse=True)[:80]
    digest = "\n".join(
        f"{index + 1}. {item.start:.1f}-{item.end:.1f}s | {item.title}" for index, item in enumerate(shortlist)
    )
    prompt = (
        "Ты редактор коротких роликов. Из списка таймкодных фрагментов выбери самые смешные, "
        "неловкие, неожиданные или эмоционально вирусные. Не выдумывай таймкоды и выбери только "
        f"номера из списка. Нужны {wanted} непересекающихся фрагментов. Верни только JSON-массив "
        "объектов вида {\"id\": 3, \"title\": \"короткий заголовок\", \"reason\": \"почему\"}.\n\n" + digest
    )
    location = "Облачная" if target.is_cloud else "Локальная"
    log(f"{location} языковая модель оценивает смысл и реакции…")
    try:
        content = chat_completion(
            target,
            [
                {"role": "system", "content": "Отвечай только корректным JSON, без markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=180,
        )
        result = _extract_json(content)
        if not isinstance(result, list):
            raise ValueError("модель не вернула JSON-массив")
        picks: list[Candidate] = []
        seen: set[int] = set()
        for item in result:
            if not isinstance(item, dict):
                continue
            index = item.get("id")
            if not isinstance(index, int) or not 1 <= index <= len(shortlist) or index in seen:
                continue
            seen.add(index)
            base = shortlist[index - 1]
            title = str(item.get("title") or base.title).strip()[:90]
            reason = str(item.get("reason") or base.reason).strip()[:180]
            picks.append(Candidate(base.start, base.end, base.score + 2, title, reason))
        if picks:
            return picks
        raise ValueError("модель не выбрала валидные фрагменты")
    except Exception as error:
        log(f"{location} LLM недоступна или дала неверный ответ ({error}); использую локальный скоринг.")
        return None


def non_overlapping_best(candidates: list[Candidate], wanted: int) -> list[Candidate]:
    picked: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        # one second breathing room preserves complete captions at boundaries
        conflict = any(candidate.start < item.end + 1 and candidate.end + 1 > item.start for item in picked)
        if not conflict:
            picked.append(candidate)
        if len(picked) >= wanted:
            break
    return sorted(picked, key=lambda item: item.start)


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_value, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_value:02},{milliseconds:03}"


def ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds_value, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{seconds_value:02}.{centiseconds:02}"


def clip_caption_groups(words: list[Word], clip_start: float) -> list[tuple[float, float, str]]:
    groups: list[tuple[float, float, str]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        text_length = sum(len(item.text) + 1 for item in current)
        terminal = bool(re.search(r"[.!?…]$", word.text))
        if len(current) >= 3 or text_length >= 26 or terminal:
            groups.append((current[0].start - clip_start, current[-1].end - clip_start, " ".join(item.text for item in current)))
            current = []
    if current:
        groups.append((current[0].start - clip_start, current[-1].end - clip_start, " ".join(item.text for item in current)))
    return groups


def write_captions(words: list[Word], clip_start: float, width: int, height: int, srt_file: Path, ass_file: Path) -> None:
    groups = clip_caption_groups(words, clip_start)
    if not groups:
        groups = [(0.0, 1.0, "[нет распознанной речи]")]
    srt_lines: list[str] = []
    ass_lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "",
        "[V4+ Styles]",
        f"Style: Caption,DejaVu Sans,{max(28, round(height * .051))},&H00FFFFFF,&H000000FF,&H00101010,&H00000000,1,1,0,0,100,100,0,0,1,{max(2, round(height * .004))},1,2,{round(height * .78)},40,40,1",
        "", "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for index, (start, end, text) in enumerate(groups, start=1):
        end = max(end, start + .15)
        srt_lines.extend([str(index), f"{srt_timestamp(start)} --> {srt_timestamp(end)}", text, ""])
        # Escape ASS control syntax while keeping normal Unicode text untouched.
        ass_text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        ass_lines.append(f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},Caption,,0,0,0,,{ass_text}")
    srt_file.write_text("\n".join(srt_lines), encoding="utf-8")
    ass_file.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")


def render_clip(
    source: Path,
    candidate: Candidate,
    all_words: list[Word],
    clip_number: int,
    output_dir: Path,
    height: int,
    settings: Settings,
    log: Log,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    width = int(round(height * 9 / 16 / 2) * 2)
    clip_name = f"clip-{clip_number:02}"
    mp4_file = output_dir / f"{clip_name}.mp4"
    srt_file = output_dir / f"{clip_name}.srt"
    ass_file = output_dir / f"{clip_name}.ass"
    words = [word for word in all_words if word.end >= candidate.start and word.start <= candidate.end]
    write_captions(words, candidate.start, width, height, srt_file, ass_file)
    # Input is centered over a blurred 9:16 copy so no speech/presenter is cropped away.
    subtitles = str(ass_file).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    filters = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=20:10[blur];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[front];"
        f"[blur][front]overlay=(W-w)/2:(H-h)/2,subtitles=filename='{subtitles}'[video]"
    )
    duration = max(.25, candidate.end - candidate.start)
    command = [
        ffmpeg_executable(), "-y", "-ss", f"{candidate.start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-filter_complex", filters, "-map", "[video]", "-map", "0:a?",
        "-threads", str(settings.ffmpeg_threads), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(mp4_file),
    ]
    log(f"Рендер {clip_number}: {candidate.start:.0f}–{candidate.end:.0f} сек, вертикальное видео с субтитрами…")
    run_command(command, log)
    if not mp4_file.is_file() or mp4_file.stat().st_size < 1024:
        raise PipelineError(f"Не был создан готовый файл {clip_name}.")
    return {
        **candidate.as_dict(),
        "duration": round(duration, 2),
        "video": mp4_file.name,
        "subtitles": srt_file.name,
    }


def run_pipeline(job: dict, settings: Settings, log: Log) -> list[dict]:
    """Execute the complete local pipeline and return finished clip metadata."""
    job_dir = settings.jobs_dir / job["id"]
    source_dir = job_dir / "source"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    url = validate_source_url(job["source_url"])
    log("Проверяю и загружаю источник через yt-dlp…")
    source = download_source(url, source_dir, settings, log)
    duration = probe_duration(source, log)
    if duration < job["min_seconds"]:
        raise PipelineError("Видео короче минимальной длительности клипа.")

    log("Распознаю речь локально: никакое аудио не отправляется в облако.")
    words = transcribe(source, settings, log)
    write_json(job_dir / "transcript.json", {"words": [asdict(word) for word in words]})
    full_text = " ".join(word.text for word in words)
    (job_dir / "transcript.txt").write_text(full_text, encoding="utf-8")

    log("Ищу потенциально смешные, нелепые и эмоциональные моменты…")
    candidates = build_candidates(words, job["min_seconds"], job["max_seconds"])
    if not candidates:
        raise PipelineError("Не удалось собрать фрагменты нужной длительности из распознанной речи.")
    llm_picks = choose_with_configured_llm(
        candidates,
        job["clip_count"],
        settings,
        log,
        allow_cloud=bool(job.get("cloud_processing_allowed")),
    )
    selected = non_overlapping_best(llm_picks or candidates, job["clip_count"])
    if not selected:
        raise PipelineError("Не удалось выбрать непересекающиеся моменты.")
    log(f"Выбрано моментов: {len(selected)} из {len(candidates)} кандидатов.")

    clips = [
        render_clip(source, candidate, words, index, output_dir, job["render_height"], settings, log)
        for index, candidate in enumerate(selected, start=1)
    ]
    write_json(output_dir / "clips.json", clips)
    log("Готово: MP4 уже содержит вшитые субтитры, рядом доступен SRT.")
    return clips


def remove_job_files(job_dir: Path) -> None:
    # The caller constructs job_dir from a server-generated UUID, never user input.
    shutil.rmtree(job_dir, ignore_errors=True)
