"""Fetch a free local Piper voice model into the data directory.

Run inside the container (or any Python with network access):

    python -m app.download_voice dmitri

Then enable the voice in .env, for example:

    PIPER_VOICE_MODEL=/data/voices/ru_RU-dmitri-medium.onnx

and restart the service (`docker compose up -d`). Voices come from the public
rhasspy/piper-voices model repository; they are downloaded once and afterwards
narration works fully offline. No text ever leaves the machine.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

VOICES = {
    "dmitri": "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium",
    "irina": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
    "ruslan": "ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium",
    "amy": "en/en_US/amy/medium/en_US-amy-medium",
    "lessac": "en/en_US/lessac/medium/en_US-lessac-medium",
}

USER_AGENT = "cliplab-local/1.0 (+https://github.com/local-cliplab)"


def _fetch(url: str, target: Path, log) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    log(f"Скачиваю {url}")
    try:
        with urllib.request.urlopen(request, timeout=180) as response, open(target, "wb") as handle:
            shutil_copied = 0
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                shutil_copied += len(chunk)
    except OSError as error:
        target.unlink(missing_ok=True)
        raise SystemExit(f"Не удалось скачать голос: {error}")
    log(f"Готово: {target} ({shutil_copied / 1_048_576:.1f} МБ)")


def download_voice(name: str, voices_dir: Path, log=print) -> Path:
    relative = VOICES[name]
    voices_dir.mkdir(parents=True, exist_ok=True)
    model = voices_dir / f"{relative.rsplit('/', 1)[-1]}.onnx"
    config = model.with_suffix(".onnx.json")
    _fetch(BASE_URL + relative + ".onnx", model, log)
    _fetch(BASE_URL + relative + ".onnx.json", config, log)
    if model.stat().st_size < 1_000_000:
        model.unlink(missing_ok=True)
        raise SystemExit("Скачанный файл модели подозрительно мал; проверьте сеть и повторите.")
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Скачать локальный голос Piper в каталог данных.")
    parser.add_argument("voice", choices=sorted(VOICES), help="dmitri/irina/ruslan (рус.), amy/lessac (англ.)")
    parser.add_argument(
        "--dir",
        default=os.path.join(os.getenv("DATA_DIR", "./data"), "voices"),
        help="Каталог для голосов (по умолчанию $DATA_DIR/voices).",
    )
    args = parser.parse_args(argv)
    voices_dir = Path(args.dir).resolve()
    model = download_voice(args.voice, voices_dir)
    print()
    print("Голос сохранён. Включите его в .env и перезапустите сервис:")
    print()
    if str(voices_dir).startswith("/data"):
        print(f"  PIPER_VOICE_MODEL={model}")
    else:
        print(f"  PIPER_VOICE_MODEL=/data/voices/{model.name}   # путь внутри контейнера")
    print()
    print("  docker compose up -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
