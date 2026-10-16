from __future__ import annotations

import json
import subprocess
from pathlib import Path


def ffprobe_duration(path: str | Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = json.loads(result.stdout)["format"]["duration"]
    return float(duration)


def format_time(seconds: float, milliseconds: bool = False) -> str:
    milliseconds_value = round(max(0, seconds) * 1000)
    hours, remainder = divmod(milliseconds_value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    separator = "," if milliseconds else "."
    fraction = f"{separator}{millis:03d}" if milliseconds else ""
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{fraction}"
