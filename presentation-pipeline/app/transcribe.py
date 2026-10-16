"""Local speech transcription with source-time subtitle output."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from .subtitles import write_srt


def _setting(config, name, default):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def transcribe(video_path, cfg, out_dir) -> dict:
    """Transcribe on the configured device, with CUDA fallback in auto mode."""
    from faster_whisper import WhisperModel

    settings = _setting(cfg, "transcription", {})
    model_name = _setting(settings, "model", "large-v3")
    device = _setting(settings, "device", "auto")
    compute_type = _setting(settings, "compute_type", "auto")
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("transcription.device must be auto, cuda, or cpu")
    options = {
        "language": _setting(settings, "language", "lt"),
        "task": _setting(settings, "task", "transcribe"),
        "vad_filter": _setting(settings, "vad", True),
        "word_timestamps": _setting(settings, "word_timestamps", True),
    }

    def run(device, compute_type):
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments, info = model.transcribe(str(video_path), **options)
        return {
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability,
                        }
                        for word in (segment.words or [])
                    ],
                }
                for segment in segments
            ],
        }

    def compute_for(target):
        if compute_type != "auto":
            return compute_type
        return "float16" if target == "cuda" else "int8"

    if device == "auto":
        try:
            result = run("cuda", compute_for("cuda"))
        except Exception:
            logging.getLogger(__name__).warning(
                "CUDA transcription failed; retrying on CPU.", exc_info=True
            )
            result = run("cpu", compute_for("cpu"))
    else:
        result = run(device, compute_for(device))

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcript.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_srt(result["segments"], output / f"{Path(video_path).stem}.source.srt")
    return result
