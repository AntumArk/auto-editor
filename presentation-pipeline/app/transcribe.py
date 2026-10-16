"""Local speech transcription with source-time subtitle output."""

import json
import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from .subtitles import write_srt
from .utils import release_gpu_memory


def _setting(config, name, default):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _align_words(video_path, result, settings, device):
    model = metadata = audio = None
    try:
        import whisperx

        model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device,
            model_name=_setting(settings, "align_model", None),
        )
        audio = whisperx.load_audio(str(video_path))
        aligned = whisperx.align(
            deepcopy(result["segments"]), model, metadata, audio, device,
            return_char_alignments=False,
        )
        segments = aligned["segments"]
        if len(segments) != len(result["segments"]):
            raise ValueError("Alignment changed the number of segments")
        updated = deepcopy(result)
        for original, refined in zip(updated["segments"], segments):
            for key in ("start", "end"):
                original[key] = refined.get(key, original[key])
            if refined.get("words"):
                original["words"] = [
                    {
                        **word,
                        "start": word.get("start", original["start"]),
                        "end": word.get("end", original["end"]),
                    }
                    for word in refined["words"]
                ]
        return updated
    except Exception:
        logging.getLogger(__name__).warning(
            "WhisperX alignment unavailable or failed; keeping Whisper timestamps.",
            exc_info=True,
        )
        return result
    finally:
        model = metadata = audio = None
        release_gpu_memory()


def transcribe(video_path, cfg, out_dir) -> dict:
    """Transcribe, release Whisper, then optionally align and label speakers."""
    from faster_whisper import WhisperModel

    settings = _setting(cfg, "transcription", {})
    model_name = _setting(settings, "model", "large-v3")
    device = _setting(settings, "device", "auto")
    compute_type = _setting(settings, "compute_type", "auto")
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("transcription.device must be auto, cuda, or cpu")
    vad_filter = _setting(settings, "vad_filter", None)
    options = {
        "language": _setting(settings, "language", "lt"),
        "task": _setting(settings, "task", "transcribe"),
        "vad_filter": _setting(settings, "vad", True) if vad_filter is None else vad_filter,
        "word_timestamps": _setting(settings, "word_timestamps", True),
        "beam_size": _setting(settings, "beam_size", 5),
        "condition_on_previous_text": _setting(settings, "condition_on_previous_text", False),
        "initial_prompt": _setting(settings, "initial_prompt", None),
    }
    vad_parameters = _setting(settings, "vad_parameters", None)
    if vad_parameters is not None:
        options["vad_parameters"] = vad_parameters

    def run(device, compute_type):
        model = segments = None
        try:
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
        finally:
            segments = model = None
            release_gpu_memory()

    if device == "cpu":
        attempts = [("cpu", "int8" if compute_type == "auto" else compute_type)]
    else:
        cuda_compute = "int8_float16" if compute_type == "auto" else compute_type
        attempts = list(dict.fromkeys([
            ("cuda", cuda_compute), ("cuda", "int8_float16"), ("cpu", "int8"),
        ]))
    for index, (target, precision) in enumerate(attempts):
        try:
            result = run(target, precision)
            break
        except Exception:
            if index == len(attempts) - 1:
                raise
            logging.getLogger(__name__).warning(
                "Transcription failed on %s/%s; retrying on %s/%s.",
                target, precision, *attempts[index + 1],
            )

    if _setting(settings, "align", False):
        result = _align_words(video_path, result, settings, target)
    diarization = _setting(cfg, "diarization", {})
    if _setting(diarization, "enabled", False):
        from .diarize import diarize

        result["segments"] = diarize(video_path, result["segments"], diarization)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcript.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_srt(
        result["segments"], output / f"{Path(video_path).stem}.source.srt",
        speaker_prefix=_setting(diarization, "speaker_prefix", False),
    )
    return result
