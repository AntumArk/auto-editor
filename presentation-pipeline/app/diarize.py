"""Optional speaker diarization and dependency-free time-overlap assignment."""

import os
from collections.abc import Mapping
from copy import deepcopy


def assign_speakers(segments, turns, fill_nearest=False):
    """Copy segments/words and label by total overlap; ties use first turn order.

    Turns are dictionaries with start/end (seconds) and speaker. Without overlap
    an item stays unlabeled, unless nearest-turn filling is explicitly enabled.
    """
    turns = list(turns)

    def assign(item):
        totals = {}
        for turn in turns:
            overlap = max(
                0, min(item["end"], turn["end"]) - max(item["start"], turn["start"])
            )
            if overlap > 0:
                speaker = turn["speaker"]
                totals[speaker] = totals.get(speaker, 0) + overlap
        if totals:
            item["speaker"] = max(totals, key=totals.get)
        elif fill_nearest and turns:
            nearest = min(
                turns,
                key=lambda turn: max(
                    turn["start"] - item["end"], item["start"] - turn["end"], 0
                ),
            )
            item["speaker"] = nearest["speaker"]

    result = deepcopy(segments)
    for segment in result:
        assign(segment)
        for word in segment.get("words") or []:
            assign(word)
    return result


def diarize(video_path, segments, settings):
    """Run gated pyannote 3.1 after Whisper has been released."""
    def setting(name, default=None):
        if isinstance(settings, Mapping):
            return settings.get(name, default)
        return getattr(settings, name, default)

    token_env = setting("hf_token", "HF_TOKEN")
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            "Diarization requires a Hugging Face token in the configured "
            "environment variable and acceptance of both pyannote model terms."
        )
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Diarization requires requirements-extra.txt (pyannote.audio)."
        ) from exc

    from .utils import release_gpu_memory

    pipeline = annotation = None
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token,
        )
        pipeline.to(torch.device(setting("device", "cpu")))
        options = {
            name: setting(name)
            for name in ("min_speakers", "max_speakers")
            if setting(name) is not None
        }
        annotation = pipeline(str(video_path), **options)
        annotation = getattr(annotation, "speaker_diarization", annotation)
        turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        return assign_speakers(segments, turns, setting("fill_nearest", False))
    finally:
        pipeline = annotation = None
        release_gpu_memory()
