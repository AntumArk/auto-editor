"""Pure, conservative pause candidates in half-open media time intervals."""

from collections.abc import Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    id: str
    start: float
    end: float
    duration: float
    context: dict


def _get(value, key, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _number(value, label):
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _items(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    return value


def _bounds(value, label):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{label} must contain start and end")
        start, end = value
    else:
        start, end = _get(value, "start"), _get(value, "end")
    start = _number(start, f"{label}.start")
    end = _number(end, f"{label}.end")
    if end < start:
        raise ValueError(f"{label}.end precedes start")
    return start, end


def _merge(spans):
    merged = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def generate_candidates(transcript, slide_info, duration, cfg):
    """Intersect explicit static intervals with silence, without touching speech.

    Transcript segment AND word bounds are protected, including configured speech
    margins. Slide changes split static intervals even when their settling time
    is zero. Malformed timestamps raise rather than inventing unsafe silence.
    Inputs may use mappings or attribute-based objects; transcript may also be
    a list of segments. No inputs are mutated.
    """
    duration = _number(duration, "duration")
    cuts = _get(cfg, "cuts", {})
    minimum = _number(_get(cuts, "min_silence_s", 2), "min_silence_s")
    margin = _number(_get(cuts, "speech_margin_s", 0.2), "speech_margin_s")
    settle = _number(
        _get(cuts, "settle_after_slide_change_s", 1),
        "settle_after_slide_change_s",
    )
    if min(duration, minimum, margin, settle) < 0:
        raise ValueError("duration and cut settings must be nonnegative")

    protected = []
    speech = []

    def protect(value, label, text_key):
        start, end = _bounds(value, label)
        text = _get(value, text_key, "")
        speech.append({"start": start, "end": end, "text": str(text or "")})
        left, right = max(0.0, start - margin), min(duration, end + margin)
        if left == right and 0 <= left < duration:
            right = math.nextafter(right, math.inf)
        protected.append((left, right))

    if isinstance(transcript, (list, tuple)):
        segments, words = transcript, []
    elif transcript is None:
        raise ValueError("transcript must contain explicit segments or words")
    else:
        missing = object()
        segments = _get(transcript, "segments", missing)
        words = _get(transcript, "words", missing)
        if segments is missing and words is missing:
            if not isinstance(transcript, Mapping) or transcript:
                raise ValueError("transcript must contain segments or words")
            segments, words = [], []
        segments = [] if segments is missing else segments
        words = [] if words is missing else words
    for segment in _items(segments, "segments"):
        protect(segment, "segment", "text")
        for word in _items(_get(segment, "words", []), "segment.words"):
            protect(word, "word", "word")
    for word in _items(words, "words"):
        protect(word, "word", "word")

    static_intervals = _items(
        _get(slide_info, "static_intervals", []), "static_intervals"
    )
    static = []
    for interval in static_intervals:
        start, end = _bounds(interval, "static interval")
        static.append((max(0.0, start), min(duration, end)))

    changes = _get(slide_info, "change_times")
    if changes is None:
        changes = _get(slide_info, "slide_changes")
    if changes is None:
        changes = _get(slide_info, "change_points")
    if changes is None:
        changes = _get(slide_info, "changes", [])
    change_times = []
    for change in _items(changes, "slide changes"):
        if isinstance(change, Mapping) or hasattr(change, "time"):
            change = _get(change, "time", _get(change, "timestamp"))
        time = _number(change, "slide change")
        # Changes just before the media may still have a settling interval in it.
        end = max(time + settle, math.nextafter(time, math.inf))
        protected.append((max(0.0, time), min(duration, end)))
        if 0 <= time <= duration:
            change_times.append(time)
    change_times = sorted(set(change_times))
    protected = _merge(protected)
    speech.sort(key=lambda item: (item["start"], item["end"]))

    candidates = []
    for start, end in _merge(static):
        cursor = start
        gaps = []
        for left, right in protected:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                gaps.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            gaps.append((cursor, end))
        for left, right in gaps:
            if right <= left or right - left < minimum:
                continue
            before = [item for item in speech if item["end"] <= left]
            after = [item for item in speech if item["start"] >= right]
            context = {
                "transcript_before": (
                    max(before, key=lambda item: item["end"])["text"] if before else ""
                ),
                "transcript_after": after[0]["text"] if after else "",
                "previous_slide_change": next(
                    (time for time in reversed(change_times) if time <= left), None
                ),
                "next_slide_change": next(
                    (time for time in change_times if time >= right), None
                ),
            }
            candidates.append(
                Candidate(
                    id=f"candidate-{len(candidates) + 1:04d}",
                    start=left,
                    end=right,
                    duration=right - left,
                    context=context,
                )
            )
    return candidates
