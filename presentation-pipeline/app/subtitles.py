"""Pure-Python subtitle retiming and UTF-8 SRT output.

Times are seconds. Kept source intervals describe a union: they are sorted and
overlapping or touching intervals are merged. Removed gaps collapse to a single
edited-time boundary. Inputs and their metadata are never modified.
"""

from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import math
from pathlib import Path


def _seconds(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Times must be finite")
    return result


def _prepare_intervals(keep_intervals):
    intervals = []
    for start, end in keep_intervals:
        start, end = _seconds(start), _seconds(end)
        if end < start:
            raise ValueError("Interval end must not precede its start")
        if end > start:
            intervals.append((start, end))

    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _map_time(t, intervals):
    elapsed = 0.0
    for start, end in intervals:
        if t <= start:
            break
        elapsed += min(t, end) - start
        if t < end:
            break
    return elapsed


def map_time(t, keep_intervals):
    """Return the total kept duration before source time ``t``.

    Gaps collapse to cut boundaries; times before/after all keeps clamp to zero/
    total kept duration. Empty keeps map every finite time to zero.
    """
    return _map_time(_seconds(t), _prepare_intervals(keep_intervals))


def _remap_item(item, intervals):
    start, end = _seconds(item["start"]), _seconds(item["end"])
    if end < start:
        raise ValueError("Subtitle end must not precede its start")
    overlaps = [
        (max(start, left), min(end, right))
        for left, right in intervals
        if max(start, left) < min(end, right)
    ]
    if not overlaps:
        return None
    result = deepcopy(item)
    result["start"] = _map_time(overlaps[0][0], intervals)
    result["end"] = _map_time(overlaps[-1][1], intervals)
    return result


def remap_segments(segments, keep_intervals):
    """Clip, drop and retime segment dictionaries and optional word dictionaries.

    Each item requires ``start`` and ``end``. Only positive overlap survives;
    one segment spanning multiple cuts remains one segment. Words are clipped
    independently against the same keeps. Text and other metadata are preserved
    as deep copies, and segment/word ordering is retained.
    """
    intervals = _prepare_intervals(keep_intervals)
    result = []
    for segment in segments:
        mapped = _remap_item(segment, intervals)
        if mapped is None:
            continue
        if segment.get("words") is not None:
            mapped["words"] = [
                word
                for original in segment["words"]
                if (word := _remap_item(original, intervals)) is not None
            ]
        result.append(mapped)
    return result


def format_timestamp(seconds):
    """Format nonnegative seconds as HH:MM:SS,mmm, rounding halves upward."""
    value = _seconds(seconds)
    if value < 0:
        raise ValueError("SRT timestamps must be nonnegative")
    milliseconds = int(
        (Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def format_srt(segments, speaker_prefix=False):
    """Return sequentially numbered SRT cues, including a final blank line."""
    cues = []
    for number, segment in enumerate(segments, 1):
        start, end = _seconds(segment["start"]), _seconds(segment["end"])
        if end < start:
            raise ValueError("Subtitle end must not precede its start")
        prefix = (
            f"[{segment['speaker']}] "
            if speaker_prefix and segment.get("speaker") else ""
        )
        cues.append(
            f"{number}\n"
            f"{format_timestamp(segment['start'])} --> "
            f"{format_timestamp(segment['end'])}\n"
            f"{prefix}{segment['text'].lstrip() if prefix else segment['text']}\n\n"
        )
    return "".join(cues)


def write_srt(segments, path, speaker_prefix=False):
    """Write SRT to a string or path-like destination as UTF-8 with LF newlines."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as output:
        output.write(format_srt(segments, speaker_prefix=speaker_prefix))
