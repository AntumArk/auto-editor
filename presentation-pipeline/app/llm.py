"""Local LLM review of deterministic cut candidates (Ollama).

The LLM is a reviewer only: it receives candidate intervals produced by the
deterministic candidate generator plus surrounding context, and returns a
decision per candidate id. It must never produce or alter timestamps; any
timestamp-like fields in its output are ignored.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

VALID_ACTIONS = ("approve", "shorten", "keep")
MIN_PAUSE_S = 0.2
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_URL = "http://ollama:11434"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_CONTEXT_S = 15.0
DEFAULT_BATCH_SIZE = 10
DEFAULT_TARGET_PAUSE_S = 0.6

REASON_DEFAULT = "deterministic_default"
REASON_DISABLED = "llm_disabled"
REASON_UNAVAILABLE = "llm_unavailable"
REASON_INVALID_JSON = "llm_invalid_json"
REASON_INVALID_ACTION = "llm_invalid_action"

SYSTEM_PROMPT = (
    "You review pause-shortening candidates in a slide-presentation screen "
    "recording. Each candidate is a silent interval where the slide does not "
    "change. You must NOT output timestamps. For every candidate id, decide:\n"
    '- "approve": remove the pause entirely\n'
    '- "shorten": shorten the pause to target_pause_s seconds\n'
    '- "keep": leave the pause unchanged (e.g. deliberate dramatic pause)\n'
    "Respond with strict JSON only, in the form:\n"
    '{"decisions": [{"id": <candidate id>, "action": "approve|shorten|keep", '
    '"target_pause_s": <number or null>, "reason": "<short reason>", '
    '"needs_review": <true|false>}]}'
)


@dataclass
class Decision:
    candidate_id: Any
    action: str
    target_pause_s: float | None
    reason: str
    needs_review: bool = False


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(obj: Any, keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value
    return default


def _cfg_value(cfg: Any, section: str, key: str, default: Any) -> Any:
    sec = _get(cfg, section) if cfg is not None else None
    if sec is None:
        return default
    value = _get(sec, key)
    return default if value is None else value


def _candidate_id(candidate: Any) -> Any:
    return _first(candidate, ("id", "candidate_id"))


def _candidate_bounds(candidate: Any) -> tuple[float, float]:
    start = float(_first(candidate, ("start", "start_s"), 0.0))
    end = float(_first(candidate, ("end", "end_s"), start))
    return start, end


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    return num if math.isfinite(num) else None


def _clamp_pause(value: float, duration: float) -> float:
    return min(max(value, MIN_PAUSE_S), max(duration, 0.0))


def _default_target(cfg: Any) -> float:
    return float(_cfg_value(cfg, "cuts", "target_pause_s", DEFAULT_TARGET_PAUSE_S))


def _default_decision(candidate: Any, cfg: Any, reason: str) -> Decision:
    start, end = _candidate_bounds(candidate)
    return Decision(
        candidate_id=_candidate_id(candidate),
        action="shorten",
        target_pause_s=_clamp_pause(_default_target(cfg), end - start),
        reason=reason,
        needs_review=False,
    )


def default_decisions(
    candidates: Sequence[Any], cfg: Any, reason: str = REASON_DEFAULT
) -> list[Decision]:
    return [_default_decision(c, cfg, reason) for c in candidates]


def validate_llm_output(
    raw: Any, candidates: Sequence[Any], cfg: Any
) -> list[Decision]:
    """Validate raw LLM output into one Decision per candidate (pure function).

    Output order matches ``candidates``. Unknown ids and any extra fields (such
    as invented timestamps) are ignored. Invalid entries and missing
    candidates fall back to the deterministic default.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except ValueError:
            return default_decisions(candidates, cfg, REASON_INVALID_JSON)
    else:
        data = raw

    items = data.get("decisions") if isinstance(data, Mapping) else data
    if not isinstance(items, list):
        return default_decisions(candidates, cfg, REASON_INVALID_JSON)

    by_id: dict[str, Mapping] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        cid = _first(item, ("id", "candidate_id"))
        if cid is None or isinstance(cid, (Mapping, list)):
            continue
        by_id.setdefault(str(cid), item)

    decisions: list[Decision] = []
    for candidate in candidates:
        item = by_id.get(str(_candidate_id(candidate)))
        if item is None:
            decisions.append(_default_decision(candidate, cfg, REASON_DEFAULT))
            continue

        action = item.get("action")
        if not isinstance(action, str) or action.strip().lower() not in VALID_ACTIONS:
            decisions.append(_default_decision(candidate, cfg, REASON_INVALID_ACTION))
            continue
        action = action.strip().lower()

        target: float | None = None
        if action == "shorten":
            start, end = _candidate_bounds(candidate)
            value = _as_number(item.get("target_pause_s"))
            if value is None:
                value = _default_target(cfg)
            target = _clamp_pause(value, end - start)

        reason = item.get("reason")
        decisions.append(
            Decision(
                candidate_id=_candidate_id(candidate),
                action=action,
                target_pause_s=target,
                reason=reason.strip() if isinstance(reason, str) else "",
                needs_review=item.get("needs_review") is True,
            )
        )
    return decisions


def _transcript_segments(transcript: Any) -> list[Any]:
    if transcript is None:
        return []
    if isinstance(transcript, list):
        return transcript
    segments = _get(transcript, "segments")
    return list(segments) if segments else []


def _context_text(transcript: Any, start: float, end: float, window: float) -> str:
    lo, hi = start - window, end + window
    parts = []
    for seg in _transcript_segments(transcript):
        s = _as_number(_get(seg, "start"))
        e = _as_number(_get(seg, "end"))
        text = _get(seg, "text")
        if s is None or e is None or not isinstance(text, str):
            continue
        if e >= lo and s <= hi:
            where = "before" if e <= start else "after" if s >= end else "during"
            parts.append(f"[{where}] {text.strip()}")
    return "\n".join(parts)


def _slide_change_times(slide_info: Any) -> list[float]:
    if slide_info is None:
        return []
    if not isinstance(slide_info, list):
        changes = _first(slide_info, ("changes", "slide_changes"))
        slide_info = list(changes) if changes else []
    times = []
    for item in slide_info:
        t = _as_number(item)
        if t is None:
            t = _as_number(_first(item, ("time", "t", "start")))
        if t is not None:
            times.append(t)
    return sorted(times)


def _build_payload(
    candidates: Sequence[Any], transcript: Any, slide_info: Any, cfg: Any
) -> list[dict]:
    window = float(_cfg_value(cfg, "llm", "context_s", DEFAULT_CONTEXT_S))
    changes = _slide_change_times(slide_info)
    payload = []
    for candidate in candidates:
        start, end = _candidate_bounds(candidate)
        payload.append(
            {
                "id": _candidate_id(candidate),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_s": round(end - start, 3),
                "transcript_context": _context_text(transcript, start, end, window),
                "nearby_slide_changes": [
                    round(t, 3) for t in changes if start - window <= t <= end + window
                ],
            }
        )
    return payload


def _call_ollama(messages: list[dict], cfg: Any) -> str:
    url = str(_cfg_value(cfg, "llm", "url", DEFAULT_URL)).rstrip("/")
    body = {
        "model": _cfg_value(cfg, "llm", "model", DEFAULT_MODEL),
        "messages": messages,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        f"{url}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(_cfg_value(cfg, "llm", "timeout_s", DEFAULT_TIMEOUT_S))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read().decode("utf-8"))
    content = reply["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("unexpected Ollama response")
    return content


def review_candidates(
    candidates: Sequence[Any], transcript: Any, slide_info: Any, cfg: Any
) -> list[Decision]:
    """Ask the local LLM to review candidates; never trusts it for timestamps."""
    candidates = list(candidates)
    if not candidates:
        return []
    if not _cfg_value(cfg, "llm", "enabled", False):
        return default_decisions(candidates, cfg, REASON_DISABLED)

    batch_size = max(1, int(_cfg_value(cfg, "llm", "batch_size", DEFAULT_BATCH_SIZE)))
    decisions: list[Decision] = []
    try:
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            payload = _build_payload(batch, transcript, slide_info, cfg)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"candidates": payload}, ensure_ascii=False),
                },
            ]
            raw = _call_ollama(messages, cfg)
            decisions.extend(validate_llm_output(raw, batch, cfg))
    except Exception:
        return default_decisions(candidates, cfg, REASON_UNAVAILABLE)
    return decisions
