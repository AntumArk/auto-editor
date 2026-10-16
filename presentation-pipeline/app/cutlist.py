"""Pure source-time cut-list math and subprocess-only auto-editor export."""

import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _nonnegative(value, name):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return number


@dataclass
class CutList:
    removed: list[tuple[float, float]]
    keep: list[tuple[float, float]]
    original_duration: float
    edited_duration: float

    @classmethod
    def from_dict(cls, value):
        return cls(
            removed=[tuple(interval) for interval in value["removed"]],
            keep=[tuple(interval) for interval in value["keep"]],
            original_duration=value["original_duration"],
            edited_duration=value["edited_duration"],
        )

    def to_dict(self):
        return asdict(self)

    def to_json(self, path=None):
        """Return JSON and, optionally, write it to a UTF-8 file."""
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


def build_cutlist(candidates, decisions, duration, cfg):
    """Shorten centrally, retaining half the target pause at each edge.

    An explicit ``approve`` permits full removal of a sufficiently long pause.
    A positive target override instead retains that much pause. Invalid decision
    values fall back to deterministic shortening; decisions cannot set bounds.
    """
    duration = _nonnegative(duration, "duration")
    cuts = _get(cfg, "cuts", {})
    default_target = _nonnegative(_get(cuts, "target_pause_s", 0.6), "target_pause_s")
    full_min = _nonnegative(_get(cuts, "full_remove_min_s", 8.0), "full_remove_min_s")
    if not isinstance(decisions, Mapping):
        decisions = {
            _get(decision, "candidate_id", _get(decision, "id")): decision
            for decision in (decisions or [])
        }
    removed = []
    for candidate in candidates:
        start = _nonnegative(_get(candidate, "start"), "candidate start")
        end = _nonnegative(_get(candidate, "end"), "candidate end")
        if not start < end <= duration:
            raise ValueError("Candidate must lie within the source duration")
        length = end - start
        decision = decisions.get(_get(candidate, "id"), {})
        action = _get(decision, "action", "shorten")
        if action == "keep":
            continue
        if action not in ("approve", "shorten"):
            action = "shorten"
        override = _get(decision, "target_pause_s")
        target = default_target
        if override is not None:
            try:
                target = _nonnegative(override, "target_pause_s")
            except (TypeError, ValueError):
                action = "shorten"
        full_removal = action == "approve" and length >= full_min
        if full_removal and (override is None or target == 0):
            target = 0.0
        elif target == 0:
            # Zero retention must never bypass the explicit full-removal gate.
            continue
        target = min(target, length)
        if target < length:
            removed.append((start + target / 2, end - target / 2))

    merged = []
    for start, end in sorted(removed):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    keep = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            keep.append((cursor, start))
        cursor = end
    if cursor < duration:
        keep.append((cursor, duration))
    return CutList(merged, keep, duration, math.fsum(end - start for start, end in keep))


def _run_auto_editor(command):
    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        raise RuntimeError(f"auto-editor export failed (exit {exc.returncode}): {detail}") from exc


def _edit_command(video, cutlist):
    command = ["auto-editor", str(Path(video).resolve()), "--edit", "none"]
    for start, end in cutlist.removed:
        # Bare numbers mean frames in this CLI, so always include seconds units.
        command.extend(["--cut-out", f"{start:.9f}sec,{end:.9f}sec"])
    return command


def render_preview(video, cutlist, out_dir, name=None):
    """Render a browser-compatible preview using the same explicit cut ranges."""
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(name).name if name is not None else Path(video).stem
    preview = output / f"{stem}.preview.mp4"
    try:
        _run_auto_editor([
            *_edit_command(video, cutlist), "--export", "default",
            "--video-codec", "libx264", "--audio-codec", "aac", "--no-open",
            "--output", str(preview),
        ])
    except (RuntimeError, OSError):
        preview.unlink(missing_ok=True)
        raise
    return preview


def export_with_auto_editor(video, cutlist, out_dir, cfg, name=None):
    """Write audit JSON and Resolve FCPXML; optionally render a preview MP4.

    Set ``cfg.export.preview`` to enable the preview. Return artifact paths.
    ``--edit none`` keeps the source except for our explicit cut ranges.
    """
    video = Path(video).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).name if name is not None else video.stem
    audit = out_dir / f"{stem}.cutlist.json"
    timeline = out_dir / f"{stem}.fcpxml"
    cutlist.to_json(audit)
    command = _edit_command(video, cutlist)
    _run_auto_editor(
        [*command, "--export", "resolve", "--output", str(timeline)],
    )
    artifacts = {"cutlist": audit, "resolve": timeline}
    if _get(_get(cfg, "export", {}), "preview", False):
        artifacts["preview"] = render_preview(video, cutlist, out_dir, name=stem)
    return artifacts
