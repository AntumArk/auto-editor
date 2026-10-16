from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass
class PathsConfig:
    inbox: str = "/data/inbox"
    output: str = "/data/output"
    workdir: str = "/data/work"


@dataclass
class TranscriptionConfig:
    model: str = "large-v3"
    language: str = "lt"
    task: str = "transcribe"
    vad: bool = True
    word_timestamps: bool = True


@dataclass
class VisualConfig:
    sample_fps: float = 3
    change_threshold: float = 0.04
    downscale_width: int = 320


@dataclass
class CutsConfig:
    min_silence_s: float = 2.0
    target_pause_s: float = 0.6
    speech_margin_s: float = 0.25
    settle_after_slide_change_s: float = 1.0
    full_remove_min_s: float = 8.0


@dataclass
class LLMConfig:
    enabled: bool = True
    provider: str = "ollama"
    url: str = "http://ollama:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout_s: int = 120


@dataclass
class ReportConfig:
    language: str = "lt"
    min_chapter_gap_s: float = 10


@dataclass
class WatcherConfig:
    poll_interval_s: float = 5
    stable_checks: int = 2
    video_extensions: tuple[str, ...] = (
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".m4v",
    )


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    cuts: CutsConfig = field(default_factory=CutsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)


def _dataclass_from_dict(cls: type, values: dict[str, Any]):
    hints = get_type_hints(cls)
    result = {}
    for field in fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        field_type = hints[field.name]
        if isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__"):
            value = _dataclass_from_dict(field_type, value)
        elif get_origin(field_type) is tuple and isinstance(value, list):
            value = tuple(value)
        result[field.name] = value
    return cls(**result)


def load_config(
    path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> PipelineConfig:
    """Load YAML configuration and apply PIPELINE__SECTION__KEY environment overrides."""
    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    for name, raw_value in (os.environ if environ is None else environ).items():
        parts = name.split("__")
        if len(parts) < 3 or parts[0] != "PIPELINE":
            continue
        section = values
        for part in parts[1:-1]:
            section = section.setdefault(part.lower(), {})
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid value for environment override {name}") from exc
        section[parts[-1].lower()] = value

    return _dataclass_from_dict(PipelineConfig, values)
