"""Editable per-job settings shared by the form, validation, and cache identity."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

from werkzeug.exceptions import BadRequest

from app.config import PipelineConfig, restore_job_settings


MODELS = ("tiny", "base", "small", "medium", "large-v3")


@dataclass(frozen=True)
class Setting:
    section: str
    key: str
    label: str
    kind: str = "number"
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    @property
    def name(self):
        return {"transcription.model": "model", "transcription.language": "language",
                "llm.enabled": "llm"}.get(f"{self.section}.{self.key}",
                                        f"{self.section}.{self.key}")


SETTINGS = (
    Setting("transcription", "model", "Whisper model", "select", choices=MODELS),
    Setting("transcription", "language", "Spoken language", "language",
            help="Leave blank to detect automatically; otherwise use a language code such as en or lt."),
    Setting("transcription", "device", "Device", "select", choices=("cpu", "auto", "cuda"),
            help="CPU is recommended for legacy 2 GB GPUs. CUDA requires a GPU-enabled container."),
    Setting("transcription", "compute_type", "Compute precision", "select",
            choices=("auto", "int8", "float32", "float16", "int8_float16", "int8_float32"),
            help="Auto chooses int8 on CPU and float16 on CUDA."),
    Setting("transcription", "task", "Task", "select", choices=("transcribe", "translate"),
            help="Translate produces English subtitles, not arbitrary target languages."),
    Setting("transcription", "vad", "Filter non-speech (VAD)", "checkbox"),
    Setting("transcription", "word_timestamps", "Word-level timestamps", "checkbox"),
    Setting("visual", "sample_fps", "Samples per second", minimum=0.1, maximum=30),
    Setting("visual", "change_threshold", "Slide change threshold", minimum=0, maximum=1),
    Setting("visual", "downscale_width", "Analysis width (pixels)", "integer", 16, 3840),
    Setting("cuts", "min_silence_s", "Minimum silence (seconds)", minimum=0.1, maximum=3600),
    Setting("cuts", "target_pause_s", "Retained pause (seconds)", minimum=0, maximum=3600,
            help="Zero does not bypass the explicit approval required for full removal."),
    Setting("cuts", "speech_margin_s", "Speech padding (seconds)", minimum=0, maximum=60),
    Setting("cuts", "settle_after_slide_change_s", "Slide settling time (seconds)",
            minimum=0, maximum=60),
    Setting("cuts", "full_remove_min_s", "Minimum pause for approved full removal (seconds)",
            minimum=0, maximum=3600),
    Setting("llm", "enabled", "Review cuts with Ollama", "checkbox",
            help="Requires the configured Ollama service. Unavailable LLMs use conservative defaults."),
    Setting("llm", "model", "Ollama model", "text"),
    Setting("llm", "timeout_s", "Request timeout (seconds)", "integer", 1, 3600),
    Setting("report", "language", "Summary language", "language"),
    Setting("report", "min_chapter_gap_s", "Minimum chapter gap (seconds)", minimum=10, maximum=3600),
    Setting("export", "preview", "Render edited MP4 preview", "checkbox",
            help="Adds encoding time. You can also render a preview later without retranscribing."),
)

GROUPS = (
    ("transcription", "Transcription"),
    ("visual", "Slide detection"),
    ("cuts", "Pause and cut rules"),
    ("llm", "LLM review"),
    ("report", "Report and chapters"),
    ("export", "Export"),
)


def snapshot(config):
    values = asdict(config)
    return {section: values[section] for section, _ in GROUPS
            if section != "llm"} | {
        "llm": {key: values["llm"][key] for key in ("enabled", "model", "timeout_s")}
    }


def settings_key(config):
    # Service configuration also affects results, but is never exposed in the UI.
    values = snapshot(config) | {"llm": asdict(config.llm)}
    for setting in SETTINGS:
        if setting.kind in ("number", "integer"):
            number = values[setting.section][setting.key]
            values[setting.section][setting.key] = (
                int(number) if setting.kind == "integer" else float(number)
            )
    payload = json.dumps(values, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def parse_settings(config: PipelineConfig, form):
    result = deepcopy(config)
    full_form = form.get("settings_form") == "1"
    for setting in SETTINGS:
        group = getattr(result, setting.section)
        current = getattr(group, setting.key)
        if setting.kind == "checkbox":
            if full_form or setting.name in form or setting.name == "llm":
                raw = form.get(setting.name, "off")
                if raw not in ("on", "off"):
                    raise BadRequest(f"{setting.label}: invalid checkbox value")
                setattr(group, setting.key, raw == "on")
            continue
        if setting.name not in form:
            continue
        raw = form[setting.name].strip()
        if setting.kind == "select":
            if raw not in (*setting.choices, current):
                raise BadRequest(f"{setting.label}: unsupported option")
            value = raw
        elif setting.kind == "language":
            if not re.fullmatch(r"[a-z]{2,3}", raw) and (
                raw or setting.section != "transcription"
            ):
                raise BadRequest(f"{setting.label}: use a two- or three-letter language code")
            value = raw or None
        elif setting.kind == "text":
            if not raw or len(raw) > 200 or not re.fullmatch(r"[A-Za-z0-9._:/-]+", raw):
                raise BadRequest(f"{setting.label}: invalid model name")
            value = raw
        else:
            try:
                value = int(raw) if setting.kind == "integer" else float(raw)
            except ValueError as exc:
                raise BadRequest(f"{setting.label}: enter a number") from exc
            if not math.isfinite(value) or (
                setting.minimum is not None and value < setting.minimum
            ) or (setting.maximum is not None and value > setting.maximum):
                raise BadRequest(
                    f"{setting.label}: must be between {setting.minimum} and {setting.maximum}"
                )
        setattr(group, setting.key, value)
    device, precision = result.transcription.device, result.transcription.compute_type
    if device == "cpu" and precision in ("float16", "int8_float16"):
        raise BadRequest("Float16 precision requires CUDA; choose auto, int8, or float32 on CPU.")
    return result
