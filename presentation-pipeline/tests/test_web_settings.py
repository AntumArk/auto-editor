from dataclasses import fields
from pathlib import Path
import sys

import pytest
from werkzeug.datastructures import MultiDict
from werkzeug.exceptions import BadRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PipelineConfig, load_config, restore_job_settings
from app.web_settings import GROUPS, SETTINGS, parse_settings, settings_key, snapshot


def form(config):
    result = {"settings_form": "1"}
    for setting in SETTINGS:
        value = getattr(getattr(config, setting.section), setting.key)
        if setting.kind == "checkbox":
            if value:
                result[setting.name] = "on"
        else:
            result[setting.name] = "" if value is None else str(value)
    return MultiDict(result)


def test_all_per_job_settings_are_exposed_and_round_trip():
    config = PipelineConfig()
    config.transcription.language = None
    values = form(config)
    restored = parse_settings(config, values)
    assert restored == config
    assert settings_key(restored) == settings_key(config)
    for section, _ in GROUPS:
        exposed = {item.key for item in SETTINGS if item.section == section}
        expected = {field.name for field in fields(getattr(config, section))}
        if section == "llm":
            expected -= {"provider", "url"}
        assert exposed == expected


def test_editable_values_restore_without_exposing_or_replacing_service_config():
    config = PipelineConfig()
    saved = snapshot(config)
    assert "paths" not in saved
    assert "url" not in saved["llm"]
    assert "provider" not in saved["llm"]
    config.llm.url = "http://new-service:1234"
    loaded = restore_job_settings(config, saved)
    assert loaded.llm.url == config.llm.url
    assert snapshot(loaded) == saved


def test_settings_identity_changes_for_every_setting_but_not_paths():
    config = PipelineConfig()
    baseline = settings_key(config)
    assert baseline == settings_key(restore_job_settings(config, snapshot(config)))
    config.paths.output = "/another/output"
    assert settings_key(config) == baseline
    for setting in SETTINGS:
        modified = PipelineConfig()
        value = getattr(getattr(modified, setting.section), setting.key)
        if isinstance(value, bool):
            value = not value
        elif isinstance(value, (float, int)):
            value += 1
        else:
            value = str(value) + "-changed"
        setattr(getattr(modified, setting.section), setting.key, value)
        assert settings_key(modified) != baseline, setting.name


@pytest.mark.parametrize("key,value", [
    ("visual.sample_fps", "nan"), ("visual.sample_fps", "0"),
    ("visual.change_threshold", "inf"), ("visual.change_threshold", "1.01"),
    ("visual.downscale_width", "3.5"), ("cuts.target_pause_s", "-1"),
    ("cuts.speech_margin_s", "-0.1"), ("cuts.full_remove_min_s", "bad"),
    ("report.min_chapter_gap_s", "9"), ("llm.timeout_s", "0"),
    ("llm.model", "hello;echo"), ("llm.model", ""),
    ("report.language", ""), ("report.language", "../lt"),
    ("language", "english"), ("model", "../../model"),
    ("transcription.device", "metal"), ("transcription.compute_type", "bad"),
    ("transcription.task", "summarize"), ("transcription.vad", "yes"),
])
def test_invalid_settings_rejected(key, value):
    config = PipelineConfig()
    values = form(config)
    values[key] = value
    with pytest.raises(BadRequest):
        parse_settings(config, values)


def test_cpu_half_precision_rejected():
    config = PipelineConfig()
    values = form(config)
    values["transcription.device"] = "cpu"
    values["transcription.compute_type"] = "float16"
    with pytest.raises(BadRequest, match="requires CUDA"):
        parse_settings(config, values)


def test_full_form_unchecked_boxes_are_false_and_config_unchanged():
    config = PipelineConfig()
    values = form(config)
    for setting in SETTINGS:
        if setting.kind == "checkbox":
            values.pop(setting.name, None)
    result = parse_settings(config, values)
    assert config.transcription.vad is True
    assert result.transcription.vad is False
    assert result.transcription.word_timestamps is False
    assert result.llm.enabled is False
    assert result.export.preview is False


def test_preview_yaml_and_environment_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("export:\n  preview: true\n")
    assert load_config(path, environ={}).export.preview is True
    assert load_config(path, environ={"PIPELINE__EXPORT__PREVIEW": "false"}).export.preview is False
