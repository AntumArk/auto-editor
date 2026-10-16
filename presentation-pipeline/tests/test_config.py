import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config


def test_loads_default_config():
    config = load_config()

    assert config.paths.inbox == "/data/inbox"
    assert config.transcription.model == "large-v3"
    assert config.transcription.device == "auto"
    assert config.transcription.compute_type == "auto"
    assert config.transcription.beam_size == 5
    assert config.transcription.condition_on_previous_text is False
    assert config.transcription.initial_prompt is None
    assert config.transcription.vad is True
    assert config.transcription.vad_filter is None
    assert config.transcription.vad_parameters is None
    assert config.transcription.align is False
    assert config.transcription.align_model is None
    assert config.diarization.enabled is True
    assert config.diarization.hf_token == "HF_TOKEN"
    assert config.diarization.device == "cpu"
    assert config.diarization.speaker_prefix is False
    assert config.diarization.fill_nearest is False
    assert config.cuts.min_silence_s == 2.0
    assert config.watcher.video_extensions[0] == ".mp4"


def test_transcription_overrides():
    config = load_config(environ={
        "PIPELINE__TRANSCRIPTION__DEVICE": "cpu",
        "PIPELINE__TRANSCRIPTION__COMPUTE_TYPE": "float32",
        "PIPELINE__TRANSCRIPTION__LANGUAGE": "null",
    })
    assert config.transcription.device == "cpu"
    assert config.transcription.compute_type == "float32"
    assert config.transcription.language is None


def test_invalid_device():
    with pytest.raises(ValueError, match="transcription.device"):
        load_config(environ={"PIPELINE__TRANSCRIPTION__DEVICE": "invalid"})


def test_quality_and_diarization_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "transcription:\n  beam_size: 7\n  initial_prompt: Vilnius\n"
        "  vad_parameters:\n    min_silence_duration_ms: 600\n"
        "  align: true\n  align_model: test-aligner\n"
        "diarization:\n  enabled: true\n  hf_token: CUSTOM_HF_TOKEN\n"
        "  min_speakers: 2\n  max_speakers: 4\n  device: cuda\n"
    )
    config = load_config(path, environ={
        "PIPELINE__DIARIZATION__SPEAKER_PREFIX": "true",
        "PIPELINE__DIARIZATION__FILL_NEAREST": "true",
        "PIPELINE__TRANSCRIPTION__VAD_FILTER": "false",
        "PIPELINE__TRANSCRIPTION__CONDITION_ON_PREVIOUS_TEXT": "true",
    })
    assert config.transcription.beam_size == 7
    assert config.transcription.initial_prompt == "Vilnius"
    assert config.transcription.vad_parameters == {"min_silence_duration_ms": 600}
    assert config.transcription.align is True
    assert config.transcription.align_model == "test-aligner"
    assert config.transcription.vad_filter is False
    assert config.transcription.condition_on_previous_text is True
    assert config.diarization.enabled is True
    assert config.diarization.hf_token == "CUSTOM_HF_TOKEN"
    assert config.diarization.min_speakers == 2
    assert config.diarization.max_speakers == 4
    assert config.diarization.device == "cuda"
    assert config.diarization.speaker_prefix is True
    assert config.diarization.fill_nearest is True


@pytest.mark.parametrize("settings", [
    {"DEVICE": "auto"}, {"MIN_SPEAKERS": "0"}, {"MAX_SPEAKERS": "-1"},
    {"MIN_SPEAKERS": "4", "MAX_SPEAKERS": "2"},
])
def test_invalid_diarization_settings(settings):
    with pytest.raises(ValueError, match="diarization"):
        load_config(environ={
            f"PIPELINE__DIARIZATION__{key}": value for key, value in settings.items()
        })

def test_environment_override_parses_value_types(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cuts:\n  min_silence_s: 2.0\nllm:\n  enabled: true\n")

    config = load_config(
        config_path,
        {
            "PIPELINE__CUTS__MIN_SILENCE_S": "3.5",
            "PIPELINE__LLM__ENABLED": "false",
        },
    )

    assert config.cuts.min_silence_s == 3.5
    assert config.llm.enabled is False
