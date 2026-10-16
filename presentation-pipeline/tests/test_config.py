import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config


def test_loads_default_config():
    config = load_config()

    assert config.paths.inbox == "/data/inbox"
    assert config.transcription.model == "large-v3"
    assert config.cuts.min_silence_s == 2.0
    assert config.watcher.video_extensions[0] == ".mp4"


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
