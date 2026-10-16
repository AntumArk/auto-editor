import sys
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DiarizationConfig
from app.diarize import assign_speakers, diarize


def test_total_overlap_and_independent_word_assignment():
    segments = [{
        "start": 0, "end": 6, "text": "one two",
        "words": [
            {"start": 0, "end": 2, "word": "one"},
            {"start": 2, "end": 5, "word": "two"},
        ],
    }]
    turns = [
        {"start": 0, "end": 2, "speaker": "A"},
        {"start": 2, "end": 5, "speaker": "B"},
        {"start": 5, "end": 7, "speaker": "A"},
    ]
    original = deepcopy(segments)
    result = assign_speakers(segments, turns)
    assert result[0]["speaker"] == "A"
    assert [word["speaker"] for word in result[0]["words"]] == ["A", "B"]
    assert segments == original


def test_overlapping_speech_ties_use_first_turn():
    result = assign_speakers([{"start": 0, "end": 2}], [
        {"start": 0, "end": 2, "speaker": "A"},
        {"start": 0, "end": 2, "speaker": "B"},
    ])
    assert result[0]["speaker"] == "A"


def test_no_overlap_and_optional_nearest_fill():
    segments = [{
        "start": 4, "end": 5, "words": [{"start": 4, "end": 4, "word": "zero"}],
    }]
    turns = [
        {"start": 0, "end": 1, "speaker": "A"},
        {"start": 6, "end": 7, "speaker": "B"},
    ]
    assert assign_speakers(segments, turns) == segments
    result = assign_speakers(segments, turns, fill_nearest=True)
    assert result[0]["speaker"] == "B"
    assert result[0]["words"][0]["speaker"] == "B"
    assert assign_speakers(segments, [], fill_nearest=True) == segments
    assert assign_speakers([], turns) == []
    assert "speaker" not in assign_speakers([{"start": 1, "end": 2}], turns)[0]


def test_diarization_lazy_backend_and_environment_token(monkeypatch):
    monkeypatch.setenv("TEST_HF_TOKEN", "test-only-placeholder")
    annotation = Mock()
    annotation.itertracks.return_value = [
        (SimpleNamespace(start=0, end=1), None, "SPEAKER_00"),
    ]
    pipeline = Mock(return_value=annotation)
    factory = Mock(return_value=pipeline)
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(
        Pipeline=SimpleNamespace(from_pretrained=factory),
    ))
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.task", SimpleNamespace(
        Problem=int, Resolution=str, Specifications=dict,
    ))
    monkeypatch.setitem(sys.modules, "torch.torch_version", SimpleNamespace(
        TorchVersion=str,
    ))
    cache = Mock()
    safe_globals = Mock(return_value=nullcontext())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        device=lambda value: value, cuda=SimpleNamespace(empty_cache=cache),
        serialization=SimpleNamespace(safe_globals=safe_globals),
    ))
    result = diarize("video.mp4", [{"start": 0, "end": 1}], DiarizationConfig(
        hf_token="TEST_HF_TOKEN", device="cuda", min_speakers=1, max_speakers=3,
    ))
    factory.assert_called_once_with(
        "pyannote/speaker-diarization-3.1", use_auth_token="test-only-placeholder",
    )
    pipeline.to.assert_called_once_with("cuda")
    pipeline.assert_called_once_with("video.mp4", min_speakers=1, max_speakers=3)
    assert result[0]["speaker"] == "SPEAKER_00"
    cache.assert_called_once()
    safe_globals.assert_called_once_with([str, dict, int, str])


def test_missing_token_is_actionable(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="environment variable"):
        diarize("video.mp4", [], DiarizationConfig())


def test_missing_optional_backend_is_actionable(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-only-placeholder")
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    with pytest.raises(RuntimeError, match="requirements-extra.txt"):
        diarize("video.mp4", [], DiarizationConfig())
