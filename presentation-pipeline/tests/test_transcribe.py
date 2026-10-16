import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PipelineConfig
from app.transcribe import transcribe


@pytest.mark.parametrize("device,compute,expected", [
    ("cpu", "auto", [("cpu", "int8")]),
    ("cuda", "auto", [("cuda", "float16")]),
    ("cpu", "float32", [("cpu", "float32")]),
    ("auto", "auto", [("cuda", "float16")]),
])
def test_device_selection(tmp_path, monkeypatch, device, compute, expected):
    model = Mock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=1))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    config = PipelineConfig()
    config.transcription.device = device
    config.transcription.compute_type = compute
    transcribe("video.mp4", config, tmp_path)
    assert [(call.kwargs["device"], call.kwargs["compute_type"])
            for call in factory.call_args_list] == expected


@pytest.mark.parametrize("error", [RuntimeError, ValueError, OSError, NotImplementedError])
@pytest.mark.parametrize("lazy", [False, True])
def test_auto_falls_back_including_lazy_segment_errors(tmp_path, monkeypatch, error, lazy):
    def segments():
        raise error("Unsupported GPU")
        yield

    cpu = Mock()
    cpu.transcribe.return_value = ([], SimpleNamespace(language="en", duration=1))
    gpu = Mock()
    gpu.transcribe.return_value = (segments(), SimpleNamespace(language="en", duration=1))
    factory = Mock(side_effect=[gpu if lazy else error("Unsupported GPU"), cpu])
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    transcribe("video.mp4", PipelineConfig(), tmp_path)
    assert [call.kwargs["device"] for call in factory.call_args_list] == ["cuda", "cpu"]
    assert factory.call_args_list[-1].kwargs["compute_type"] == "int8"


def test_explicit_cuda_does_not_silently_fallback(tmp_path, monkeypatch):
    factory = Mock(side_effect=OSError("Unsupported GPU"))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    config = PipelineConfig()
    config.transcription.device = "cuda"
    with pytest.raises(OSError):
        transcribe("video.mp4", config, tmp_path)
    assert factory.call_count == 1
