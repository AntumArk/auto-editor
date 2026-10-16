import sys
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PipelineConfig
from app.transcribe import transcribe


@pytest.mark.parametrize("device,compute,expected", [
    ("cpu", "auto", [("cpu", "int8")]),
    ("cuda", "auto", [("cuda", "int8_float16")]),
    ("cpu", "float32", [("cpu", "float32")]),
    ("auto", "auto", [("cuda", "int8_float16")]),
    ("cuda", "float32", [("cuda", "float32")]),
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


def test_fallback_exhaustion_raises_cpu_error(tmp_path, monkeypatch):
    factory = Mock(side_effect=OSError("Unsupported GPU"))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    config = PipelineConfig()
    config.transcription.device = "cuda"
    with pytest.raises(OSError):
        transcribe("video.mp4", config, tmp_path)
    assert factory.call_count == 2
    assert factory.call_args_list[-1].kwargs == {"device": "cpu", "compute_type": "int8"}


@pytest.mark.parametrize("device", ["auto", "cuda"])
@pytest.mark.parametrize("lazy", [False, True])
def test_configured_cuda_oom_retries_quantized_then_cpu(tmp_path, monkeypatch, device, lazy):
    def segments():
        raise RuntimeError("CUDA out of memory")
        yield

    gpu = Mock()
    gpu.transcribe.side_effect = lambda *a, **kw: (
        segments(), SimpleNamespace(language="lt", duration=1)
    )
    cpu = Mock()
    cpu.transcribe.return_value = ([], SimpleNamespace(language="lt", duration=1))
    failure = gpu if lazy else RuntimeError("CUDA out of memory")
    factory = Mock(side_effect=[failure, failure, cpu])
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    config = PipelineConfig()
    config.transcription.device = device
    config.transcription.compute_type = "float16"
    transcribe("video.mp4", config, tmp_path)
    assert [call.kwargs for call in factory.call_args_list] == [
        {"device": "cuda", "compute_type": "float16"},
        {"device": "cuda", "compute_type": "int8_float16"},
        {"device": "cpu", "compute_type": "int8"},
    ]


def test_quantized_cuda_retry_can_succeed(tmp_path, monkeypatch):
    model = Mock()
    model.transcribe.return_value = ([], SimpleNamespace(language="lt", duration=1))
    factory = Mock(side_effect=[RuntimeError("CUDA out of memory"), model])
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    config = PipelineConfig()
    config.transcription.compute_type = "float16"
    transcribe("video.mp4", config, tmp_path)
    assert [call.kwargs["compute_type"] for call in factory.call_args_list] == [
        "float16", "int8_float16",
    ]


def test_transcription_options(tmp_path, monkeypatch):
    model = Mock()
    model.transcribe.return_value = ([], SimpleNamespace(language="lt", duration=1))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=Mock(return_value=model),
    ))
    config = PipelineConfig()
    transcribe("video.mp4", config, tmp_path)
    assert model.transcribe.call_args.kwargs == {
        "language": "lt", "task": "transcribe", "vad_filter": True,
        "word_timestamps": True, "beam_size": 5,
        "condition_on_previous_text": False, "initial_prompt": None,
    }
    config.transcription.beam_size = 7
    config.transcription.condition_on_previous_text = True
    config.transcription.initial_prompt = "Vilnius, CTranslate2"
    config.transcription.vad_filter = False
    config.transcription.vad_parameters = {"min_silence_duration_ms": 600}
    transcribe("video.mp4", config, tmp_path)
    options = model.transcribe.call_args.kwargs
    assert options["beam_size"] == 7
    assert options["condition_on_previous_text"] is True
    assert options["initial_prompt"] == "Vilnius, CTranslate2"
    assert options["vad_filter"] is False
    assert options["vad_parameters"] == {"min_silence_duration_ms": 600}
    config.transcription.vad_filter = None
    config.transcription.vad = False
    transcribe("video.mp4", config, tmp_path)
    assert model.transcribe.call_args.kwargs["vad_filter"] is False


def test_release_whisper_before_alignment_and_diarization(tmp_path, monkeypatch):
    import app.diarize

    references = []
    events = []

    class Model:
        def __init__(self, *args, **kwargs):
            references.append(weakref.ref(self))

        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(
                start=0, end=1, text=" Labas", avg_logprob=-0.1, no_speech_prob=0.01,
                words=[SimpleNamespace(start=0, end=1, word="Labas", probability=0.9)],
            )]), SimpleNamespace(language="lt", duration=1)

    def align(segments, *args, **kwargs):
        assert references[0]() is None
        events.append("align")
        return {"segments": [{
            "start": 0.1, "end": 0.9,
            "words": [{"start": 0.1, "end": 0.9, "word": "Labas", "score": 0.95}],
        }]}

    def diarize(path, segments, settings):
        assert references[0]() is None
        assert events == ["cache", "align", "cache"]
        segments[0]["speaker"] = "SPEAKER_00"
        return segments

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    load_model = Mock(return_value=(object(), {}))
    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(
        load_align_model=load_model, load_audio=Mock(return_value=[]), align=align,
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: events.append("cache")),
    ))
    monkeypatch.setattr(app.diarize, "diarize", diarize)
    config = PipelineConfig()
    config.transcription.align = True
    config.transcription.align_model = "test-aligner"
    config.diarization.enabled = True
    config.diarization.speaker_prefix = True
    result = transcribe("video.mp4", config, tmp_path)
    load_model.assert_called_once_with(
        language_code="lt", device="cuda", model_name="test-aligner",
    )
    assert result["segments"][0]["words"][0]["start"] == 0.1
    assert result["segments"][0]["avg_logprob"] == -0.1
    assert json.loads((tmp_path / "transcript.json").read_text()) == result
    assert "[SPEAKER_00] Labas" in (tmp_path / "video.source.srt").read_text()


@pytest.mark.parametrize("unavailable", [True, False])
def test_alignment_failure_keeps_transcript(tmp_path, monkeypatch, caplog, unavailable):
    model = Mock()
    model.transcribe.return_value = ([], SimpleNamespace(language="lt", duration=1))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=Mock(return_value=model),
    ))
    monkeypatch.setitem(sys.modules, "whisperx", None if unavailable else SimpleNamespace(
        load_align_model=Mock(side_effect=RuntimeError("No default alignment model")),
    ))
    config = PipelineConfig()
    config.transcription.align = True
    result = transcribe("video.mp4", config, tmp_path)
    assert result == {"language": "lt", "duration": 1, "segments": []}
    assert "keeping Whisper timestamps" in caplog.text
