import sys
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PipelineConfig
from app.transcribe import _align_words, transcribe


@pytest.fixture(autouse=True)
def local_diarization_stub(monkeypatch):
    monkeypatch.setattr("app.diarize.diarize", lambda path, segments, settings: segments)


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


@pytest.mark.parametrize("disabled", [False, True])
def test_local_diarization_enabled_by_default_can_be_disabled(tmp_path, monkeypatch, disabled):
    model = Mock()
    model.transcribe.return_value = ([], SimpleNamespace(language="lt", duration=1))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=Mock(return_value=model),
    ))
    diarize = Mock(return_value=[])
    monkeypatch.setattr("app.diarize.diarize", diarize)
    config = PipelineConfig()
    if disabled:
        config.diarization.enabled = False
    transcribe("video.mp4", config, tmp_path)
    assert diarize.call_count == (0 if disabled else 1)


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


def test_alignment_merges_split_sentences_without_losing_metadata(monkeypatch):
    result = {"language": "lt", "duration": 5, "segments": [
        {"start": 0, "end": 3, "text": "Labas. 2026!", "avg_logprob": -0.2,
         "words": [{"start": 0, "end": 1, "word": "Labas."}]},
        {"start": 3, "end": 5, "text": "Vilnius.", "avg_logprob": -0.1, "words": []},
    ]}
    align = Mock(side_effect=[
        {"segments": [
            {"start": 0.1, "end": 1.1,
             "words": [{"start": 0.1, "end": 1.1, "word": "Labas."}]},
            {"start": 1.2, "end": 2.9,
             "words": [{"word": "2026!", "start": None}]},
        ]},
        {"segments": [{"start": 3.2, "end": 4.8, "words": [
            {"start": 3.2, "end": 4.8, "word": "Vilnius."},
        ]}]},
    ])
    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(
        load_align_model=lambda **kw: (object(), {}),
        load_audio=lambda path: [], align=align,
    ))
    refined = _align_words("video.mp4", result, {}, "cpu")
    assert len(refined["segments"]) == 2
    assert [segment["text"] for segment in refined["segments"]] == ["Labas. 2026!", "Vilnius."]
    assert [segment["avg_logprob"] for segment in refined["segments"]] == [-0.2, -0.1]
    assert refined["segments"][0]["start"] == 0.1
    assert refined["segments"][0]["end"] == 2.9
    assert refined["segments"][1]["words"][0]["start"] == 3.2
    assert refined["segments"][0]["words"][1] == {
        "word": "2026!", "start": 0.1, "end": 2.9,
    }
    assert result["segments"][0]["start"] == 0
    assert len(result["segments"][0]["words"]) == 1
    assert [call.args[0][0]["text"] for call in align.call_args_list] == [
        "Labas. 2026!", "Vilnius.",
    ]


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (2, 1), (-1, 1)])
def test_invalid_alignment_timestamps_gracefully_keep_original(monkeypatch, caplog, start, end):
    result = {"language": "lt", "segments": [
        {"start": 0, "end": 1, "text": "Labas", "words": []},
    ]}
    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(
        load_align_model=lambda **kw: (object(), {}), load_audio=lambda path: [],
        align=lambda *a, **kw: {"segments": [{"start": start, "end": end}]},
    ))
    assert _align_words("video.mp4", result, {}, "cpu") is result
    assert "keeping Whisper timestamps" in caplog.text


def test_failed_lazy_model_is_released_before_retry(tmp_path, monkeypatch):
    references = []

    class Model:
        def __init__(self, name, device, compute_type):
            if references:
                assert references[-1]() is None
            references.append(weakref.ref(self))
            self.device = device

        def transcribe(self, *args, **kwargs):
            return self.segments(), SimpleNamespace(language="lt", duration=1)

        def segments(self):
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            yield from []

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    transcribe("video.mp4", PipelineConfig(), tmp_path)
    assert all(reference() is None for reference in references)
