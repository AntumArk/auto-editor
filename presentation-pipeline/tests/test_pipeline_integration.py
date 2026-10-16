"""Exercise pipeline wiring without media tools, GPU models, or network."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import candidates, cutlist, llm, main, report, subtitles, transcribe, vision
from app.config import PipelineConfig
from app.ingest import ingest
from app.state import JobState


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    config = PipelineConfig()
    config.paths.workdir = str(tmp_path / "work")
    config.paths.output = str(tmp_path / "output")
    config.llm.enabled = False
    config.cuts.speech_margin_s = 0
    config.cuts.settle_after_slide_change_s = 0
    config.cuts.target_pause_s = 2
    config.export = SimpleNamespace(preview=True)
    source = tmp_path / "Original talk.mp4"
    source.write_bytes(b"mock video input")
    transcript = {"segments": [
        {"start": 0, "end": 5, "text": "Introduction."},
        {"start": 15, "end": 35, "text": "Second topic."},
        {"start": 45, "end": 70, "text": "Final topic."},
    ]}
    mocks = {}
    for module, name, options in (
        (main, "ffprobe_duration", {"return_value": 80}),
        (transcribe, "transcribe", {"return_value": transcript}),
        (vision, "detect_slide_changes", {"return_value": {
            "change_times": [0, 25, 55], "static_intervals": [(0, 80)],
        }}),
        (candidates, "generate_candidates", {"wraps": candidates.generate_candidates}),
        (llm, "review_candidates", {"wraps": llm.review_candidates}),
        (cutlist, "build_cutlist", {"wraps": cutlist.build_cutlist}),
        (subtitles, "write_srt", {"wraps": subtitles.write_srt}),
        (report, "write_report", {"wraps": report.write_report}),
        (report, "make_chapters", {"wraps": report.make_chapters}),
    ):
        mocks[name] = Mock(**options)
        monkeypatch.setattr(module, name, mocks[name])

    def export(command, *, check):
        assert check is True
        assert command[:4] == ["auto-editor", str(next(
            Path(config.paths.workdir).glob("*/source.mp4")
        )), "--edit", "all"]
        Path(command[-1]).write_text("mock export", encoding="utf-8")

    mocks["export"] = Mock(side_effect=export)
    monkeypatch.setattr(cutlist.subprocess, "run", mocks["export"])
    monkeypatch.setattr(
        llm.urllib.request, "urlopen", Mock(side_effect=AssertionError("Network forbidden"))
    )
    import requests

    monkeypatch.setattr(
        requests, "post", Mock(side_effect=AssertionError("Network forbidden"))
    )
    return config, source, mocks


def assert_outputs(output, config, mocks, keep=False):
    assert {path.name for path in output.iterdir()} == {
        "Original talk.fcpxml", "Original talk.cutlist.json",
        "Original talk.preview.mp4", "Original talk.source.srt",
        "Original talk.edited.srt", "Original talk.report.md",
    }
    audit = json.loads((output / "Original talk.cutlist.json").read_text())
    expected_removed = [[6, 14], [71, 79]] if keep else [[6, 14], [36, 44], [71, 79]]
    assert audit["removed"] == expected_removed
    assert audit["original_duration"] == 80
    assert audit["edited_duration"] == (64 if keep else 56)
    assert audit["edited_duration"] == 80 - sum(end - start for start, end in audit["removed"])
    source_srt = (output / "Original talk.source.srt").read_text()
    edited_srt = (output / "Original talk.edited.srt").read_text()
    assert "00:00:15,000 --> 00:00:35,000" in source_srt
    assert "00:00:07,000 --> 00:00:27,000" in edited_srt
    assert ("00:00:37,000" if keep else "00:00:29,000") in edited_srt
    text = (output / "Original talk.report.md").read_text()
    assert "00:00 Introduction." in text
    assert "Original duration: 80.00 s" in text
    assert f"Edited duration: {audit['edited_duration']:.2f} s" in text
    assert f"Time removed: {80 - audit['edited_duration']:.2f} s" in text
    assert f"Number of cuts: {len(expected_removed)}" in text
    assert mocks["make_chapters"].call_args.args[0] == [0, 17, 47 if keep else 39]
    if keep:
        assert "[00:27] LLM decision needs review: Deliberate pause" in text
    job_dir = Path(config.paths.workdir) / output.name
    assert json.loads((job_dir / "report.json").read_text()) == {
        "report": str(output / "Original talk.report.md")
    }
    assert JobState.load(job_dir / "state.json").is_complete("done")


@pytest.mark.parametrize("keep", [False, True])
def test_process_file_end_to_end_and_completed_job_is_not_recomputed(pipeline, keep):
    config, source, mocks = pipeline
    if keep:
        mocks["review_candidates"].side_effect = lambda items, *_: [
            llm.Decision(item.id, "keep" if index == 1 else "shorten", None,
                         "Deliberate pause", needs_review=index == 1)
            for index, item in enumerate(items)
        ]
    output = main.process_file(source, config, move=False)
    assert_outputs(output, config, mocks, keep)
    counts = {name: mock.call_count for name, mock in mocks.items()}
    assert main.process_file(source, config, move=False) == output
    assert {name: mock.call_count for name, mock in mocks.items()} == counts


@pytest.mark.parametrize("interruption", ["export", "write_report"])
def test_restart_restores_cutlist_and_skips_completed_stages(pipeline, interruption):
    config, source, mocks = pipeline
    imported = ingest(source, config, move=False)
    interrupted = mocks[interruption]
    original_effect = interrupted.side_effect
    interrupted.side_effect = RuntimeError("Simulated interruption")
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        main._process_job(imported, config)
    state = JobState.load(imported.job_dir / "state.json")
    assert state.stage == ("cut" if interruption == "export" else "subtitles")
    restored = main._restore(
        json.loads((imported.job_dir / "cutlist.json").read_text()), cutlist, "CutList"
    )
    assert isinstance(restored, cutlist.CutList)
    assert restored.keep[0] == (0, 6)
    interrupted.side_effect = original_effect
    main.resume_pending(config)
    output = Path(config.paths.output) / imported.job_id
    assert_outputs(output, config, mocks)
    for stage in ("transcribe", "detect_slide_changes", "generate_candidates",
                  "review_candidates", "build_cutlist"):
        assert mocks[stage].call_count == 1
    assert mocks["write_srt"].call_count == 2
    assert mocks["export"].call_count == (3 if interruption == "export" else 2)


def test_report_discards_slide_changes_inside_removed_ranges(pipeline, monkeypatch):
    config, source, mocks = pipeline
    monkeypatch.setattr(cutlist, "build_cutlist", Mock(return_value=cutlist.CutList(
        removed=[(20, 30)], keep=[(0, 20), (30, 80)],
        original_duration=80, edited_duration=70,
    )))
    main.process_file(source, config, move=False)
    assert mocks["make_chapters"].call_args.args[0] == [0, 45]


@pytest.mark.parametrize("prefix", [False, True])
def test_speaker_labels_reach_source_and_edited_subtitles(pipeline, prefix):
    config, source, mocks = pipeline
    config.diarization.speaker_prefix = prefix
    for segment in mocks["transcribe"].return_value["segments"]:
        segment["speaker"] = "SPEAKER_00"
    output = main.process_file(source, config, move=False)
    for suffix in ("source", "edited"):
        text = (output / f"Original talk.{suffix}.srt").read_text()
        assert ("[SPEAKER_00] Introduction." in text) is prefix
    assert mocks["write_srt"].call_args.args[0][0]["speaker"] == "SPEAKER_00"


@pytest.mark.parametrize("flag", ["--keep-source", "--copy", None])
def test_process_cli_source_handling(pipeline, monkeypatch, flag):
    config, source, mocks = pipeline
    monkeypatch.setattr(main, "load_config", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["pipeline", "process", str(source)] + ([flag] if flag else []))
    main.main()
    assert source.exists() == bool(flag)
    assert next(Path(config.paths.workdir).glob("*/source.mp4")).is_file()
