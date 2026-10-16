import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.state import JobState


def test_state_persists_and_resumes(tmp_path):
    state = JobState.create("job-1", tmp_path, "/inbox/video.mp4")
    state.mark_stage("transcribed", "transcript.json")

    resumed = JobState.load(tmp_path / "state.json")

    assert resumed.stage == "transcribed"
    assert resumed.is_complete("ingested")
    assert resumed.is_complete("transcribed")
    assert not resumed.is_complete("visual")
    assert resumed.artifacts["transcribed"] == "transcript.json"


def test_state_rejects_skipped_stage(tmp_path):
    state = JobState.create("job-2", tmp_path, "/inbox/video.mp4")

    with pytest.raises(ValueError):
        state.mark_stage("visual")


def test_repeating_completed_stage_does_not_regress_state(tmp_path):
    state = JobState.create("job-3", tmp_path, "/inbox/video.mp4")
    state.mark_stage("transcribed")
    state.mark_stage("ingested")

    assert state.stage == "transcribed"


def test_progress_persists_without_completing_stage(tmp_path, monkeypatch):
    state = JobState.create("job-progress", tmp_path, "source.mp4")
    monkeypatch.setattr("app.state.time.time", lambda: 100)
    state.report_progress("transcribed", "Loading model.")
    monkeypatch.setattr("app.state.time.time", lambda: 110)
    state.report_progress("transcribed", "Transcribing.", 30, 120)
    restored = JobState.load(state.path)
    assert restored.stage == "ingested"
    assert restored.artifacts["progress"] == {
        "stage": "transcribed", "message": "Transcribing.",
        "completed": 30, "total": 120, "started_at": 100, "updated_at": 110,
    }
    state.mark_stage("transcribed")
    assert "progress" not in JobState.load(state.path).artifacts


def test_new_activity_resets_stage_timer_and_measurements(tmp_path, monkeypatch):
    state = JobState.create("job-progress", tmp_path, "source.mp4")
    monkeypatch.setattr("app.state.time.time", lambda: 100)
    state.report_progress("transcribed", "Transcribing.", 30, 120)
    monkeypatch.setattr("app.state.time.time", lambda: 200)
    state.report_progress("visual", "Detecting slides.")
    progress = JobState.load(state.path).artifacts["progress"]
    assert progress["started_at"] == 200
    assert progress["completed"] is None
    assert progress["total"] is None
