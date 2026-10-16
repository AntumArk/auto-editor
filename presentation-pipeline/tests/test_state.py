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
