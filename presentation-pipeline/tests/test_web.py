import io
import json
from pathlib import Path
import re
import sys
from threading import Event
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main, web
from app.config import PipelineConfig
from app.state import JobState, STAGES


@pytest.fixture
def ui(tmp_path, monkeypatch):
    config = PipelineConfig()
    config.paths.inbox = str(tmp_path / "inbox")
    config.paths.workdir = str(tmp_path / "work")
    config.paths.output = str(tmp_path / "output")
    config.transcription.device = "cpu"
    config.llm.enabled = False

    def process(imported, settings):
        state = JobState.load(imported.job_dir / "state.json")
        for stage in STAGES[1:]:
            state.mark_stage(stage)
        state.artifacts["cut_exported"] = True
        state.save()
        output = Path(settings.paths.output) / imported.job_id
        output.mkdir()
        for suffix in web.OUTPUT_SUFFIXES:
            (output / f"talk{suffix}").write_text(
                '# Report\n\n<script>alert(1)</script>\n\n[bad](javascript:alert(1))'
            )
        return output

    run = Mock(side_effect=process)
    monkeypatch.setattr(web, "_process_job", run)
    app = web.create_app(config)
    app.testing = True
    client = app.test_client()
    token = re.search(r'name="token" value="([^"]+)"', client.get("/").text).group(1)
    yield config, app, client, token, run
    app.extensions["pipeline_worker"].executor.shutdown(wait=True)


def drain(app):
    app.extensions["pipeline_worker"].executor.submit(lambda: None).result(timeout=5)


def upload(client, token, filename="talk.mp4", **options):
    return client.post("/jobs", data={
        "token": token, "video": (io.BytesIO(b"video" * 200_000), filename),
        "model": "small", "language": "", **options,
    })


def test_upload_job_detail_report_and_outputs(ui):
    config, app, client, token, run = ui
    response = upload(client, token, "../../talk.mp4", llm="on")
    assert response.status_code == 303
    drain(app)
    imported, settings = run.call_args.args
    assert imported.original_name == "talk.mp4"
    assert imported.video_path.read_bytes() == b"video" * 200_000
    assert settings.transcription.model == "small"
    assert settings.transcription.device == "cpu"
    assert settings.transcription.language is None
    assert settings.llm.enabled is True
    assert config.llm.enabled is False
    assert config.transcription.model == "large-v3"
    page = client.get(response.location)
    assert page.status_code == 200
    assert "done" in page.text
    assert 'http-equiv="refresh"' not in page.text
    assert "<h1>Report</h1>" in page.text
    assert "<script>" not in page.text
    assert 'href="javascript:' not in page.text
    assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]
    for suffix in web.OUTPUT_SUFFIXES:
        download = client.get(f"{response.location}/download/talk{suffix}")
        assert download.status_code == 200
        assert "attachment" in download.headers["Content-Disposition"]
    assert "talk.mp4" in client.get("/jobs").text
    assert list((Path(config.paths.workdir) / ".uploads").iterdir()) == []


def test_server_file_is_preserved_and_completed_job_is_reused(ui):
    config, app, client, token, run = ui
    source = Path(config.paths.inbox) / "talk.mp4"
    source.write_bytes(b"video")
    data = {"token": token, "path": str(source), "model": "base", "language": "lt"}
    response = client.post("/jobs", data=data)
    assert response.status_code == 303
    drain(app)
    assert source.read_bytes() == b"video"
    assert run.call_args.args[1].transcription.language == "lt"
    assert run.call_args.args[1].llm.enabled is False
    assert client.post("/jobs", data=data).location == response.location
    drain(app)
    assert run.call_count == 1


def test_worker_records_error_and_keeps_last_completed_stage(ui):
    config, app, client, token, run = ui

    def fail(imported, settings):
        state = JobState.load(imported.job_dir / "state.json")
        state.mark_stage("transcribed")
        raise RuntimeError("<script>unsupported GPU</script>")

    run.side_effect = fail
    response = upload(client, token)
    drain(app)
    page = client.get(response.location)
    assert "failed" in page.text
    assert 'http-equiv="refresh"' not in page.text
    assert "Last completed stage: transcribed" in page.text
    assert "&lt;script&gt;unsupported GPU" in page.text
    state = JobState.load(next(Path(config.paths.workdir).glob("*/state.json")))
    assert state.status == "failed"
    assert "unsupported GPU" in state.error


def test_single_worker_queues_jobs_and_deduplicates_active_source(ui):
    config, app, client, token, run = ui
    started, release = Event(), Event()
    original = run.side_effect

    def block(imported, settings):
        started.set()
        assert release.wait(timeout=5)
        return original(imported, settings)

    run.side_effect = block
    try:
        first = upload(client, token)
        assert started.wait(timeout=5)
        second = upload(client, token, "second.mp4")
        duplicate = upload(client, token)
        assert first.location == duplicate.location
        assert run.call_count == 1
        assert "queued" in client.get(second.location).text
        assert client.get(second.location + "/status").json["status"] == "queued"
        assert client.get(first.location + "/status").json["status"] == "running"
        assert 'data-status-url=' in client.get(first.location).text
    finally:
        release.set()
    drain(app)
    assert run.call_count == 2
    assert "done" in client.get(second.location).text


@pytest.mark.parametrize("data", [
    {}, {"path": "missing.mp4"}, {"path": "../outside.mp4"}, {"path": "/etc/passwd"},
    {"path": "bad\0.mp4"},
    {"path": "talk.txt"}, {"path": "talk.mp4", "language": "<script>"},
    {"path": "talk.mp4", "model": "../../model"},
])
def test_rejects_invalid_submissions(ui, data):
    config, app, client, token, run = ui
    response = client.post("/jobs", data={"token": token, **data})
    assert response.status_code in (400, 404)
    assert run.call_count == 0


def test_rejects_csrf_wrong_extensions_and_ambiguous_input(ui):
    config, app, client, token, run = ui
    assert upload(client, "wrong").status_code == 403
    assert upload(client, "neteisingas-žetonas").status_code == 403
    assert upload(client, token, "script.exe").status_code == 400
    assert upload(client, token, path="talk.mp4").status_code == 400
    assert run.call_count == 0
    assert client.get("/jobs/missing").status_code == 404


def test_upload_size_limit(ui):
    config, app, client, token, run = ui
    app.config["MAX_CONTENT_LENGTH"] = 128
    assert upload(client, token).status_code == 413
    assert run.call_count == 0


def test_settings_saved_changed_settings_create_separate_job_and_rerun_preserves_source(ui):
    config, app, client, token, run = ui
    response = upload(client, token, **{
        "cuts.target_pause_s": "1.5", "export.preview": "on",
        "visual.sample_fps": "2", "report.language": "en",
    })
    drain(app)
    imported, settings = run.call_args.args
    assert settings.cuts.target_pause_s == 1.5
    assert settings.export.preview is True
    assert settings.visual.sample_fps == 2
    state = JobState.load(imported.job_dir / "state.json")
    assert state.artifacts["settings"]["report"]["language"] == "en"
    assert "url" not in state.artifacts["settings"]["llm"]
    changed = client.post(response.location + "/rerun", data={
        "token": token, "cuts.target_pause_s": "2",
    })
    assert changed.status_code == 303
    assert changed.location != response.location
    drain(app)
    new_imported, new_settings = run.call_args.args
    assert new_imported.original_name == "talk.mp4"
    assert new_settings.cuts.target_pause_s == 2
    assert new_settings.report.language == "en"
    assert imported.video_path.is_file()
    assert JobState.load(imported.job_dir / "state.json").artifacts["settings"]["cuts"]["target_pause_s"] == 1.5
    again = client.post(changed.location + "/rerun", data={
        "token": token, "cuts.target_pause_s": "2",
    })
    drain(app)
    assert again.location == changed.location
    assert run.call_count == 2


def test_source_and_preview_range_requests_and_mime(ui):
    _, app, client, token, run = ui
    response = upload(client, token)
    drain(app)
    source = client.get(response.location + "/source", headers={"Range": "bytes=0-9"})
    assert source.status_code == 206
    assert source.data == b"videovideo"
    assert source.mimetype == "video/mp4"
    preview = client.get(response.location + "/media/talk.preview.mp4",
                         headers={"Range": "bytes=0-5"})
    assert preview.status_code == 206
    assert preview.mimetype == "video/mp4"
    assert len(preview.data) == 6
    assert client.get(response.location + "/media/talk.report.md").status_code == 404
    assert client.get(response.location + "/media/unknown.preview.mp4").status_code == 404
    assert "media-src 'self'" in preview.headers["Content-Security-Policy"]


def test_preview_render_uses_cached_cutlist_and_does_not_reprocess(ui, monkeypatch):
    from app import cutlist

    _, app, client, token, run = ui
    response = upload(client, token)
    drain(app)
    imported, _ = run.call_args.args
    (imported.job_dir / "cutlist.json").write_text(json.dumps({
        "removed": [[2, 4]], "keep": [[0, 2], [4, 10]],
        "original_duration": 10, "edited_duration": 8,
    }))
    render = Mock(return_value=None)
    monkeypatch.setattr(cutlist, "render_preview", render)
    assert client.post(response.location + "/preview", data={"token": "wrong"}).status_code == 403
    assert client.post(response.location + "/preview", data={"token": token}).status_code == 303
    drain(app)
    assert run.call_count == 1
    assert render.call_args.args[0] == imported.video_path
    assert render.call_args.args[1].removed == [(2, 4)]
    assert render.call_args.kwargs["name"] == "talk"
    assert client.get(response.location + "/status").json["preview"]["status"] == "done"
    render.side_effect = RuntimeError("<bad preview>")
    client.post(response.location + "/preview", data={"token": token})
    drain(app)
    page = client.get(response.location)
    assert "&lt;bad preview&gt;" in page.text
    assert client.get(response.location + "/status").json["preview"]["status"] == "failed"
    assert client.get(response.location + "/media/talk.preview.mp4").status_code == 404


def test_review_displays_cached_decisions_and_escaped_reasons(ui):
    _, app, client, token, run = ui
    response = upload(client, token)
    drain(app)
    imported, _ = run.call_args.args
    (imported.job_dir / "candidates.json").write_text(json.dumps([
        {"id": "one", "start": 1, "end": 5, "duration": 4, "context": {}},
    ]))
    (imported.job_dir / "decisions.json").write_text(json.dumps([
        {"candidate_id": "one", "action": "keep", "target_pause_s": None,
         "reason": "<img src=x onerror=alert(1)>", "needs_review": True},
    ]))
    (imported.job_dir / "cutlist.json").write_text(json.dumps({
        "original_duration": 10, "edited_duration": 10, "removed": [], "keep": [[0, 10]],
    }))
    page = client.get(response.location)
    assert page.status_code == 200
    assert "Needs review" in page.text
    assert "&lt;img src=x" in page.text
    assert "<img src=x" not in page.text
    assert 'data-seek="1"' in page.text
    assert 'id="cut-overview"' in page.text


def test_new_routes_cannot_escape_roots_and_require_csrf(ui, tmp_path):
    config, app, client, token, run = ui
    response = upload(client, token)
    drain(app)
    imported, _ = run.call_args.args
    for action in ("rerun", "resume", "preview"):
        assert client.post(response.location + "/" + action, data={"token": "wrong"}).status_code == 403
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private")
    imported.video_path.unlink()
    imported.video_path.symlink_to(outside)
    assert client.get(response.location + "/source").status_code == 404
    assert client.post(response.location + "/rerun", data={"token": token}).status_code == 404
    output = Path(config.paths.output) / imported.job_id / "talk.preview.mp4"
    output.unlink()
    output.symlink_to(outside)
    assert client.get(response.location + "/media/talk.preview.mp4").status_code == 404


def test_retry_failed_job_restores_settings(ui):
    _, app, client, token, run = ui
    original = run.side_effect
    run.side_effect = RuntimeError("temporary failure")
    response = upload(client, token, **{"cuts.target_pause_s": "1.75"})
    drain(app)
    assert "Retry from saved stage" in client.get(response.location).text
    run.side_effect = original
    assert client.post(response.location + "/resume", data={"token": token}).status_code == 303
    drain(app)
    assert run.call_count == 2
    assert run.call_args.args[1].cuts.target_pause_s == 1.75
    assert client.get(response.location + "/status").json["status"] == "done"


def test_upload_extensions_follow_watcher_config(ui):
    config, _, _, _, run = ui
    config.watcher.video_extensions = (".clip",)
    app = web.create_app(config)
    try:
        client = app.test_client()
        token = re.search(r'name="token" value="([^"]+)"', client.get("/").text).group(1)
        assert upload(client, token).status_code == 400
        assert upload(client, token, "talk.CLIP").status_code == 303
        drain(app)
        assert run.call_count == 1
    finally:
        app.extensions["pipeline_worker"].executor.shutdown(wait=True)


def test_downloads_and_local_input_cannot_escape_roots(ui, tmp_path):
    config, app, client, token, run = ui
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private")
    (Path(config.paths.inbox) / "linked.mp4").symlink_to(outside)
    assert client.post("/jobs", data={
        "token": token, "path": "linked.mp4",
    }).status_code == 404
    response = upload(client, token)
    drain(app)
    imported = run.call_args.args[0]
    output = Path(config.paths.output) / imported.job_id
    (output / "private.report.md").symlink_to(outside)
    assert client.get(response.location + "/download/private.report.md").status_code == 404
    (output / "secret.txt").write_text("private")
    assert client.get(response.location + "/download/secret.txt").status_code == 404
    assert client.get(response.location + "/download/%2e%2e%2foutside.mp4").status_code == 404
    assert client.get("/jobs/%2e%2e").status_code == 404
    assert "private.report.md" not in client.get(response.location).text
    moved = tmp_path / "moved-output"
    output.rename(moved)
    output.symlink_to(moved, target_is_directory=True)
    assert client.get(response.location + "/download/talk.fcpxml").status_code == 404


def test_restart_marks_incomplete_web_jobs_failed(tmp_path):
    config = PipelineConfig()
    config.paths.inbox = str(tmp_path / "inbox")
    config.paths.workdir = str(tmp_path / "work")
    config.paths.output = str(tmp_path / "output")
    state = JobState.create("talk-123", tmp_path / "work" / "talk-123", "source.mp4")
    state.status = "running"
    state.save()
    app = web.create_app(config)
    try:
        restored = JobState.load(state.path)
        assert restored.status == "failed"
        assert "Retry from the saved stage" in restored.error
    finally:
        app.extensions["pipeline_worker"].executor.shutdown(wait=True)


@pytest.mark.parametrize("options,host,port", [
    ([], "127.0.0.1", 8000),
    (["--host", "0.0.0.0", "--port", "8080"], "0.0.0.0", 8080),
])
def test_serve_cli(monkeypatch, options, host, port):
    app = Mock()
    monkeypatch.setattr(web, "create_app", Mock(return_value=app))
    monkeypatch.setattr(main, "load_config", Mock(return_value=PipelineConfig()))
    monkeypatch.setattr(sys, "argv", ["pipeline", "serve", *options])
    main.main()
    app.run.assert_called_once_with(
        host=host, port=port, debug=False, use_reloader=False,
    )
