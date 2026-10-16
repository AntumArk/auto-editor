"""Local, single-worker web interface for the presentation pipeline."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import re
import secrets
from tempfile import TemporaryDirectory
from threading import Lock

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from markupsafe import Markup
import mistune
from werkzeug.utils import secure_filename

from app.config import PipelineConfig
from app.ingest import ingest
from app.main import _process_job
from app.state import JobState, STAGES


OUTPUT_SUFFIXES = (".fcpxml", ".cutlist.json", ".source.srt", ".edited.srt", ".report.md")
MODELS = ("tiny", "base", "small", "medium", "large-v3")


def _contained(root: Path, *parts: str) -> Path:
    try:
        path = root.joinpath(*parts).resolve()
    except (OSError, ValueError):
        abort(404)
    if path == root or not path.is_relative_to(root):
        abort(404)
    return path


class JobWorker:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self.lock = Lock()
        self.active = set()

    def submit(self, source: Path, config: PipelineConfig, move: bool) -> str:
        with self.lock:
            imported = ingest(source, config, move=move)
            state = JobState.create(
                imported.job_id, imported.job_dir,
                str(imported.video_path), imported.original_name,
            )
            if imported.job_id in self.active or state.is_complete("done"):
                return imported.job_id
            state.status, state.error = "queued", None
            state.save()
            self.active.add(imported.job_id)
            self.executor.submit(self._run, imported, config)
            return imported.job_id

    def _run(self, imported, config):
        try:
            state = JobState.load(imported.job_dir / "state.json")
            state.status = "running"
            state.save()
            _process_job(imported, config)
            state = JobState.load(state.path)
            state.status, state.error = "done", None
            state.save()
        except Exception as exc:
            state = JobState.load(imported.job_dir / "state.json")
            state.status, state.error = "failed", str(exc)
            state.save()
        finally:
            with self.lock:
                self.active.discard(imported.job_id)


def create_app(config: PipelineConfig) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024**3
    token = secrets.token_urlsafe(32)
    workdir = Path(config.paths.workdir).resolve()
    output = Path(config.paths.output).resolve()
    inbox = Path(config.paths.inbox).resolve()
    for directory in (workdir, output, inbox):
        directory.mkdir(parents=True, exist_ok=True)
    uploads = workdir / ".uploads"
    uploads.mkdir(exist_ok=True)
    worker = JobWorker()
    app.extensions["pipeline_worker"] = worker
    markdown = mistune.create_markdown(escape=True)
    extensions = {extension.lower() for extension in config.watcher.video_extensions}

    def job_state(job_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id):
            abort(404)
        path = _contained(workdir, job_id, "state.json")
        if not path.is_file():
            abort(404)
        return JobState.load(path)

    def states():
        result = []
        for path in sorted(workdir.glob("*/state.json")):
            if not path.resolve().is_relative_to(workdir):
                continue
            try:
                result.append(JobState.load(path))
            except (OSError, ValueError, KeyError):
                continue
        return result

    for state in states():
        if state.status in ("queued", "running") and not state.is_complete("done"):
            state.status = "failed"
            state.error = "Server stopped before completion. Resubmit the source to resume."
            state.save()

    def outputs(job_id):
        directory = _contained(output, job_id)
        if not directory.is_dir():
            return []
        return [
            path.name for path in sorted(directory.iterdir())
            if path.is_file() and path.name.endswith(OUTPUT_SUFFIXES)
            and path.resolve().is_relative_to(directory)
        ]

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    @app.get("/jobs")
    def jobs():
        return render_template(
            "jobs.html", jobs=states(), token=token, models=MODELS, config=config,
        )

    @app.post("/jobs")
    def submit():
        if not secrets.compare_digest(request.form.get("token", "").encode(), token.encode()):
            abort(403)
        settings = deepcopy(config)
        model = request.form.get("model", settings.transcription.model)
        if model not in (*MODELS, settings.transcription.model):
            abort(400, "Unsupported transcription model")
        language = request.form.get("language", settings.transcription.language or "").strip()
        if language and not re.fullmatch(r"[a-z]{2,3}", language):
            abort(400, "Language must be a two- or three-letter code, or blank for auto")
        settings.transcription.model = model
        settings.transcription.language = language or None
        settings.llm.enabled = request.form.get("llm") == "on"
        upload = request.files.get("video")
        local_path = request.form.get("path", "").strip()
        if bool(upload and upload.filename) == bool(local_path):
            abort(400, "Choose either an upload or a server file path")

        def validate_video(path):
            if path.suffix.lower() not in extensions:
                abort(400, "Unsupported video extension")

        if local_path:
            source = Path(local_path).expanduser()
            source = _contained(inbox, str(source))
            validate_video(source)
            if not source.is_file():
                abort(400, "Server file does not exist in the inbox")
            job_id = worker.submit(source, settings, move=False)
        else:
            filename = secure_filename(upload.filename)
            validate_video(Path(filename))
            with TemporaryDirectory(dir=uploads) as temporary:
                source = Path(temporary) / filename
                with source.open("wb") as destination:
                    while chunk := upload.stream.read(1024 * 1024):
                        destination.write(chunk)
                job_id = worker.submit(source, settings, move=True)
        return redirect(url_for("job", job_id=job_id), code=303)

    @app.get("/jobs/<job_id>")
    def job(job_id):
        state = job_state(job_id)
        files = outputs(job_id)
        reports = [name for name in files if name.endswith(".report.md")]
        rendered_report = None
        if reports:
            report = _contained(_contained(output, job_id), reports[0])
            rendered_report = Markup(markdown(report.read_text(encoding="utf-8")))
        return render_template(
            "job.html", state=state, files=files, report=rendered_report,
            stages=STAGES, completed=STAGES.index(state.stage) + 1,
        )

    @app.get("/jobs/<job_id>/download/<filename>")
    def download(job_id, filename):
        job_state(job_id)
        if filename not in outputs(job_id):
            abort(404)
        path = _contained(_contained(output, job_id), filename)
        return send_file(path, as_attachment=True, download_name=filename)

    return app
