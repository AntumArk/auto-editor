"""Local, single-worker web interface for the presentation pipeline."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
from pathlib import Path
import re
import secrets
from tempfile import TemporaryDirectory
from threading import Lock

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from markupsafe import Markup
import mistune
from werkzeug.utils import secure_filename

from app.config import PipelineConfig, restore_job_settings
from app.ingest import IngestedFile, ingest
from app.main import _process_job
from app.state import JobState, STAGES
from app.web_settings import GROUPS, SETTINGS, parse_settings, settings_key, snapshot


OUTPUT_SUFFIXES = (
    ".fcpxml", ".cutlist.json", ".source.srt", ".edited.srt", ".report.md", ".preview.mp4",
)


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

    def submit(self, source: Path, config: PipelineConfig, move: bool, original_name=None) -> str:
        with self.lock:
            imported = ingest(
                source, config, move=move, settings_key=settings_key(config),
                original_name=original_name,
            )
            state = JobState.create(
                imported.job_id, imported.job_dir,
                str(imported.video_path), imported.original_name,
            )
            if imported.job_id in self.active or state.is_complete("done"):
                return imported.job_id
            state.artifacts["settings"] = snapshot(config)
            state.status, state.error = "queued", None
            state.save()
            self.active.add(imported.job_id)
            self.executor.submit(self._run, imported, config)
            return imported.job_id

    def resume(self, imported, config):
        with self.lock:
            if imported.job_id in self.active:
                return
            state = JobState.load(imported.job_dir / "state.json")
            if state.is_complete("done"):
                return
            state.status, state.error = "queued", None
            state.save()
            self.active.add(imported.job_id)
            self.executor.submit(self._run, imported, config)

    def preview(self, imported, config):
        with self.lock:
            if imported.job_id in self.active:
                return
            state = JobState.load(imported.job_dir / "state.json")
            state.artifacts["preview"] = {"status": "queued"}
            state.save()
            self.active.add(imported.job_id)
            self.executor.submit(self._preview, imported, config)

    def _preview(self, imported, config):
        from app.cutlist import CutList, render_preview

        state = JobState.load(imported.job_dir / "state.json")
        try:
            state.artifacts["preview"] = {"status": "running"}
            state.save()
            cuts = CutList.from_dict(json.loads((imported.job_dir / "cutlist.json").read_text()))
            render_preview(
                imported.video_path, cuts, Path(config.paths.output) / imported.job_id,
                name=Path(imported.original_name).stem,
            )
            state.artifacts["preview"] = {"status": "done"}
        except Exception as exc:
            logging.getLogger(__name__).exception("Preview %s failed", imported.job_id)
            state.artifacts["preview"] = {"status": "failed", "error": str(exc)}
        finally:
            state.save()
            with self.lock:
                self.active.discard(imported.job_id)

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
            logging.getLogger(__name__).exception("Job %s failed", imported.job_id)
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
        if state.artifacts.get("preview", {}).get("status") in ("queued", "running"):
            state.artifacts["preview"] = {
                "status": "failed", "error": "Server stopped during preview. Try rendering again.",
            }
            state.save()
        if state.status in ("queued", "running") and not state.is_complete("done"):
            state.status = "failed"
            state.error = "Server stopped before completion. Retry from the saved stage to resume."
            state.save()

    def outputs(job_id):
        directory = _contained(output, job_id)
        if not directory.is_dir():
            return []
        state = job_state(job_id)
        preview_ready = (
            state.artifacts.get("preview", {}).get("status") == "done"
            or (not state.artifacts.get("preview") and state.artifacts.get("cut_exported") is True)
        )
        return [
            path.name for path in sorted(directory.iterdir())
            if path.is_file() and path.name.endswith(OUTPUT_SUFFIXES)
            and path.resolve().is_relative_to(directory)
            and (not path.name.endswith(".preview.mp4") or preview_ready)
        ]

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'self'; script-src 'self'; "
            "media-src 'self'; connect-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    @app.get("/jobs")
    def jobs():
        return render_template(
            "jobs.html", jobs=states(), token=token, config=config,
            settings=SETTINGS, groups=GROUPS,
        )

    def check_token():
        if not secrets.compare_digest(request.form.get("token", "").encode(), token.encode()):
            abort(403)

    def imported_file(state):
        directory = _contained(workdir, state.job_id)
        sources = [path for path in directory.glob("source.*")
                   if path.suffix.lower() in extensions]
        if len(sources) != 1:
            abort(409, "Source video is missing or ambiguous")
        source = _contained(directory, sources[0].name)
        if not source.is_file():
            abort(404)
        return IngestedFile(state.job_id, source, directory, state.original_name or source.name)

    @app.post("/jobs")
    def submit():
        check_token()
        settings = parse_settings(config, request.form)
        upload = request.files.get("video")
        local_path = request.form.get("path", "").strip()
        if bool(upload and upload.filename) == bool(local_path):
            abort(400, "Choose either an upload or a server file path")

        def validate_video(path):
            if path.suffix.lower() not in extensions:
                abort(400, "Unsupported video extension")

        if local_path:
            source = _contained(inbox, local_path)
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
        details = {}
        review_warnings = []
        for stage, filename in (
            ("candidates", "candidates.json"), ("llm", "decisions.json"), ("cut", "cutlist.json"),
        ):
            if state.is_complete(stage):
                path = _contained(_contained(workdir, job_id), filename)
                if path.is_file():
                    details[stage] = json.loads(path.read_text(encoding="utf-8"))
                else:
                    review_warnings.append(
                        f"Saved review artifact {filename} is missing; its details cannot be shown."
                    )
        decisions = {item["candidate_id"]: item for item in details.get("llm", [])}
        reviews = [
            {"candidate": item, "decision": decisions.get(item["id"])}
            for item in details.get("candidates", [])
        ]
        saved = state.artifacts.get("settings")
        job_config = restore_job_settings(config, saved) if saved else deepcopy(config)
        return render_template(
            "job.html", state=state, files=files, report=rendered_report,
            stages=STAGES, completed=STAGES.index(state.stage) + 1,
            token=token, config=job_config, settings=SETTINGS, groups=GROUPS,
            saved_settings=saved, reviews=reviews, cuts=details.get("cut"),
            review_warnings=review_warnings,
            previews=[name for name in files if name.endswith(".preview.mp4")],
            preview=state.artifacts.get("preview", {}),
        )

    @app.post("/jobs/<job_id>/rerun")
    def rerun(job_id):
        check_token()
        state = job_state(job_id)
        imported = imported_file(state)
        defaults = restore_job_settings(config, state.artifacts.get("settings", {}))
        settings = parse_settings(defaults, request.form)
        new_id = worker.submit(
            imported.video_path, settings, move=False, original_name=imported.original_name,
        )
        return redirect(url_for("job", job_id=new_id), code=303)

    @app.post("/jobs/<job_id>/resume")
    def resume(job_id):
        check_token()
        state = job_state(job_id)
        worker.resume(
            imported_file(state), restore_job_settings(config, state.artifacts.get("settings", {})),
        )
        return redirect(url_for("job", job_id=job_id), code=303)

    @app.post("/jobs/<job_id>/preview")
    def preview(job_id):
        check_token()
        state = job_state(job_id)
        if not state.is_complete("done"):
            abort(409, "Finish the job before rendering a preview")
        worker.preview(imported_file(state), config)
        return redirect(url_for("job", job_id=job_id), code=303)

    @app.get("/jobs/<job_id>/status")
    def status(job_id):
        state = job_state(job_id)
        return {
            "stage": state.stage, "status": state.status, "error": state.error,
            "preview": state.artifacts.get("preview", {}), "files": outputs(job_id),
        }

    @app.get("/jobs/<job_id>/source")
    def source_video(job_id):
        imported = imported_file(job_state(job_id))
        return send_file(imported.video_path, conditional=True)

    @app.get("/jobs/<job_id>/media/<filename>")
    def media(job_id, filename):
        job_state(job_id)
        if not filename.endswith(".preview.mp4") or filename not in outputs(job_id):
            abort(404)
        return send_file(_contained(_contained(output, job_id), filename), conditional=True)

    @app.get("/jobs/<job_id>/download/<filename>")
    def download(job_id, filename):
        job_state(job_id)
        if filename not in outputs(job_id):
            abort(404)
        path = _contained(_contained(output, job_id), filename)
        return send_file(path, as_attachment=True, download_name=filename)

    return app
