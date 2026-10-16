from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from app.config import PipelineConfig, load_config
from app.ingest import ingest
from app.state import JobState
from app.utils import ffprobe_duration


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, default=_json_default, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _module(name: str):
    try:
        return importlib.import_module(f"app.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"app.{name}":
            raise RuntimeError(
                f"Pipeline stage module app.{name} is not installed yet."
            ) from exc
        raise


def _restore(value: Any, module: Any, class_name: str) -> Any:
    if isinstance(value, list):
        return [_restore(item, module, class_name) for item in value]
    cls = getattr(module, class_name, None)
    if cls is not None and isinstance(value, dict):
        if hasattr(cls, "from_dict"):
            return cls.from_dict(value)
        try:
            return cls(**value)
        except TypeError:
            return value
    return value


def _stage(
    state: JobState,
    stage: str,
    artifact_path: Path,
    function,
):
    if state.is_complete(stage):
        return _read_json(artifact_path)
    result = function()
    _write_json(artifact_path, result)
    state.mark_stage(stage, str(artifact_path))
    return result


def _process_job(imported, config: PipelineConfig) -> Path:
    output_dir = Path(config.paths.output) / imported.job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    state = JobState.create(
        imported.job_id,
        imported.job_dir,
        str(imported.video_path),
        imported.original_name,
    )
    if state.is_complete("done"):
        return output_dir

    duration = ffprobe_duration(imported.video_path)
    transcript_path = imported.job_dir / "transcript.json"
    transcript = _stage(
        state,
        "transcribed",
        transcript_path,
        lambda: _module("transcribe").transcribe(
            imported.video_path, config, imported.job_dir
        ),
    )

    visual_path = imported.job_dir / "visual.json"
    slide_info = _stage(
        state,
        "visual",
        visual_path,
        lambda: _module("vision").detect_slide_changes(imported.video_path, config),
    )

    candidates_path = imported.job_dir / "candidates.json"
    candidates_module = _module("candidates")
    candidates_data = _stage(
        state,
        "candidates",
        candidates_path,
        lambda: candidates_module.generate_candidates(
            transcript, slide_info, duration, config
        ),
    )
    candidates = _restore(candidates_data, candidates_module, "Candidate")

    decisions_path = imported.job_dir / "decisions.json"
    llm_module = _module("llm")
    decisions_data = _stage(
        state,
        "llm",
        decisions_path,
        lambda: llm_module.review_candidates(candidates, transcript, slide_info, config),
    )
    decisions = _restore(decisions_data, llm_module, "Decision")

    cutlist_path = imported.job_dir / "cutlist.json"
    cutlist_module = _module("cutlist")
    cutlist_data = _stage(
        state,
        "cut",
        cutlist_path,
        lambda: cutlist_module.build_cutlist(candidates, decisions, duration, config),
    )
    cutlist = _restore(cutlist_data, cutlist_module, "CutList")
    source_stem = Path(imported.original_name).stem
    if state.artifacts.get("cut_exported") is not True:
        cutlist_module.export_with_auto_editor(
            imported.video_path, cutlist, output_dir, config, name=source_stem
        )
        state.artifacts["cut_exported"] = True
        state.save()

    subtitles_module = _module("subtitles")
    subtitles_dir = output_dir

    def write_subtitles():
        source_srt = subtitles_dir / f"{source_stem}.source.srt"
        edited_srt = subtitles_dir / f"{source_stem}.edited.srt"
        prefix_options = {"speaker_prefix": True} if config.diarization.speaker_prefix else {}
        subtitles_module.write_srt(transcript["segments"], source_srt, **prefix_options)
        remapped = subtitles_module.remap_segments(transcript["segments"], cutlist.keep)
        subtitles_module.write_srt(remapped, edited_srt, **prefix_options)
        return {"source": str(source_srt), "edited": str(edited_srt)}

    _stage(state, "subtitles", imported.job_dir / "subtitles.json", write_subtitles)

    report_module = _module("report")

    def write_report():
        summary = report_module.summarize(transcript, config)
        notes = report_module.review_notes(
            transcript, decisions, cutlist, config, candidates=candidates
        )
        edited_changes = [
            subtitles_module.map_time(time, cutlist.keep)
            for time in slide_info["change_times"]
            if not any(start <= time < end for start, end in cutlist.removed)
        ]
        edited_segments = subtitles_module.remap_segments(transcript["segments"], cutlist.keep)
        chapters = report_module.make_chapters(
            edited_changes, edited_segments, cutlist.edited_duration,
            config.report.min_chapter_gap_s,
        )
        stats = {
            "original_duration": cutlist.original_duration,
            "edited_duration": cutlist.edited_duration,
            "number_of_cuts": len(cutlist.removed),
        }
        report_path = output_dir / f"{source_stem}.report.md"
        report_module.write_report(
            report_path, source_stem, summary, chapters, notes, stats
        )
        return {"report": str(report_path)}

    _stage(
        state,
        "report",
        imported.job_dir / "report.json",
        write_report,
    )
    state.mark_stage("done")
    return output_dir


def process_file(source: str | Path, config: PipelineConfig, move: bool = True) -> Path:
    return _process_job(ingest(source, config, move=move), config)


def resume_pending(config: PipelineConfig) -> None:
    from app.ingest import IngestedFile

    workdir = Path(config.paths.workdir)
    if not workdir.is_dir():
        return
    for job_dir in sorted(workdir.iterdir()):
        state_path = job_dir / "state.json"
        if not job_dir.is_dir() or not state_path.is_file():
            continue
        state = JobState.load(state_path)
        if state.is_complete("done"):
            continue
        video_files = sorted(job_dir.glob("source.*"))
        if not video_files:
            print(f"Cannot resume {state.job_id}: source video is missing", flush=True)
            continue
        video_path = video_files[0]
        imported = IngestedFile(
            state.job_id,
            video_path,
            job_dir,
            state.original_name or video_path.name,
        )
        try:
            _process_job(imported, config)
        except Exception as exc:
            print(f"Failed to resume {state.job_id}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local presentation-editing pipeline")
    parser.add_argument("--config", help="Path to pipeline YAML configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("watch", help="Poll the configured inbox for video files")
    process_parser = commands.add_parser("process", help="Process one video file")
    process_parser.add_argument("file", type=Path)
    process_parser.add_argument(
        "--copy", "--keep-source", dest="keep_source", action="store_true",
        help="Copy the video into the workdir instead of moving the original",
    )
    serve_parser = commands.add_parser("serve", help="Run the local web UI")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "watch":
        from app.watcher import watch

        watch(config)
    elif args.command == "serve":
        from app.web import create_app

        create_app(config).run(
            host=args.host, port=args.port, debug=False, use_reloader=False,
        )
    else:
        output_dir = process_file(args.file, config, move=not args.keep_source)
        print(f"Processed output: {output_dir}")


if __name__ == "__main__":
    main()
