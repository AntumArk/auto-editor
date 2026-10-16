from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from app.config import PipelineConfig


def wait_until_stable(
    path: str | Path,
    stable_checks: int = 2,
    poll_interval_s: float = 5,
) -> bool:
    """Wait until a file's size is unchanged for the requested consecutive checks."""
    file_path = Path(path)
    previous_size = -1
    unchanged_checks = 0
    while file_path.is_file():
        size = file_path.stat().st_size
        if size == previous_size:
            unchanged_checks += 1
            if unchanged_checks >= max(1, stable_checks):
                return True
        else:
            unchanged_checks = 0
            previous_size = size
        time.sleep(max(0, poll_interval_s))
    return False


def iter_ready_files(config: PipelineConfig) -> Iterator[Path]:
    inbox = Path(config.paths.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    extensions = {extension.lower() for extension in config.watcher.video_extensions}
    while True:
        for path in sorted(inbox.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                if wait_until_stable(
                    path,
                    config.watcher.stable_checks,
                    config.watcher.poll_interval_s,
                ):
                    yield path
        time.sleep(max(0, config.watcher.poll_interval_s))


def watch(config: PipelineConfig) -> None:
    from app.main import process_file
    from app.main import resume_pending

    resume_pending(config)
    for path in iter_ready_files(config):
        try:
            process_file(path, config)
        except Exception as exc:
            print(f"Failed to process {path}: {exc}", flush=True)
