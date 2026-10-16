from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import PipelineConfig


@dataclass
class IngestedFile:
    job_id: str
    video_path: Path
    job_dir: Path
    original_name: str


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest(
    source: str | Path,
    config: PipelineConfig,
    move: bool = True,
    settings_key: str | None = None,
    original_name: str | None = None,
) -> IngestedFile:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    name = original_name or source_path.name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem).strip(".-_") or "video"
    job_id = f"{slug}-{_content_hash(source_path)[:12]}"
    if settings_key is not None:
        if not re.fullmatch(r"[a-f0-9]{12}", settings_key):
            raise ValueError("Invalid settings key")
        job_id += f"-{settings_key}"
    job_dir = Path(config.paths.workdir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    destination = job_dir / f"source{source_path.suffix.lower()}"

    if not destination.exists():
        if move:
            shutil.move(str(source_path), destination)
        else:
            shutil.copy2(source_path, destination)
    elif move and source_path != destination.resolve():
        source_path.unlink()
    return IngestedFile(job_id, destination, job_dir, name)
