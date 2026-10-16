from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STAGES = (
    "ingested",
    "transcribed",
    "visual",
    "candidates",
    "llm",
    "cut",
    "subtitles",
    "report",
    "done",
)


@dataclass
class JobState:
    job_id: str
    job_dir: Path
    source: str
    original_name: str | None = None
    stage: str = "ingested"
    artifacts: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error: str | None = None

    @property
    def path(self) -> Path:
        return self.job_dir / "state.json"

    @classmethod
    def create(
        cls,
        job_id: str,
        job_dir: str | Path,
        source: str,
        original_name: str | None = None,
    ) -> JobState:
        state = cls(job_id, Path(job_dir), source, original_name)
        state.job_dir.mkdir(parents=True, exist_ok=True)
        if state.path.exists():
            return cls.load(state.path)
        state.save()
        return state

    @classmethod
    def load(cls, path: str | Path) -> JobState:
        state_path = Path(path)
        with state_path.open(encoding="utf-8") as source_file:
            data = json.load(source_file)
        if data.get("stage") not in STAGES:
            raise ValueError(f"Invalid job stage in {state_path}")
        return cls(
            job_id=data["job_id"],
            job_dir=state_path.parent,
            source=data["source"],
            original_name=data.get("original_name"),
            stage=data["stage"],
            artifacts=data.get("artifacts", {}),
            status=data.get("status", "pending"),
            error=data.get("error"),
        )

    def save(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": self.job_id,
            "source": self.source,
            "original_name": self.original_name,
            "stage": self.stage,
            "artifacts": self.artifacts,
            "status": self.status,
            "error": self.error,
        }
        descriptor, temporary_name = tempfile.mkstemp(dir=self.job_dir, prefix=".state-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
                json.dump(payload, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def is_complete(self, stage: str) -> bool:
        if stage not in STAGES:
            raise ValueError(f"Unknown job stage: {stage}")
        return STAGES.index(self.stage) >= STAGES.index(stage)

    def report_progress(
        self, stage: str, message: str,
        completed: float | None = None, total: float | None = None,
    ) -> None:
        previous = self.artifacts.get("progress", {})
        now = time.time()
        self.artifacts["progress"] = {
            "stage": stage,
            "message": message,
            "completed": completed,
            "total": total,
            "started_at": previous["started_at"] if previous.get("stage") == stage else now,
            "updated_at": now,
        }
        self.save()

    def mark_stage(self, stage: str, artifact: Any = None) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown job stage: {stage}")
        current_index = STAGES.index(self.stage)
        next_index = STAGES.index(stage)
        if next_index <= current_index:
            return
        if next_index != current_index + 1:
            raise ValueError(f"Cannot advance job from {self.stage} to {stage}")
        if artifact is not None:
            self.artifacts[stage] = artifact
        self.artifacts.pop("progress", None)
        self.stage = stage
        self.save()
