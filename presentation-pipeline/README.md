# Local presentation-editing pipeline

This separate Python service watches a NAS-mounted inbox and orchestrates local
transcription, slide analysis, deterministic cut candidates, optional Ollama
review, auto-editor export, subtitle remapping, and a Markdown review report.
It does not modify or replace the existing Nim `auto-editor` CLI. Each job has
a JSON state file and cached stage artifacts under `/data/work/<job_id>`, so a
restart resumes from the last completed stage.

Testing instructions: [docs/TESTING.md](docs/TESTING.md). Report results with
[docs/TEST_REPORT_TEMPLATE.md](docs/TEST_REPORT_TEMPLATE.md).

## Linux and NVIDIA setup

Install Docker Engine and the Docker Compose plugin. For GPU access, install the
NVIDIA driver and NVIDIA Container Toolkit for your Linux distribution, then
configure Docker with `sudo nvidia-ctk runtime configure --runtime=docker` and
restart Docker.

Set the host-side NAS mount paths, or use the `/data/...` defaults:

```sh
export PIPELINE_INBOX=/mnt/nas/presentation-inbox
export PIPELINE_OUTPUT=/mnt/nas/presentation-output
export PIPELINE_WORKDIR=/mnt/nas/presentation-work
docker compose -f presentation-pipeline/docker-compose.yml up --build -d
docker compose -f presentation-pipeline/docker-compose.yml exec ollama \
  ollama pull qwen2.5:7b-instruct
```

To process one file instead of watching the inbox, run
`python3 -m app.main process /data/inbox/presentation.mp4` in the pipeline
container. Configuration is in `configs/config.yaml`; any setting can be
overridden with a `PIPELINE__SECTION__KEY` environment variable, for example
`PIPELINE__CUTS__MIN_SILENCE_S=3.0`.

## Outputs

Each job writes its Resolve timeline/export and the following review assets to
`/data/output/<job_id>/`:

- `<name>.source.srt` and `<name>.edited.srt`
- `<name>.report.md`
- `<name>.fcpxml` (Resolve timeline)
- `<name>.cutlist.json` (the single output audit file)
- `<name>.preview.mp4` (when preview export is enabled)

`<name>` is the original input file stem, not the working copy's `source` name.
Stage JSON files remain cached separately in the job's work directory.

For CPU-only tests without GPU models or network services, install
`requirements-test.txt` and run `pytest` from `presentation-pipeline/`.

The pipeline modules for transcription, visual analysis, candidate generation,
LLM review, cut-list export, subtitles, and reporting are loaded only when
their stage is run.
