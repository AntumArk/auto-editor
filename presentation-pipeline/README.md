# Local presentation-editing pipeline

Record a slide presentation, drop a copy into a NAS inbox, and process it locally
into a DaVinci Resolve timeline, subtitles, and a review report. The pipeline
proposes cuts during silence on unchanged slides; an optional local Ollama model
reviews those candidates. Review the result before publishing.

```text
ingest → transcribe → slide detection → cut candidates → LLM review
       → export → subtitles → report
```

This Python service is separate from the existing Nim `auto-editor` CLI: it calls
the `auto-editor` executable, without modifying the Nim core. The Docker image
currently installs the packaged CLI from `requirements.txt`, not this repository's
Nim build.

**Integration status:** the parallel integration PR (task `c825d575`) fixes stage
wiring and original-name output naming. The output contract below describes that
integrated version; this checkout still has wiring mismatches and may produce
`source.*` exports or fail at subtitles/report generation. Merge the integration
fixes before expecting a successful end-to-end run. Do not assume a pipeline CI
workflow is present until that PR lands.

See the [FAQ](docs/FAQ.md), [testing plan](docs/TESTING.md), and
[test report template](docs/TEST_REPORT_TEMPLATE.md).
Commands below have been checked against source, not run on a GPU/NAS/Resolve
installation. OS packages, mount options, and Resolve UI details are approximate
and must be adapted to your environment.

## 1. Requirements and GPU setup

- Linux host with a working NVIDIA driver (`nvidia-smi` succeeds).
- NVIDIA GPU; the intended test target is an RTX 3070 with 8 GB VRAM, not a
  guarantee that both default models fit. See [memory advice](docs/FAQ.md#cuda-out-of-memory-on-8-gb).
- Docker Engine and the Docker Compose **plugin** (`docker compose version`).
  Follow the [Docker installation guide](https://docs.docker.com/engine/install/)
  for your distribution. Docker access is effectively root access; grant it only
  to trusted users.
- NVIDIA Container Toolkit and a driver compatible with the CUDA 12.4.1 image.
  A host CUDA development toolkit is not required.
- Writable inbox, output, and working directories; space for source videos,
  models, artifacts, and any rendered previews. Internet is needed initially for
  packages/images and model downloads.

Install the NVIDIA driver using your distribution's instructions, then follow
the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to add its package repository. For Debian/Ubuntu, after adding that repository,
the approximate remaining commands are:

```sh
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Do not proceed until the last command sees your GPU. Both Compose services
request NVIDIA GPUs.

## 2. Mount the NAS and choose storage

Example mount point: `/mnt/nas`. Create it first and choose **one** appropriate
`/etc/fstab` entry, replacing the server/share names:

```fstab
# NFS example; requires the distribution's NFS client package
nas.example:/exports/presentations /mnt/nas nfs defaults,_netdev 0 0
# SMB/CIFS alternative; requires cifs-utils
//nas.example/presentations /mnt/nas cifs credentials=/etc/samba/pipeline.credentials,vers=3.0,uid=1000,gid=1000,file_mode=0660,dir_mode=0770,_netdev 0 0
```

For SMB, create the credentials file outside the repository with your NAS
`username=` and `****** entries; restrict it with
`sudo chmod 600 /etc/samba/pipeline.credentials`. Never commit credentials.
Mount with `sudo mount /mnt/nas` and verify the actual mounted filesystem before
starting Docker, including after a reboot. Otherwise Docker may write to the
local mount-point directory instead of the NAS.

NFS uses server-side ownership/UID/GID mappings; root squashing may prevent the
container's root user from writing. SMB `uid`/`gid` values must match your chosen
local owner, and server ACLs still apply. The image currently runs as root;
verify create, read, rename, and delete permissions from inside the container.
Do not solve permission problems with world-writable shares.

```sh
mkdir -p /mnt/nas/presentation-inbox /mnt/nas/presentation-output
# Local scratch storage is usually faster than keeping work artifacts on NAS.
sudo mkdir -p /var/lib/presentation-pipeline/work
export PIPELINE_INBOX=/mnt/nas/presentation-inbox
export PIPELINE_OUTPUT=/mnt/nas/presentation-output
export PIPELINE_WORKDIR=/var/lib/presentation-pipeline/work
```

These are **host bind-mount variables** used by Compose:

| Compose variable | Host default | Container destination |
| --- | --- | --- |
| `PIPELINE_INBOX` | `/data/inbox` | `/data/inbox` |
| `PIPELINE_OUTPUT` | `/data/output` | `/data/output` |
| `PIPELINE_WORKDIR` | `/data/work` | `/data/work` |

Persist them in your shell setup or a local `.env` beside
`docker-compose.yml` (do not commit machine-specific values). They are not the
same as application configuration overrides such as `PIPELINE__PATHS__INBOX`.
Normally keep the container paths unchanged.

## 3. Quick start

Run Compose commands from the pipeline directory:

```sh
cd /home/runner/work/auto-editor/auto-editor/presentation-pipeline
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose up -d pipeline
docker compose logs -f pipeline
```

Whisper downloads its model on first use. Allow time and disk space for both
model downloads. Start with a short clip; large-v3 plus a 7B LLM can exhaust
8 GB VRAM even though the stages run sequentially.

Copy a recording using a temporary extension, then rename it when copying is
complete (replace the source path):

```sh
cp /path/to/presentation.mp4 "$PIPELINE_INBOX/presentation.mp4.part"
mv "$PIPELINE_INBOX/presentation.mp4.part" "$PIPELINE_INBOX/presentation.mp4"
```

The watcher only checks that size stays unchanged for consecutive polls; it
cannot prove a network copy is complete. **Ingest moves the inbox file into
the work directory. Keep an original backup outside the inbox.**
Results appear at `$PIPELINE_OUTPUT/<job_id>/` on the host, corresponding to
`/data/output/<job_id>/` in the container. The job ID combines a sanitized source
stem and the first 12 characters of its content SHA-256.

### Process one file manually

Stop the watcher first so it cannot race the manual job. The input must be
visible through a container mount; this command also **moves** it:

```sh
docker compose stop pipeline
docker compose run --rm pipeline python3 -m app.main process /data/inbox/presentation.mp4
docker compose up -d pipeline
```

To run without LLM review (deterministic cuts and non-LLM report fallback):

```sh
docker compose stop pipeline
docker compose run --rm -e PIPELINE__LLM__ENABLED=false pipeline \
  python3 -m app.main process /data/inbox/presentation.mp4
```

That override affects only this one-off container. For the watcher, add
`PIPELINE__LLM__ENABLED: "false"` to the `pipeline.environment` mapping in Compose
and recreate with `docker compose up -d pipeline`. Exporting this variable in the
host shell or adding it to `.env` alone does **not** pass it into the service.
Compose still starts its `ollama` dependency unless `--no-deps` is used.

## 4. Configuration reference

Defaults below are from [`configs/config.yaml`](configs/config.yaml).
Application overrides use `PIPELINE__SECTION__KEY`, parsed as YAML values, e.g.
`PIPELINE__CUTS__MIN_SILENCE_S=3.0`. Pass them with `docker compose run -e ...`
or explicitly add them to the Compose service environment.

| Key | Default | Meaning |
| --- | --- | --- |
| `paths.inbox` | `/data/inbox` | Container directory polled for new videos. |
| `paths.output` | `/data/output` | Root for per-job deliverables. |
| `paths.workdir` | `/data/work` | Source videos, state, and cached stage artifacts. |
| `transcription.model` | `large-v3` | faster-whisper model name or supported model path. |
| `transcription.language` | `lt` | Whisper language hint; YAML `null` enables automatic detection. |
| `transcription.task` | `transcribe` | Whisper task; `transcribe` preserves spoken language, `translate` translates into English. |
| `transcription.vad` | `true` | Enable voice activity filtering during transcription. |
| `transcription.word_timestamps` | `true` | Request word timings for subtitle remapping and speech protection. |
| `visual.sample_fps` | `3` | Frames sampled per second for slide-change analysis. |
| `visual.change_threshold` | `0.04` | Normalized mean frame-difference threshold; lower is more sensitive. |
| `visual.downscale_width` | `320` | Maximum analysis frame width in pixels. |
| `cuts.min_silence_s` | `2.0` | Minimum candidate silence after speech/slide protection. |
| `cuts.target_pause_s` | `0.6` | Total pause retained when shortening a candidate, split between its edges. |
| `cuts.speech_margin_s` | `0.25` | Protected seconds before and after speech. |
| `cuts.settle_after_slide_change_s` | `1.0` | Protected seconds after a detected slide change. |
| `cuts.full_remove_min_s` | `8.0` | Minimum candidate length for full removal with explicit LLM approval. |
| `llm.enabled` | `true` | Enable local LLM review and report summarization. |
| `llm.provider` | `ollama` | Provider label; current implementation calls Ollama, not a provider-switching backend. |
| `llm.url` | `http://ollama:11434` | Ollama endpoint; `ollama` is the Compose service hostname. |
| `llm.model` | `qwen2.5:7b-instruct` | Ollama model tag; pull it before use. |
| `llm.timeout_s` | `120` | Candidate-review request timeout in seconds; report summarization currently uses its own 30-second timeout. |
| `report.language` | `lt` | Requested LLM summary language; deterministic fallback is not guaranteed to be localized. |
| `report.min_chapter_gap_s` | `10` | Desired chapter spacing; YouTube's 10-second minimum still applies. |
| `watcher.poll_interval_s` | `5` | Seconds between size checks and inbox scans. |
| `watcher.stable_checks` | `2` | Consecutive unchanged-size checks before ingest. |
| `watcher.video_extensions` | `[.mp4, .mov, .mkv, .avi, .webm, .m4v]` | Allowed filename suffixes, matched case-insensitively. |

Editing the host YAML does not update an existing image: the Dockerfile copies it
at build time. Rebuild/recreate, or bind-mount your own YAML and use
`python3 -m app.main --config /path/in/container/config.yaml watch`
(`--config` precedes the subcommand). Completed stages remain cached even after
configuration changes; [reset the job](#reset-a-job) to apply changes throughout.
Unknown keys are ignored by the dataclass loader: do not assume undocumented
environment variables such as a compute-type setting take effect.

## 5. Outputs and Resolve/YouTube

With the integration fixes, `/data/output/<job_id>/` contains files named after
the **original video stem**, e.g. `lecture.mp4` produces:

| File | Use |
| --- | --- |
| `<name>.fcpxml` | Cut timeline for DaVinci Resolve, referencing the ingested source media. |
| `<name>.cutlist.json` | Auditable removed/kept source-time ranges and original/edited durations. |
| `<name>.source.srt` | Subtitles on the original, uncut timebase. |
| `<name>.edited.srt` | Subtitles remapped onto the edited timeline. |
| `<name>.report.md` | Summary, review notes, statistics, and YouTube chapter text. |
| `<name>.preview.mp4` (optional) | Rendered preview, only when preview export is supported and enabled. |

Preview is not enabled by default. The exporter has a programmatic
`cfg.export.preview` hook, but the current YAML/dataclass config has no `export`
section; `PIPELINE__EXPORT__PREVIEW=true` does not enable it through this CLI.
Do not expect a preview from the stock Compose setup.

In Resolve, choose **File → Import → Timeline**, select the `.fcpxml`, and verify
the source/timeline frame rate before editing. Menu wording varies by Resolve
version. Container paths such as `/data/work/<job_id>/source.mp4` may not exist
on your Resolve workstation: relink media to the corresponding host/NAS file,
or copy the source and relink it there. Preserve that source while using the
timeline; FCPXML is not a self-contained video.

Import `<name>.edited.srt` as a subtitle track for the edited timeline, and
spot-check sync immediately after cuts. For YouTube, upload the rendered edited
video, then upload that SRT as timed subtitles in YouTube Studio. Use
`.source.srt` only with the uncut recording. Additional edits in Resolve require
retiming subtitles and chapters.

Copy the report's **YouTube Chapters** timestamp/title block into the video
description. Chapters use edited time and must start at `00:00`, contain at least
three entries, and each last at least 10 seconds, including the final chapter.
Short videos may intentionally have no chapter block.

## 6. Maintenance and testing

To update, pull reviewed repository changes using your normal Git workflow, then:

```sh
docker compose build --pull
docker compose pull ollama
docker compose up -d
```

To stop temporarily: `docker compose stop`. To remove containers and the network:
`docker compose down`. Both preserve bind-mounted jobs and named model caches.
Avoid `docker compose down -v` unless intentionally deleting the named model
volumes; it does not erase bind-mounted inbox/output/work directories.

### Reset a job

Restarts resume pending jobs from cached stages; completed jobs are skipped.
The same source stem/content maps to the same job ID. To force a fresh run:

1. Stop the watcher (`docker compose stop pipeline`) and any manual processing.
2. Find the exact job ID in the work directory/state. **Back up its source video
   outside the work directory first**; ingest moved your inbox copy there.
3. Delete only that job's `/data/work/<job_id>` folder (on the host,
   `$PIPELINE_WORKDIR/<job_id>`). This removes the source, state, and all cached
   steps. Do not delete the entire work root.
4. Archive/remove that job's matching output folder to avoid stale deliverables.
5. Copy the backed-up video into the inbox again and start the watcher.

Do not delete only output files: a completed state will prevent regeneration.

For unit tests without GPU/network execution, follow
[docs/TESTING.md](docs/TESTING.md); dependencies must already be installed or
downloaded first. It also covers Docker/GPU smoke tests and short/full recordings.
Use [docs/TEST_REPORT_TEMPLATE.md](docs/TEST_REPORT_TEMPLATE.md) to record results.
Some older testing examples use generic `cutlist.json`/`report.md`; use the
original-stem output names above for the integrated pipeline.
