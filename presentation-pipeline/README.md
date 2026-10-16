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
installs the packaged CLI from `requirements.txt`, not this repository's Nim build.
Each job has a JSON state file and cached stage artifacts under
`/data/work/<job_id>`, so resubmitted jobs can reuse completed stages.

See the [FAQ](docs/FAQ.md), [testing plan](docs/TESTING.md), and
[test report template](docs/TEST_REPORT_TEMPLATE.md).
Commands below have been checked against source, not run on a GPU/NAS/Resolve
installation. OS packages, mount options, and Resolve UI details are approximate
and must be adapted to your environment.

## Low-end / CPU-only quick start

No NVIDIA GPU, driver, Container Toolkit, or Ollama is required for this setup.
CPU transcription is slower; start with a short clip and the `small` model
(use `tiny` or `base` on especially limited machines). Initial downloads require
internet access.

### Docker

Run from `presentation-pipeline/`:

```sh
docker compose -f docker-compose.cpu.yml up --build -d
docker compose -f docker-compose.cpu.yml logs -f pipeline
```

Open <http://127.0.0.1:8000>. This is a **standalone** Compose file, not an
override to merge with `docker-compose.yml`: it starts only the pipeline, without
GPU reservations or an Ollama dependency. It reuses the existing CUDA-based
Dockerfile; that shared image is larger than a dedicated CPU image but can run
on CPU-only Docker without the NVIDIA Container Toolkit.

Defaults are LLM disabled, transcription device `cpu`, model `small`, and language
`null` (automatic detection). Bind mounts default to `./data/inbox`,
`./data/output`, and `./data/work`; optionally set `PIPELINE_INBOX`,
`PIPELINE_OUTPUT`, and `PIPELINE_WORKDIR` to writable host directories before
starting. Container paths remain `/data/inbox`, `/data/output`, and `/data/work`.
The named `whisper-cache` volume persists downloaded models.

Stop with `docker compose -f docker-compose.cpu.yml down`; omit `-v` to retain
the model cache.

### Native Python

Install Python **3.10+**, its venv support, and FFmpeg (including `ffprobe`) using
your OS package manager; ensure both FFmpeg commands are on `PATH`. From
`presentation-pipeline/`:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
export PIPELINE__PATHS__INBOX="$PWD/data/inbox"
export PIPELINE__PATHS__OUTPUT="$PWD/data/output"
export PIPELINE__PATHS__WORKDIR="$PWD/data/work"
export PIPELINE__LLM__ENABLED=false
export PIPELINE__DIARIZATION__ENABLED=false
export PIPELINE__TRANSCRIPTION__DEVICE=cpu
export PIPELINE__TRANSCRIPTION__MODEL=small
export PIPELINE__TRANSCRIPTION__LANGUAGE=null
python3 -m app.main serve
```

`serve`/ingest create the local directories as needed; their parent locations
must be writable. Requirements install the packaged `auto-editor` CLI; keep the
venv active so it is on `PATH`. Open <http://127.0.0.1:8000>.

### Web UI

Upload a video (maximum **16 GiB**) or submit a file path **inside the configured
inbox only**. For Docker, a path means `/data/inbox/presentation.mp4`, not an
arbitrary host path. The server keeps the source file in the inbox rather than
moving it. Choose a Whisper model (`tiny`, `base`, `small`, `medium`, or
`large-v3`), language (leave blank for automatic detection, or enter a code such
as `en`/`lt`), and whether
to use LLM review. Enabling LLM requires a reachable, configured Ollama service
with its model already pulled; the CPU Compose file does not provide one.

One worker processes jobs sequentially. The job list shows status and the last
completed stage; errors appear on the job detail page. **Refresh manually** for
updates. Completed jobs expose output download links and a rendered Markdown
report with raw HTML escaped. Settings apply only to newly submitted jobs;
completed stages are cached by source name/content, so changing settings does
not recompute an existing job.
See [Reset a job](#reset-a-job) when you need a fresh run.

Run **one server instance per work directory**, and do not run a watcher or
manual processor against the same work directory concurrently. Queued jobs are
not automatically restarted after a server restart: resubmit them to reuse
persisted stages.

Native `serve` binds to loopback by default. The CPU container listens on
`0.0.0.0` internally, but publishes only `127.0.0.1:8000:8000` on the host.
There is **no authentication or TLS**. Use `--host 0.0.0.0` (and change Docker
port publishing, if needed) only on a trusted network; never expose it directly
to the internet.

## 1. Requirements and GPU setup (optional)

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

For the GPU deployment, do not proceed until the last command sees your GPU.
Both services in the default `docker-compose.yml` request NVIDIA GPUs; the
CPU-only quick start above does not.

## 2. Mount the NAS and choose storage

Example mount point: `/mnt/nas`. Create it first and choose **one** appropriate
`/etc/fstab` entry, replacing the server/share names:

```fstab
# NFS example; requires the distribution's NFS client package
nas.example:/exports/presentations /mnt/nas nfs defaults,_netdev 0 0
# SMB/CIFS alternative; requires cifs-utils
//nas.example/presentations /mnt/nas cifs credentials=/etc/samba/pipeline.credentials,vers=3.0,uid=1000,gid=1000,file_mode=0660,dir_mode=0770,_netdev 0 0
```

For SMB, create the credentials file outside the repository with your NAS login
fields in the format required by `mount.cifs`; restrict it with
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

These are **host bind-mount variables** used by the default GPU Compose file
(the CPU file instead defaults to `./data/inbox`, `./data/output`, `./data/work`):

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

Speaker diarization is enabled by default and runs **locally**, without external
transcription/diarization APIs. Before starting, accept the model terms and
export `HF_TOKEN` as described [below](#optional-alignment-and-speaker-labels).
Model files download on first use; with cached models no external inference
service is needed. After caching all models, set `HF_HUB_OFFLINE=1` to prevent
Hugging Face network access (forward it explicitly in Docker).
To skip this step, set `diarization.enabled: false`.
The CPU-only quick start explicitly disables diarization.

Run Compose commands from the pipeline directory of your checkout:

```sh
cd auto-editor/presentation-pipeline
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

To preserve the input file instead of moving it, use `process --keep-source`
(`--copy` is an alias), for example in the native environment above:

```sh
python3 -m app.main process /path/to/presentation.mp4 --keep-source
```

The CLI can process a local path outside the inbox; the Web UI deliberately
restricts submitted paths to the configured inbox. Server jobs always keep
their inbox source.

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
| `transcription.device` | `auto` | `auto` and `cuda` try CUDA, then fall back on errors (including OOM); `cpu` forces CPU. |
| `transcription.compute_type` | `auto` | `auto` selects CUDA `int8_float16` or CPU `int8`. CUDA retry order is configured precision → `int8_float16` → CPU `int8`, without duplicate attempts. An explicit CPU precision is honored. |
| `transcription.language` | `lt` | Whisper language hint; YAML `null` enables automatic detection. |
| `transcription.task` | `transcribe` | Whisper task; `transcribe` preserves spoken language, `translate` translates into English. |
| `transcription.vad` | `true` | Enable voice activity filtering during transcription. |
| `transcription.vad_filter` | `null` | Optional override for `vad`; `null` retains the existing `vad` setting. |
| `transcription.vad_parameters` | `null` | Optional faster-whisper VAD mapping, e.g. `{min_silence_duration_ms: 500}`. |
| `transcription.word_timestamps` | `true` | Request word timings for subtitle remapping and speech protection. |
| `transcription.beam_size` | `5` | Decoding beam width. |
| `transcription.condition_on_previous_text` | `false` | Avoid carrying previous text into decoding, reducing repetition/loops. |
| `transcription.initial_prompt` | `null` | Optional glossary of names/technical terms in the talk. |
| `transcription.align` | `false` | Refine word timestamps with optional WhisperX after releasing Whisper. Warn and retain original timings if unavailable or unsuccessful. |
| `transcription.align_model` | `null` | Optional language-compatible alignment model ID/path; needed for languages without a WhisperX default, such as Lithuanian. |
| `diarization.enabled` | `true` | Run local `pyannote/speaker-diarization-3.1` after Whisper/alignment. An unavailable backend/token fails with an error; set `false` to opt out. |
| `diarization.hf_token` | `HF_TOKEN` | **Environment variable name**, never the token value. |
| `diarization.min_speakers` / `max_speakers` | `null` | Optional positive speaker-count bounds; otherwise let pyannote estimate. |
| `diarization.device` | `cpu` | `cpu` or `cuda`; CPU saves VRAM at the cost of speed. |
| `diarization.fill_nearest` | `false` | Assign the nearest speaker turn when there is no time overlap; otherwise leave unlabeled. |
| `diarization.speaker_prefix` | `false` | Prefix source and edited SRT cues with `[SPEAKER_00] ` when labeled. |
| `visual.sample_fps` | `3` | Frames sampled per second for slide-change analysis. |
| `visual.change_threshold` | `0.04` | Normalized mean frame-difference threshold; lower is more sensitive. |
| `visual.downscale_width` | `320` | Maximum analysis frame width in pixels. |
| `cuts.min_silence_s` | `2.0` | Minimum candidate silence after speech/slide protection. |
| `cuts.target_pause_s` | `0.6` | Total pause retained when shortening a candidate, split between its edges. |
| `cuts.speech_margin_s` | `0.25` | Protected seconds before and after speech. |
| `cuts.settle_after_slide_change_s` | `1.0` | Protected seconds after a detected slide change. |
| `cuts.full_remove_min_s` | `8.0` | Minimum candidate length for full removal with explicit LLM approval. |
| `llm.enabled` | `true` | Enable local LLM review and report summarization. |
| `llm.provider` | `ollama` | Provider label; current implementation calls Ollama only. |
| `llm.url` | `http://ollama:11434` | Ollama endpoint; `ollama` is the Compose service hostname. |
| `llm.model` | `qwen2.5:7b-instruct` | Ollama model tag; pull it before use. |
| `llm.timeout_s` | `120` | Candidate-review request timeout in seconds; report summarization uses its own 30-second timeout. |
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
Unknown keys are ignored by the dataclass loader; use the documented settings.
For example, `PIPELINE__TRANSCRIPTION__DEVICE=cpu` forces CPU transcription and
`PIPELINE__TRANSCRIPTION__COMPUTE_TYPE=int8` explicitly selects integer precision.

### Quality on an 8 GB GPU

Keep `large-v3` for Lithuanian: `large-v3-turbo` is faster but generally worse
for Lithuanian transcription; compare on your own audio before changing models.
The default CUDA `int8_float16` mode lowers VRAM use while retaining the large
model. No fixed VRAM budget is guaranteed: clip length, software, and other
processes matter. CPU `int8` is the slow, GPU-free fallback.
Whisper is released and garbage collected after transcription; the optional
PyTorch CUDA cache is cleared. Alignment and diarization run sequentially and
release their models before LLM review. Stop/unload an already resident Ollama
model before starting transcription on 8 GB; freeing Whisper cannot evict
Ollama's model. Use a smaller LLM or disable LLM review if necessary.

### Optional alignment and speaker labels

Diarization runs locally by default; alignment remains disabled. Native users
enabling diarization need the separate heavy dependency set in their venv:

```sh
python3 -m pip install -r requirements.txt -r requirements-extra.txt
```

Docker installs these extras by default. To rebuild explicitly:

```sh
docker compose build --build-arg INSTALL_TRANSCRIPTION_EXTRAS=true pipeline
```

The extras include PyTorch and substantially enlarge installation/model caches.
To omit them, build with `INSTALL_TRANSCRIPTION_EXTRAS=false` and disable
diarization with `PIPELINE__DIARIZATION__ENABLED=false`.
Check the NVIDIA driver's compatibility with the installed PyTorch CUDA wheels,
not only the base image. First use downloads models and requires network access.
The optional pins use pyannote.audio 3.x for the requested 3.1 checkpoint; do not
independently upgrade it to 4.x, which introduces different model prerequisites.

Enable `transcription.align: true` to refine timings, not the recognized text.
For `lt`, set `transcription.align_model` to a vetted Lithuanian-compatible
wav2vec2/CTC alignment model; no unverified model is selected automatically.
Missing WhisperX, unsupported languages, or alignment errors produce a warning
and keep the original Whisper timestamps.

For diarization, accept the terms for both
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
Provide a Hugging Face read token through the environment (never YAML/source):

```sh
read -rsp 'Hugging Face read token: ' HF_TOKEN; echo
export HF_TOKEN
export PIPELINE__DIARIZATION__ENABLED=true
export PIPELINE__DIARIZATION__SPEAKER_PREFIX=true
python3 -m app.main process /path/to/presentation.mp4 --keep-source
unset HF_TOKEN
```

The default GPU Compose service forwards `HF_TOKEN` from the host environment.
For a one-off Docker run, stop the watcher first and pass the environment
explicitly with `docker compose run --rm -e HF_TOKEN
-e PIPELINE__DIARIZATION__ENABLED=true -e PIPELINE__DIARIZATION__SPEAKER_PREFIX=true
pipeline python3 -m app.main process /data/inbox/presentation.mp4 --keep-source`.
For the watcher, explicitly forward other configuration overrides in the Compose
service's `environment` mapping; host exports alone are not forwarded.

`transcript.json` stores speaker labels on segments and words, independently
assigned to the speaker with the greatest **total** overlapping time.
Speaker labels survive subtitle retiming; prefixing is optional and does not
modify transcript text. Labels are anonymous, file-local, and not guaranteed
to identify the same person across recordings. This labels speakers; it does
not filter out other speakers. Overlap/ties and nearest-fill are heuristics,
so review speaker changes manually. Compare settings on the same 3-minute
clip using the [testing plan](docs/TESTING.md), resetting cached jobs between runs.

## 5. Outputs and Resolve/YouTube

`/data/output/<job_id>/` contains files named after the **original video stem**
(not the working copy's `source` name), e.g. `lecture.mp4` produces:

| File | Use |
| --- | --- |
| `<name>.fcpxml` | Cut timeline for DaVinci Resolve, referencing the ingested source media. |
| `<name>.cutlist.json` | The single audit file: removed/kept source-time ranges and original/edited durations. |
| `<name>.source.srt` | Subtitles on the original, uncut timebase. |
| `<name>.edited.srt` | Subtitles remapped onto the edited timeline. |
| `<name>.report.md` | Summary, review notes, statistics, and YouTube chapter text. |
| `<name>.preview.mp4` (optional) | Rendered preview, only when preview export is enabled. |

Stage JSON files (`transcript.json`, `visual.json`, `candidates.json`,
`decisions.json`, …) remain cached separately in the job's work directory.

Preview is not enabled by default. The exporter has a programmatic
`cfg.export.preview` hook, but the current YAML/dataclass config has no `export`
section; `PIPELINE__EXPORT__PREVIEW=true` does not enable it through this CLI.

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

The watcher resumes pending jobs on restart; the Web UI requires resubmission
after a restart. Completed jobs are skipped, and resubmissions reuse cached stages.
The same source stem/content maps to the same job ID. To force a fresh run:

1. Stop the server/watcher (for CPU Docker,
   `docker compose -f docker-compose.cpu.yml stop pipeline`) and any manual processing.
2. Find the exact job ID in the work directory/state. **Back up its source video
   outside the work directory first**; ingest moved your inbox copy there.
3. Delete only that job's `/data/work/<job_id>` folder (on the host,
   `$PIPELINE_WORKDIR/<job_id>`). This removes the source, state, and all cached
   steps. Do not delete the entire work root.
4. Archive/remove that job's matching output folder to avoid stale deliverables.
5. Copy the backed-up video into the inbox again and start the watcher, or
   restart the server and resubmit it through the Web UI.

Do not delete only output files: a completed state will prevent regeneration.

### Tests

For CPU-only tests without GPU models or network services:

```sh
cd presentation-pipeline
pip install -r requirements-test.txt
pytest
```

The same tests run in CI on changes under `presentation-pipeline/`. For
Docker/GPU smoke tests and short/full recordings, follow
[docs/TESTING.md](docs/TESTING.md) and record results with
[docs/TEST_REPORT_TEMPLATE.md](docs/TEST_REPORT_TEMPLATE.md). Where older testing
examples use generic `cutlist.json`/`report.md`, use the original-stem output
names above.
