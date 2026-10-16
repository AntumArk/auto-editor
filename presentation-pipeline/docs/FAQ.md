# Presentation pipeline FAQ

See the [setup guide](../README.md) for prerequisites and configuration defaults.
Commands here assume you are in the `presentation-pipeline/` directory of your
checkout. Hardware, mount, and Resolve commands are approximate and have not been
verified on your machine. Replace example filenames and job IDs before running them.

## GPU not detected in the container

First run `nvidia-smi` on the host, then:

```sh
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If the host fails, fix the NVIDIA driver first. If only the container fails,
install NVIDIA Container Toolkit for your distribution, run
`sudo nvidia-ctk runtime configure --runtime=docker`, and restart Docker with
`sudo systemctl restart docker`. Check that the driver supports the image's
CUDA version. Recreate the Compose services after correcting runtime settings.
Both `pipeline` and `ollama` request GPU access; a working host driver alone
does not grant containers access.

## CUDA out of memory on 8 GB

Whisper `large-v3` and a 7B LLM can exceed an RTX 3070's 8 GB VRAM.
Sequential stages do not guarantee memory is released: Ollama may keep a model
resident and GPU allocators may retain memory. Inspect host `nvidia-smi`,
stop other GPU workloads, and begin with a short clip.

- Use a smaller Whisper model, for example pass
  `-e PIPELINE__TRANSCRIPTION__MODEL=medium` to a manual Compose run.
- Choose a smaller Ollama model, pull its exact tag in the Ollama service, and
  set `llm.model` to that tag.
- Disable LLM review with `PIPELINE__LLM__ENABLED=false`. For a manual run,
  `docker compose run --rm --no-deps -e PIPELINE__LLM__ENABLED=false pipeline ...`
  avoids starting Ollama; stop an already-running Ollama service separately.
- Set `transcription.compute_type: int8_float16` (environment:
  `PIPELINE__TRANSCRIPTION__COMPUTE_TYPE=int8_float16`) for supported GPU
  quantization. The default `auto` uses GPU `float16` or CPU `int8`.
- Set `transcription.device: cpu` to avoid CUDA entirely. With `device: auto`,
  CUDA initialization/transcription failures broadly fall back to CPU; explicit
  `device: cuda` fails instead of silently falling back.

For CLI/watcher jobs, model/configuration changes do not invalidate cached
transcription; follow the [reset procedure](../README.md#reset-a-job) to retest.
New web jobs include their per-job settings in the cache identity.

## Ollama unreachable or model not pulled

```sh
docker compose ps
docker compose logs ollama
docker compose exec ollama ollama list
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

Use the actual configured tag if different. Inside the pipeline container the
default address is `http://ollama:11434`, not `localhost` (which means the
pipeline container itself). A native host run needs an address reachable from
that host; Compose does not publish an Ollama host port by default.

Connection, timeout, missing-model, or other request failures cause candidate
review to return deterministic default cuts with reason `llm_unavailable`, not
a job failure. Inspect the job's `decisions.json` and pipeline logs.
Invalid LLM answers are validated and may fall back; LLM failure does not mean
nothing will be cut. The report has a non-LLM summary fallback.

## File not picked up

These checks apply to `watch`. The CPU quick start runs `serve`: upload or submit
an inbox path through <http://127.0.0.1:8000>; it does not watch for new files.

- Only files directly in the inbox are scanned, not subdirectories.
- Default extensions are `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, and `.m4v`
  (case-insensitive). A `.part` suffix is intentionally ignored.
- Wait for consecutive unchanged-size checks (default: two, five seconds apart).
  Size stability is not a reliable copy-completion signal if a network copy
  stalls; use a temporary extension and rename after copying.
- Verify the NAS is really mounted, the host bind-mount variables are correct,
  and container permissions allow reading and moving/deleting the input.
  Check NFS root squashing, UID/GID mappings, or SMB credentials/ACLs.
- Run `docker compose logs -f pipeline`. A job may already have moved the
  recording from the inbox into `/data/work/<job_id>/source.<extension>`.

## Why is the job not reprocessing? How do I force it?

State and artifacts live under `/data/work/<job_id>`. The watcher resumes unfinished
jobs on startup and skips completed stages/jobs. Reintroducing the same filename stem and
content reuses its job ID; changing settings alone does not trigger recomputation.
The watcher attempts pending-job resumption on startup, not on every inbox poll.
The Web UI's queue is not restored after a restart: use **Retry from saved stage**
on queued/interrupted jobs to reuse their persisted stages and saved settings.
Changing settings in a web job's form creates a separate run. Run only one server
per work directory, without concurrent watchers/manual processors.

Follow [Reset a job](../README.md#reset-a-job): stop all processors, back up the
ingested source outside work, delete only its job work folder, archive old
outputs, then reintroduce the backup and restart. Deleting the work folder
without a backup destroys the ingested copy. Deleting only outputs leaves a
completed state and will not regenerate them.

## Too many or too few cuts

Candidates must be silent **and** visually static; the model cannot invent
arbitrary cut timestamps.

| Setting | Less aggressive | More aggressive |
| --- | --- | --- |
| `cuts.min_silence_s` | Increase: require longer candidate gaps. | Decrease: allow shorter gaps. |
| `cuts.target_pause_s` | Increase: retain more of each pause. | Decrease: retain less of each pause. |
| `cuts.speech_margin_s` | Increase: protect more time around speech. | Decrease cautiously: less protection. |

Keep values nonnegative. A zero target is not a general “remove every pause”
switch: full removal requires explicit approval and the
`cuts.full_remove_min_s` threshold. Also inspect transcription timing, slide
detection, and `decisions.json` for kept candidates. For CLI/watcher runs, reset
cached jobs before comparing settings; the web form creates a separate run when
settings change. Listen around cut boundaries, not just total time saved.

## Slide changes missed, or cursor movement triggers changes

Lower `visual.change_threshold` for subtler changes; raise it to reject small
changes such as cursor motion. Increase `visual.sample_fps` to catch shorter
transitions, at the cost of more analysis work. Sampling and downscaling may
hide subtle differences; detection is not semantic slide recognition.

Compare the work artifact `visual.json` with source timestamps. Change one
setting at a time and reset the job to regenerate detection/candidates.

## What about animations or videos embedded in slides?

Movement can create repeated slide-change detections and reduce visually static
intervals, leaving fewer cut candidates. Slow/subtle movement can also be missed.
Silence does not prove that an animation or video is safe to shorten.
Increase `cuts.settle_after_slide_change_s` to protect more time after changes,
but review moving sections manually in Resolve. There is no slide-region mask,
animation understanding, or guaranteed embedded-video protection.

## Lithuanian/English mixed speech or misrecognized names

The default language hint is `lt`. Try `transcription.language: null` (or
`PIPELINE__TRANSCRIPTION__LANGUAGE=null`) for automatic language detection, or
`en` for predominantly English material. Automatic detection is not a guarantee
of accurate within-sentence code-switching. Keep `task: transcribe` if you want
the spoken language rather than an English translation.

Review names, abbreviations, technical terms, and low-confidence passages.
Correct final subtitle text manually and verify it against the audio. Report
review notes are heuristics, not corrections or a complete error list. The
current config exposes no custom vocabulary/initial-prompt setting.

## Subtitles out of sync

Use `<name>.edited.srt` for the cut timeline/render and `<name>.source.srt` for
the uncut recording. Import subtitles at timeline zero and check any Resolve
timecode offset. Subsequent edits require subtitle retiming.

Spot-check immediately before and after several cuts. Constant offset suggests
an import/timeline-start issue; growing drift suggests a frame-rate or source
timing problem; a jump after a cut suggests a remapping problem. Preserve the
`<name>.cutlist.json`, SRT, and source/edited timestamps when reporting a bug.

## Resolve import problems or wrong frame rate

Import the `.fcpxml` via **File → Import → Timeline**, not as a rendered movie.
Relink media: paths inside the XML may refer to `/data/work/...`, unavailable on
your workstation. Keep or copy the ingested source before deleting any job.
Check that your Resolve version supports the exported FCPXML and media codec.

Match the source and timeline frame rate before editing; variable-frame-rate
recordings can cause drift or unexpected cuts. If needed, make a constant-frame-
rate intermediate using your normal media workflow and process that copy as a
new job. There is no frame-rate setting in the current pipeline YAML.
Verify `docker compose run --rm pipeline auto-editor --version` and inspect
export logs: the Docker image's packaged CLI may differ from the local Nim CLI.

## Why are chapters missing?

YouTube requires **at least three chapters**, **at least 10 seconds each**,
and the first at **00:00**; the final chapter must also have 10 seconds of video.
The report omits chapters when there is insufficient edited duration.
`report.min_chapter_gap_s` can increase spacing but cannot bypass the 10-second
floor. Paste only the report's timestamp/title block into the YouTube
description. Retiming is necessary if you make further edits.

## Can I run CPU-only?

Yes. Follow the [Low-end / CPU-only quick start](../README.md#low-end--cpu-only-quick-start):

```sh
docker compose -f docker-compose.cpu.yml up --build -d
```

This standalone file has no GPU reservations or Ollama dependency; do not merge
it with the GPU Compose file. It forces device `cpu`, model `small`, automatic
language detection, and LLM disabled. The CPU build uses a plain Ubuntu 24.04
base and needs no NVIDIA driver, Container Toolkit, or CUDA image download.
Expect slower transcription; try `tiny` or `base` for limited hardware.
The README also documents native Python 3.10+, requirements, FFmpeg, and writable
local path overrides.

`transcription.device` supports `auto`, `cuda`, and `cpu`. `auto` broadly falls
back to CPU after CUDA initialization or transcription failures; explicit `cuda`
does not fall back. `transcription.compute_type: auto` selects GPU `float16` or
CPU `int8`. Disabling LLM alone does not remove the default GPU Compose file's
GPU requirements or stop an existing Ollama service.

## Why can the Web UI not open my file? Is it safe to expose?

Submitted paths must be inside the configured inbox (Docker:
`/data/inbox/...`), not arbitrary host paths. Uploads have a **16 GiB** limit.
The server keeps inbox source files; CLI `process` moves them unless you use
`--keep-source` (alias `--copy`).

The UI has one sequential worker for processing and preview rendering. Job pages
check progress every five seconds, but do not reload while video is playing or
settings have been changed; they show a refresh notice instead. The job list
still requires manual refresh. Progress shows the last completed stage, not
partial progress within a stage. The initial transcription may download a model.
Failed jobs show errors and a **Retry from saved stage** action.

The settings form covers supported per-job transcription, visual analysis, cut,
LLM, report, and preview controls. New web jobs include a settings fingerprint in
their identity; changing settings creates a separate run, rather than silently
reusing results generated with different settings. Older jobs remain accessible,
but cannot display settings that were never saved. Paths and service addresses
stay in server configuration. The CLI/watcher's source-based identity is unchanged.

Job pages provide source playback, read-only cut/decision review, downloads, and
escaped Markdown reports. **Render edited preview** creates an H.264/AAC MP4 from
a completed job's cached cutlist, without retranscribing. Source codecs may not
play in every browser; use the edited MP4 or download the source instead.
Native serving defaults to loopback; CPU Docker publishes only
`127.0.0.1:8000`. There is no authentication or TLS. Bind to `0.0.0.0` or change
the published host address only on a trusted network, never the public internet.

## Where are models cached? How much disk space is needed?

Compose uses named volumes `whisper-cache` at `/root/.cache/huggingface` in the
pipeline container and `ollama` at `/root/.ollama` in Ollama. Actual volume names
are normally prefixed by the Compose project name; inspect with
`docker volume ls` / `docker volume inspect`. Native runs use the current user's
cache locations instead.

Budget several GB per large model, Docker images, the full ingested video per
job, work JSON, deliverables, and another video-sized allocation for previews.
Exact sizes depend on model quantization, duration, and encoding. Monitor host
free space and `docker system df`. `docker compose down` preserves model caches;
`down -v` removes the named volumes and requires model downloads again.
Work/output bind mounts must be cleaned separately, after backing up media
needed by Resolve. Never prune volumes blindly.

## Is everything private and local?

Transcription, visual analysis, editing, and default Ollama inference run locally;
there is no required cloud transcription/LLM upload. Initial image/package/model
downloads need network access. A configured remote `llm.url` sends review context
and summary text there, so keep it local for private recordings.

NAS storage, transcripts, subtitles, reports, and logs may contain sensitive
content. Restrict permissions and backups; do not publish them in bug reports
without review. The default Compose file does not publish Ollama's port to the
host; do not expose it publicly without appropriate protection. Uploading to
YouTube or sharing outputs is a separate, explicit action.

## Does this modify the original auto-editor?

No. The pipeline lives in its own directory and invokes the CLI as a subprocess;
it does not refactor the Nim core. Timeline edits do not overwrite the source
media. By default CLI/watcher ingest **moves the inbox file** into its job work
directory; CLI `--keep-source`/`--copy` and Web UI jobs copy instead. Keep your
original recording backed up outside the inbox/work tree.
