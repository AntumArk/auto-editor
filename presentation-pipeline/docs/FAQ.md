# Presentation pipeline FAQ

See the [setup guide](../README.md) for prerequisites, configuration defaults,
and the integration-status caveat. Commands here assume the Compose directory:
`/home/runner/work/auto-editor/auto-editor/presentation-pipeline`.
Hardware, mount, and Resolve commands are approximate and have not been verified
on your machine. Replace example filenames and job IDs before running them.

## GPU not detected in the container

First run `nvidia-smi` on the host, then:

```sh
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
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
- `compute_type=int8_float16` is a useful faster-whisper memory-saving option,
  **but it is not exposed by the current pipeline configuration**.
  Transcription currently tries CUDA `float16`, then CPU `int8` after a
  `RuntimeError` or `ValueError`. Do not add a compute-type environment override
  expecting it to work; using that GPU precision requires a future code change.

Model/configuration changes do not invalidate cached transcription. Follow the
[reset procedure](../README.md#reset-a-job) to retest from the beginning.

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
an automatic job failure. Inspect the job's `decisions.json` and pipeline logs.
Invalid LLM answers also undergo validation and may fall back; LLM failure does
not mean nothing will be cut. The report has a non-LLM summary fallback.
This fallback does not fix unrelated orchestrator wiring errors in an
unintegrated checkout.

## File not picked up

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

State and artifacts live under `/data/work/<job_id>`. Restarting resumes unfinished
jobs and skips completed stages/jobs. Reintroducing the same filename stem and
content reuses its job ID; changing settings alone does not trigger recomputation.
The watcher attempts pending-job resumption on startup, not on every inbox poll.

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
detection, and `decisions.json` for kept candidates. Reset cached jobs before
comparing settings; listen around cut boundaries, not just total time saved.

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
timing problem; a jump after a cut suggests a remapping problem. Confirm the
integration fixes pass kept ranges to subtitle remapping. Preserve the cut list,
SRT, and source/edited timestamps when reporting a bug.

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
floor. Confirm the report uses the integrated edited-time slide/transcript data,
then paste only its timestamp/title block into the YouTube description. Retiming
is necessary if you make further edits.

## Can I run CPU-only?

Yes, but expect much slower transcription, especially with `large-v3`; prefer
`medium` or smaller and disable LLM review, or use a small CPU Ollama model.
The current Compose file requires NVIDIA devices for **both** services, so it
will not simply start on a non-GPU host.

For a CPU-only deployment, adapt a local Compose copy by removing both services'
GPU device reservation blocks (an empty override may not remove merged values).
Alternatively run the installed Python requirements and FFmpeg natively, with
writable path overrides and `python3 -m app.main process /path/to/video.mp4`.
These deployment adaptations are unverified here.

Whisper attempts CUDA first and falls back to CPU `int8` on runtime/value errors;
there is no supported `transcription.device` override to force CPU. Disabling
LLM review alone does not remove Compose GPU requirements or stop Ollama.

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
media, but **ingest moves the inbox file** into its job work directory. Keep your
original recording backed up outside the inbox/work tree.
