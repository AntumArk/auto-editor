# Presentation pipeline testing

Commands and paths below are approximate. Verify them against the current
`presentation-pipeline/README.md` and implementation before running them.

## Phase 0 – Pre-merge integration

- Merge order: Part 1 (skeleton/orchestration) first; merge the other parts in
  any order because they use separate files.
- Verify that the orchestrator and module signatures match:
  `transcribe.transcribe`, `vision.detect_slide_changes`,
  `candidates.generate_candidates`, `llm.review_candidates`,
  `cutlist.build_cutlist`, `cutlist.export_with_auto_editor`,
  `subtitles.remap_segments`/`write_srt`, and `report.write_report`.
  `remap_segments` expects keep intervals, so the orchestrator should pass
  `cutlist.keep`.
- Confirm CI and tests are green for each PR.

## Phase 1 – Unit tests (no GPU)

```bash
cd presentation-pipeline
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
pytest -v
```

**Pass:** all tests pass, including cuts, subtitles, chapters, LLM validation,
configuration, and state tests.

## Phase 2 – Docker & GPU smoke test

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
cd presentation-pipeline
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose run --rm pipeline python -c "import faster_whisper, cv2; print('ok')"
docker compose run --rm pipeline auto-editor --version
```

**Warning:** on 8 GB GPUs (for example, an RTX 3070), Whisper large-v3 and a
7B LLM will not fit in memory at the same time. Transcription should finish
and free VRAM before the LLM step. If you run out of memory, use
`compute_type=int8_float16` or a smaller LLM.

## Phase 3 – Stage-by-stage on a short clip

Make a 2–3 minute clip:

```bash
ffmpeg -ss 0 -t 180 -i full.mp4 -c copy /data/inbox/clip.mp4
```

Do not commit test videos because of their size and copyright. Download
examples locally, for example:

```bash
yt-dlp -f "bv*+ba/b" <url>
```

| Stage | Check |
|---|---|
| Ingest | The file is picked up only after copying finishes; a job folder appears in `/data/work`; a restart does not reprocess it. |
| Transcription | `transcript.json` has word timings; `clip.source.srt` reads correctly in LT/EN. |
| Slide detection | Change times match real slide changes (spot-check about five); cursor movement is not counted. |
| Cut candidates | Candidates occur only during silence on an unchanged slide; none cut into speech. |
| LLM review | Output is valid JSON; with Ollama stopped, the run still completes with defaults (reason `llm_unavailable`). |
| Export | The timeline imports into DaVinci Resolve with correct cuts; `cutlist.json` is present. |
| Subtitles | `clip.edited.srt` stays in sync in Resolve, especially right after cuts. |
| Report | The first chapter is `00:00`, chapters are at least 10 s apart, and stats are consistent (`original − edited = removed`). |

## Phase 4 – Full end-to-end

Drop a full video into `/data/inbox`, let the watcher process it, then review
the result in Resolve for natural pauses.

## Tuning knobs

- `visual.change_threshold`
- `cuts.min_silence_s`
- `cuts.target_pause_s`
- `cuts.speech_margin_s`
- `cuts.settle_after_slide_change_s`
- `llm.model`

## Reporting results

Use the copy-paste template below, also available as
[TEST_REPORT_TEMPLATE.md](TEST_REPORT_TEMPLATE.md).

```markdown
## Environment
- GPU / VRAM / driver / CUDA:
- Merged PRs (numbers) and commit SHA:

## Results per phase
- Phase 1 unit tests: pass/fail (paste failures)
- Phase 2 Docker/GPU: pass/fail
- Phase 3 per stage: pass/fail + notes
- Phase 4 end-to-end: pass/fail

## Failures
- Stage + full error / traceback (`docker compose logs pipeline`)

## Quality observations
- Cuts too aggressive / too timid? Examples with source timestamps
- Missed or false slide changes? Timestamps
- Subtitle drift? Where
- LLM decisions that were wrong?
- Transcript errors (names/terms)

## Numbers
- Original / edited duration, # cuts
- Processing time per stage
- Peak VRAM

## Attachments
- `cutlist.json`
- `report.md`
- First ~30 lines of `edited.srt`
- `config.yaml` used
```
