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
pip install -r requirements-test.txt
pytest -v
```

**Pass:** all tests pass, including cuts, subtitles, chapters, LLM validation,
configuration, state, transcription fallback/cleanup, and speaker-overlap tests.
Whisper, WhisperX, and pyannote are mocked; no models, GPU, or network are needed.

## Phase 2 – Docker & GPU smoke test

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
cd presentation-pipeline
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose run --rm pipeline python -c "import faster_whisper, cv2; print('ok')"
docker compose run --rm pipeline auto-editor --version
```

**Warning:** on 8 GB GPUs (for example, an RTX 3070), Whisper large-v3 and a
7B LLM will not fit in memory at the same time. Transcription should finish
and free VRAM before the LLM step. CUDA now defaults to `int8_float16` (including
when `compute_type: auto`), with configured precision → `int8_float16` → CPU
`int8` fallback, including errors during lazy transcription. Whisper, optional
alignment, and diarization models are released between stages; garbage
collection and optional PyTorch CUDA cache cleanup run before LLM review.
Unload/stop any already resident Ollama model before transcription. If memory
is still tight, use CPU transcription/diarization, a smaller LLM, or disable LLM.
Measure actual peak VRAM; fitting on 8 GB is a target, not a guarantee.

For optional backend smoke tests, install `requirements-extra.txt` alongside
`requirements.txt`, or use
`docker compose build --build-arg INSTALL_TRANSCRIPTION_EXTRAS=true pipeline`.
Check the host driver's compatibility with PyTorch's CUDA wheels. Accept Hugging
Face terms for **both** `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0`, then export a read token as `HF_TOKEN` without saving
it in YAML, logs, or the repository. `diarization.hf_token` names that environment
variable; Docker needs explicit `-e HF_TOKEN` forwarding. See the
[README](../README.md#optional-alignment-and-speaker-labels) for setup.

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
| Alignment (optional) | With `transcription.align: true`, compare word boundaries by listening. Lithuanian requires a vetted `align_model`; unsupported/missing models should warn and keep original timings. |
| Diarization (optional) | With `diarization.enabled: true`, inspect segment/word `speaker` labels at speaker changes. Test `min_speakers`/`max_speakers`, CPU/CUDA, and optional SRT prefixes in both source and edited files. |
| Slide detection | Change times match real slide changes (spot-check about five); cursor movement is not counted. |
| Cut candidates | Candidates occur only during silence on an unchanged slide; none cut into speech. |
| LLM review | Output is valid JSON; with Ollama stopped, the run still completes with defaults (reason `llm_unavailable`). |
| Export | The timeline imports into DaVinci Resolve with correct cuts; `cutlist.json` is present. |
| Subtitles | `clip.edited.srt` stays in sync in Resolve, especially right after cuts. |
| Report | The first chapter is `00:00`, chapters are at least 10 s apart, and stats are consistent (`original − edited = removed`). |

### Compare transcription settings on the same 3-minute clip

1. Keep an untouched copy of the clip. Use separate work/output directories per
   configuration, or follow the README's **Reset a job** instructions; completed
   stages are cached and configuration changes do not invalidate them.
2. Baseline: `large-v3`, CUDA `int8_float16`, `beam_size: 5`,
   `condition_on_previous_text: false`, VAD and word timestamps enabled.
   Prefer this over `large-v3-turbo`, which is generally worse for Lithuanian.
3. Change one option per run: a glossary in `initial_prompt`, `beam_size`,
   `vad_parameters`, then optional alignment with a language-compatible model.
   Compare the same phrases/names against a manually checked transcript.
4. Enable diarization separately (`device: cpu` is the low-VRAM default).
   Compare overlap-only assignment against `fill_nearest: true`; inspect silence,
   zero-duration words, and overlapping speakers. Labels are anonymous and may
   change numbering between runs.
5. Record text errors, word-boundary accuracy, speaker-change accuracy, subtitle
   sync after cuts, elapsed time, peak VRAM, and fallback/alignment warnings.
   Confirm VRAM falls before alignment/diarization and LLM stages. Never attach
   tokens or private recordings to test reports.

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
