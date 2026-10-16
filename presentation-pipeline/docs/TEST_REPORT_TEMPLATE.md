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
