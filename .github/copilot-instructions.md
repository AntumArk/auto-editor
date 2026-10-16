# Copilot instructions

## Agent skills
Agents working in this repository MAY create or update agent skills when they would help future work
(e.g. repeated workflows for the presentation pipeline). Rules:

- Place skills in `.github/skills/<skill-name>/SKILL.md` (lowercase, hyphenated name).
- Each `SKILL.md` starts with YAML frontmatter containing `name` and `description`, followed by concise instructions.
- Supporting scripts/examples may live alongside `SKILL.md` in the same folder.
- Only create a skill when it captures a reusable procedure (e.g. running the pipeline tests, testing auto-editor Resolve export, validating Ollama output); don't duplicate README content.
- Don't put secrets or machine-specific paths in skills.

## Project notes
- `presentation-pipeline/` is a Python, Dockerized pipeline wrapped around the existing Nim `auto-editor` CLI. Don't refactor the Nim core; call the `auto-editor` executable.
- Pure logic (cuts, subtitles, chapters, LLM validation) must be unit-testable with pytest without GPU or network.
