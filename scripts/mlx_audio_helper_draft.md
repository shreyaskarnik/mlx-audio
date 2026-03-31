# `mlx-audio-helper` Draft

This is a draft for a contributor-facing helper that makes new model additions more consistent.

## Goal

When someone adds a new model, the helper should ask a short set of structured questions and scaffold:

- the model package skeleton
- a co-located `README.md`
- a checklist of follow-up work

The point is not to fully implement a model automatically. The point is to make the first 80% deterministic so contributors do less copy/paste and the repo structure stays predictable.

## Proposed command

```bash
python scripts/mlx_audio_helper.py add-model --interactive --write-preview
```

Or in a future packaged form:

```bash
mlx-audio-helper add-model
```

For non-interactive / agent-driven use:

```bash
python scripts/mlx_audio_helper.py add-model \
  --task tts \
  --model-name "Example TTS" \
  --package-name example_tts \
  --hf-repo mlx-community/example-tts \
  --languages en,fr \
  --capabilities streaming,voice_cloning \
  --output-format json \
  --write-target preview
```

## Prompt flow

The first version should ask only a few high-signal questions:

1. Task: `tts`, `stt`, `sts`, or `vad`
2. Display name
3. Slug / package name
4. Hugging Face repo
5. Upstream repo or paper
6. Language codes
7. Capabilities:
   - `streaming`
   - `voice_cloning`
   - `timestamps`
   - `diarization`
8. Whether to create a co-located `README.md`
9. Whether the model needs a `convert.py` stub
10. Any extra notes

That keeps the flow short enough to be used, while still producing a predictable package shape.

The interactive path should feel like a small generator wizard, not a pile of raw prompts. Using `rich` for step headers, summaries, and confirmation screens plus `questionary` for checkbox-style multi-select prompts is a good fit here without turning the helper into a full TUI app.

## What the draft helper emits

Today, the script writes a safe preview under `output/helper-drafts/` instead of modifying the real package tree.

It can also write directly into the repo with:

```bash
python scripts/mlx_audio_helper.py add-model ... --write-target repo
```

That path is better suited to agents or maintainer workflows than to first-time contributors.

It generates:

- `__init__.py`
- `<model_name>.py` runtime stub
- `README.md`
- optional `convert.py`
- `plan.md`
- `answers.json`

This is enough for maintainers to review the shape before wiring the helper into the actual source tree.

## Existing-model checks

Before generating anything, the helper should fail fast if:

- the target package directory already exists
- the package name already exists under another task
- the same Hugging Face repo already appears in an existing model package

That avoids scaffolding obvious duplicates and gives contributors a quick sanity check before they start.

## Why this helps

- More consistent model package layout
- Easier onboarding for community contributors
- Cleaner starting point before maintainers wire in task-specific details

## Why this could also be a Codex skill

This flow maps well to a Codex skill because a skill can:

- inspect the repo structure before asking questions
- infer task-specific defaults from nearby model packages
- generate or edit files directly in the right locations
- update related files in one pass

In other words:

- `mlx-audio-helper` is a good local CLI for contributors
- a Codex skill is a good higher-level maintainer / automation workflow

They are complementary rather than competing ideas.

## Suggested next step

If this direction looks right, the next iteration should:

1. move the preview writer to a real scaffold writer behind an explicit flag
2. add task-specific templates
3. decide whether the canonical UX should be:
   - a packaged CLI
   - a repo script
   - a Codex skill
   - or all three
