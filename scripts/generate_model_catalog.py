from __future__ import annotations

import json
from pathlib import Path

from mlx_audio.model_catalog import ModelDocEntry, collect_model_doc_entries

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_SNIPPETS_DIR = REPO_ROOT / ".snippets" / "generated"
GENERATED_DIR = REPO_ROOT / ".generated"
UI_PUBLIC_DIR = REPO_ROOT / "mlx_audio" / "ui" / "public"

LANGUAGE_LABELS = {
    "ar": "AR",
    "de": "DE",
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "hi": "HI",
    "it": "IT",
    "ja": "JA",
    "multilingual": "Multilingual",
    "nl": "NL",
    "pt": "PT",
    "zh": "ZH",
}


def yes_no(value: bool | None) -> str:
    if value is None:
        return "--"
    return "Yes" if value else "--"


def docs_link(entry: ModelDocEntry) -> str:
    page_name = Path(entry.docs_path).name
    return f"[**{entry.name}**]({page_name}.md)"


def repo_link(entry: ModelDocEntry) -> str:
    return f"[{entry.repo}](https://huggingface.co/{entry.repo})"


def format_languages(codes: tuple[str, ...]) -> str:
    return ", ".join(LANGUAGE_LABELS.get(code, code.upper()) for code in codes) or "--"


def render_tts_table(entries: list[ModelDocEntry]) -> str:
    lines = [
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->",
        "",
        "| Model | Description | Languages | Voice Cloning | Streaming | Repo |",
        "|-------|-------------|-----------|:-------------:|:---------:|------|",
    ]
    for entry in entries:
        lines.append(
            f"| {docs_link(entry)} | {entry.description} | {format_languages(entry.languages)} | "
            f"{yes_no(entry.voice_cloning)} | {yes_no(entry.streaming)} | {repo_link(entry)} |"
        )
    return "\n".join(lines) + "\n"


def render_stt_table(entries: list[ModelDocEntry]) -> str:
    lines = [
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->",
        "",
        "| Model | Description | Languages | Streaming | Timestamps | Repo |",
        "|-------|-------------|-----------|:---------:|:----------:|------|",
    ]
    for entry in entries:
        lines.append(
            f"| {docs_link(entry)} | {entry.description} | {format_languages(entry.languages)} | "
            f"{yes_no(entry.streaming)} | {yes_no(entry.timestamps)} | {repo_link(entry)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    entries = collect_model_doc_entries(ignore_import_errors=False)
    DOCS_SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    UI_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    catalog = [entry.to_dict() for entry in entries]
    (GENERATED_DIR / "model-catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    (UI_PUBLIC_DIR / "model-catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )

    tts_entries = [entry for entry in entries if entry.task == "tts"]
    stt_entries = [entry for entry in entries if entry.task == "stt"]

    (DOCS_SNIPPETS_DIR / "tts-model-catalog.md").write_text(
        render_tts_table(tts_entries),
        encoding="utf-8",
    )
    (DOCS_SNIPPETS_DIR / "stt-model-catalog.md").write_text(
        render_stt_table(stt_entries),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
