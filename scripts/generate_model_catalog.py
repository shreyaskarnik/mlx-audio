from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx_audio.model_catalog import ModelDocEntry, collect_model_doc_entries

DOCS_SNIPPETS_DIR = REPO_ROOT / ".snippets" / "generated"
GENERATED_DIR = REPO_ROOT / ".generated"
UI_PUBLIC_DIR = REPO_ROOT / "mlx_audio" / "ui" / "public"

LANGUAGE_LABELS = {
    "ar": "AR",
    "bg": "BG",
    "cs": "CS",
    "da": "DA",
    "de": "DE",
    "el": "EL",
    "en": "EN",
    "es": "ES",
    "fi": "FI",
    "fr": "FR",
    "hr": "HR",
    "hu": "HU",
    "hi": "HI",
    "it": "IT",
    "ja": "JA",
    "ko": "KO",
    "multilingual": "Multilingual",
    "nl": "NL",
    "no": "NO",
    "pl": "PL",
    "pt": "PT",
    "ro": "RO",
    "ru": "RU",
    "sk": "SK",
    "sr": "SR",
    "sv": "SV",
    "tr": "TR",
    "uk": "UK",
    "zh": "ZH",
}


def yes_no(value: bool | None) -> str:
    if value is None:
        return "--"
    return "Yes" if value else "--"


def docs_link(entry: ModelDocEntry) -> str:
    if not entry.docs_path:
        return f"**{entry.name}**"
    if entry.docs_path.startswith(("http://", "https://")):
        return f"[**{entry.name}**]({entry.docs_path})"
    page_name = Path(entry.docs_path).name
    return f"[**{entry.name}**]({page_name}.md)"


def repo_link(entry: ModelDocEntry) -> str:
    if not entry.repo:
        return "--"
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


def render_sts_table(entries: list[ModelDocEntry]) -> str:
    lines = [
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->",
        "",
        "| Model | Description | Repo |",
        "|-------|-------------|------|",
    ]
    for entry in entries:
        lines.append(f"| {docs_link(entry)} | {entry.description} | {repo_link(entry)} |")
    return "\n".join(lines) + "\n"


def render_vad_table(entries: list[ModelDocEntry]) -> str:
    lines = [
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->",
        "",
        "| Model | Description | Streaming | Diarization | Repo |",
        "|-------|-------------|:---------:|:-----------:|------|",
    ]
    for entry in entries:
        lines.append(
            f"| {docs_link(entry)} | {entry.description} | "
            f"{yes_no(entry.streaming)} | {yes_no(entry.diarization)} | {repo_link(entry)} |"
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
    sts_entries = [entry for entry in entries if entry.task == "sts"]
    vad_entries = [entry for entry in entries if entry.task == "vad"]

    (DOCS_SNIPPETS_DIR / "tts-model-catalog.md").write_text(
        render_tts_table(tts_entries),
        encoding="utf-8",
    )
    (DOCS_SNIPPETS_DIR / "stt-model-catalog.md").write_text(
        render_stt_table(stt_entries),
        encoding="utf-8",
    )
    (DOCS_SNIPPETS_DIR / "sts-model-catalog.md").write_text(
        render_sts_table(sts_entries),
        encoding="utf-8",
    )
    (DOCS_SNIPPETS_DIR / "vad-model-catalog.md").write_text(
        render_vad_table(vad_entries),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
