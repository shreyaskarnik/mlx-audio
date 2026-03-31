from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx_audio.model_catalog import (
    ModelDocEntry,
    REPO_BLOB_MAIN_URL,
    collect_model_doc_entries,
)

DOCS_MODELS_DIR = REPO_ROOT / "docs" / "models"
DOCS_SNIPPETS_DIR = REPO_ROOT / ".snippets" / "generated"
GENERATED_DIR = REPO_ROOT / ".generated"
UI_PUBLIC_DIR = REPO_ROOT / "mlx_audio" / "ui" / "public"
DOCS_INDEX_FILES = [
    DOCS_MODELS_DIR / "index.md",
    DOCS_MODELS_DIR / "tts" / "index.md",
    DOCS_MODELS_DIR / "stt" / "index.md",
    DOCS_MODELS_DIR / "sts" / "index.md",
    DOCS_MODELS_DIR / "vad" / "index.md",
]
GENERATED_DOC_DIRS = [
    DOCS_MODELS_DIR / "tts" / "generated",
    DOCS_MODELS_DIR / "stt" / "generated",
    DOCS_MODELS_DIR / "sts" / "generated",
    DOCS_MODELS_DIR / "vad" / "generated",
]

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

TASK_INFO = {
    "tts": {
        "title": "Text-to-Speech Models",
        "overview": (
            "MLX Audio supports a wide range of TTS models optimized for Apple "
            "Silicon. The generated catalog below is derived from "
            "`ModelConfig.DOCS` metadata so model capabilities stay in sync "
            "with the implementation."
        ),
        "tips": [
            "**Fastest / smallest:** Kokoro and Soprano are good low-footprint defaults.",
            "**Voice cloning:** Qwen3-TTS, CSM, Chatterbox, and VibeVoice all support cloned or conditioned voices.",
            "**Streaming:** Qwen3-TTS, Voxtral TTS, Chatterbox Turbo, PocketTTS, and VibeVoice support chunked playback.",
        ],
    },
    "stt": {
        "title": "Speech-to-Text Models",
        "overview": (
            "MLX Audio provides speech-to-text models ranging from lightweight "
            "English-only recognizers to multilingual systems with streaming, "
            "timestamps, translation, and diarization support."
        ),
        "tips": [
            "**Broadest multilingual coverage:** Whisper and MMS cover the most languages.",
            "**Streaming:** Parakeet, Qwen3-ASR, VibeVoice-ASR, Voxtral Realtime, and Granite Speech support low-latency flows.",
            "**Alignment / timestamps:** Whisper, Parakeet, and Qwen3-ForcedAligner are the strongest choices.",
        ],
    },
    "sts": {
        "title": "Speech-to-Speech Models",
        "overview": (
            "MLX Audio includes speech enhancement, source separation, and "
            "multimodal speech interaction models under the STS umbrella."
        ),
        "tips": [
            "**Speech enhancement:** DeepFilterNet and MossFormer2 SE focus on cleanup and denoising.",
            "**Source separation:** SAM-Audio is the text-guided separation option.",
            "**Conversation / multimodal speech:** Moshi and Liquid2.5-Audio cover interactive audio workflows.",
        ],
    },
    "vad": {
        "title": "Voice Activity Detection Models",
        "overview": (
            "MLX Audio includes endpoint detection and diarization models for "
            "conversational turn-taking and speaker-aware processing."
        ),
        "tips": [
            "**Turn-taking / endpointing:** Smart Turn is the dedicated endpoint detector.",
            "**Diarization:** Sortformer variants handle speaker attribution, with v2.1 optimized for streaming.",
        ],
    },
}

TASK_LABELS = {
    "tts": "Text-to-Speech",
    "stt": "Speech-to-Text",
    "sts": "Speech-to-Speech",
    "vad": "Voice Activity Detection",
}


def yes_no(value: bool | None) -> str:
    if value is None:
        return "--"
    return "Yes" if value else "--"


def format_languages(codes: tuple[str, ...]) -> str:
    return ", ".join(LANGUAGE_LABELS.get(code, code.upper()) for code in codes) or "--"


def render_languages_cell(codes: tuple[str, ...]) -> str:
    if not codes:
        return "--"
    chips = "".join(
        f'<span class="model-language-chip">{LANGUAGE_LABELS.get(code, code.upper())}</span>'
        for code in codes
    )
    return f'<div class="model-language-chips">{chips}</div>'


def repo_link(entry: ModelDocEntry) -> str:
    if not entry.repo:
        return "--"
    return f"[{entry.repo}](https://huggingface.co/{entry.repo})"


def generated_page_href(entry: ModelDocEntry, relative_prefix: str) -> str:
    return f"{relative_prefix}/{entry.slug}.md"


def generated_page_link(entry: ModelDocEntry, relative_prefix: str) -> str:
    return f"[**{entry.name}**]({generated_page_href(entry, relative_prefix)})"


def source_link(entry: ModelDocEntry) -> str:
    source_url = source_url_for_entry(entry)
    if not source_url:
        return "--"
    return f"[Source Docs]({source_url})"


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def capabilities(entry: ModelDocEntry) -> list[str]:
    items = []
    if entry.streaming:
        items.append("Streaming")
    if entry.voice_cloning:
        items.append("Voice cloning")
    if entry.timestamps:
        items.append("Timestamps")
    if entry.diarization:
        items.append("Diarization")
    if entry.tags:
        items.extend(tag.replace("-", " ") for tag in entry.tags)
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def render_tts_table(entries: list[ModelDocEntry], relative_prefix: str) -> str:
    lines = [
        "| Model | Description | Languages | Voice Cloning | Streaming | Repo |",
        "|-------|-------------|-----------|:-------------:|:---------:|------|",
    ]
    for entry in entries:
        lines.append(
                f"| {generated_page_link(entry, relative_prefix)} | "
                f"{markdown_cell(entry.description)} | {render_languages_cell(entry.languages)} | "
                f"{yes_no(entry.voice_cloning)} | {yes_no(entry.streaming)} | {repo_link(entry)} |"
        )
    return "\n".join(lines)


def render_stt_table(entries: list[ModelDocEntry], relative_prefix: str) -> str:
    lines = [
        "| Model | Description | Languages | Streaming | Timestamps | Repo |",
        "|-------|-------------|-----------|:---------:|:----------:|------|",
    ]
    for entry in entries:
        lines.append(
                f"| {generated_page_link(entry, relative_prefix)} | "
                f"{markdown_cell(entry.description)} | {render_languages_cell(entry.languages)} | "
                f"{yes_no(entry.streaming)} | {yes_no(entry.timestamps)} | {repo_link(entry)} |"
        )
    return "\n".join(lines)


def render_sts_table(entries: list[ModelDocEntry], relative_prefix: str) -> str:
    lines = [
        "| Model | Description | Repo |",
        "|-------|-------------|------|",
    ]
    for entry in entries:
        lines.append(
            f"| {generated_page_link(entry, relative_prefix)} | "
            f"{markdown_cell(entry.description)} | {repo_link(entry)} |"
        )
    return "\n".join(lines)


def render_vad_table(entries: list[ModelDocEntry], relative_prefix: str) -> str:
    lines = [
        "| Model | Description | Streaming | Diarization | Repo |",
        "|-------|-------------|:---------:|:-----------:|------|",
    ]
    for entry in entries:
        lines.append(
            f"| {generated_page_link(entry, relative_prefix)} | "
            f"{markdown_cell(entry.description)} | {yes_no(entry.streaming)} | "
            f"{yes_no(entry.diarization)} | {repo_link(entry)} |"
        )
    return "\n".join(lines)


def render_summary_table(entries: list[ModelDocEntry], relative_prefix: str) -> str:
    uses_languages = any(entry.languages for entry in entries)
    if uses_languages:
        lines = [
            "| Model | Description | Languages | Repo |",
            "|-------|-------------|-----------|------|",
        ]
        for entry in entries:
            lines.append(
                f"| {generated_page_link(entry, relative_prefix)} | "
                f"{markdown_cell(entry.description)} | {render_languages_cell(entry.languages)} | "
                f"{repo_link(entry)} |"
            )
        return "\n".join(lines)

    lines = [
        "| Model | Description | Repo |",
        "|-------|-------------|------|",
    ]
    for entry in entries:
        lines.append(
            f"| {generated_page_link(entry, relative_prefix)} | "
            f"{markdown_cell(entry.description)} | {repo_link(entry)} |"
        )
    return "\n".join(lines)


def render_task_table(task: str, entries: list[ModelDocEntry], relative_prefix: str) -> str:
    if task == "tts":
        return render_tts_table(entries, relative_prefix)
    if task == "stt":
        return render_stt_table(entries, relative_prefix)
    if task == "sts":
        return render_sts_table(entries, relative_prefix)
    return render_vad_table(entries, relative_prefix)


def task_entries(entries: list[ModelDocEntry], task: str) -> list[ModelDocEntry]:
    return [entry for entry in entries if entry.task == task]


def repo_path_from_blob_url(url: str) -> Path | None:
    if not url.startswith(REPO_BLOB_MAIN_URL):
        return None
    relative_path = url.removeprefix(REPO_BLOB_MAIN_URL)
    return REPO_ROOT / relative_path


def local_source_path(entry: ModelDocEntry) -> Path | None:
    if not entry.docs_path:
        return None
    repo_path = repo_path_from_blob_url(entry.docs_path)
    if repo_path is not None and repo_path.exists():
        return repo_path
    if entry.docs_path.startswith(("http://", "https://")):
        return None
    docs_relative = entry.docs_path.strip("/")
    if not docs_relative:
        return None
    candidate = REPO_ROOT / "docs" / f"{docs_relative}.md"
    return candidate if candidate.exists() else None


def repo_blob_url(path: Path) -> str:
    return REPO_BLOB_MAIN_URL + path.relative_to(REPO_ROOT).as_posix()


def source_url_for_entry(entry: ModelDocEntry) -> str | None:
    source_path = local_source_path(entry)
    if source_path is not None:
        return repo_blob_url(source_path)
    if entry.docs_path and entry.docs_path.startswith(("http://", "https://")):
        return entry.docs_path
    return None


def strip_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    _, _, remainder = text.partition("\n---\n")
    return remainder if remainder else text


def strip_leading_heading(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("# "):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def rewrite_relative_links(text: str, source_path: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        target = match.group("target").strip()
        title = match.group("title") or ""
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            return match.group(0)
        resolved = (source_path.parent / target).resolve()
        if not resolved.exists() or REPO_ROOT not in resolved.parents and resolved != REPO_ROOT:
            return match.group(0)
        return f"[{label}]({repo_blob_url(resolved)}{title})"

    pattern = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")
    return pattern.sub(replace, text)


def load_source_content(entry: ModelDocEntry) -> str | None:
    source_path = local_source_path(entry)
    if source_path is None:
        return None
    text = source_path.read_text(encoding="utf-8")
    text = strip_front_matter(text)
    text = strip_leading_heading(text)
    text = rewrite_relative_links(text, source_path)
    return text or None


def render_metadata_table(entry: ModelDocEntry) -> str:
    lines = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Task | {TASK_LABELS[entry.task]} |",
        f"| Languages | {render_languages_cell(entry.languages)} |",
        f"| Repo | {repo_link(entry)} |",
        f"| Source Docs | {source_link(entry)} |",
    ]
    if entry.license:
        lines.append(f"| License | `{entry.license}` |")
    if entry.pipeline_tag:
        lines.append(f"| Pipeline Tag | `{entry.pipeline_tag}` |")
    feature_items = capabilities(entry)
    if feature_items:
        lines.append(f"| Features | {', '.join(feature_items)} |")
    return "\n".join(lines)


def render_model_detail_page(entry: ModelDocEntry) -> str:
    content = [
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->",
        "",
        f"# {entry.name}",
        "",
        entry.description,
        "",
        render_metadata_table(entry),
    ]

    source_content = load_source_content(entry)
    if source_content:
        content.extend(["", "## Documentation", "", source_content.strip()])
    else:
        content.extend(
            [
                "",
                "## Documentation",
                "",
                "No local model markdown source was found for this entry.",
                "",
                "Add a single top-level `README.md` to the model folder or set an explicit "
                "`docs_path` override in `ModelConfig.DOCS` to render fuller model docs here.",
            ]
        )

    return "\n".join(content).strip() + "\n"


def render_models_home(entries: list[ModelDocEntry]) -> str:
    sections = [
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->",
        "",
        "# Models",
        "",
        "MLX Audio supports a wide range of audio models across four categories, all optimized for Apple Silicon.",
        "",
        "Many hosted MLX checkpoints referenced in these docs live under "
        "[mlx-community](https://huggingface.co/mlx-community) on Hugging Face, the shared org "
        "for ready-to-use MLX model weights across projects like `mlx-lm`, `mlx-vlm`, and "
        "`mlx-audio`. If you are adding a new model, prefer publishing it there when possible "
        "so users can find MLX models in one consistent place.",
        "",
    ]

    for task in ("tts", "stt", "sts", "vad"):
        entries_for_task = task_entries(entries, task)
        sections.extend(
            [
                f"## {TASK_LABELS[task]}",
                "",
                TASK_INFO[task]["overview"],
                "",
                render_summary_table(entries_for_task, f"{task}/generated"),
                "",
                f"[:octicons-arrow-right-24: Browse {TASK_LABELS[task]}]({task}/index.md)",
                "",
                "---",
                "",
            ]
        )

    return "\n".join(sections[:-2]).strip() + "\n"


def render_task_index(task: str, entries: list[ModelDocEntry]) -> str:
    info = TASK_INFO[task]
    content = [
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->",
        "",
        f"# {info['title']}",
        "",
        info["overview"],
        "",
        "## Model Catalog",
        "",
        render_task_table(task, entries, "generated"),
    ]
    if info["tips"]:
        content.extend(["", '!!! tip "Choosing a model"'])
        for tip in info["tips"]:
            content.append(f"    - {tip}")

    return "\n".join(content).strip() + "\n"


def write_model_pages(entries: list[ModelDocEntry]) -> None:
    for directory in GENERATED_DOC_DIRS:
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        page_path = DOCS_MODELS_DIR / entry.task / "generated" / f"{entry.slug}.md"
        page_path.write_text(render_model_detail_page(entry), encoding="utf-8")


def write_model_indexes(entries: list[ModelDocEntry]) -> None:
    (DOCS_MODELS_DIR / "index.md").write_text(
        render_models_home(entries),
        encoding="utf-8",
    )
    for task in ("tts", "stt", "sts", "vad"):
        (DOCS_MODELS_DIR / task / "index.md").write_text(
            render_task_index(task, task_entries(entries, task)),
            encoding="utf-8",
        )


def write_legacy_snippets(entries: list[ModelDocEntry]) -> None:
    DOCS_SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_SNIPPETS_DIR / "tts-model-catalog.md").write_text(
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->\n\n"
        + render_tts_table(task_entries(entries, "tts"), "generated"),
        encoding="utf-8",
    )
    (DOCS_SNIPPETS_DIR / "stt-model-catalog.md").write_text(
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->\n\n"
        + render_stt_table(task_entries(entries, "stt"), "generated"),
        encoding="utf-8",
    )
    (DOCS_SNIPPETS_DIR / "sts-model-catalog.md").write_text(
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->\n\n"
        + render_sts_table(task_entries(entries, "sts"), "generated"),
        encoding="utf-8",
    )
    (DOCS_SNIPPETS_DIR / "vad-model-catalog.md").write_text(
        "<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->\n\n"
        + render_vad_table(task_entries(entries, "vad"), "generated"),
        encoding="utf-8",
    )


def write_catalog(entries: list[ModelDocEntry]) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    UI_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    catalog = [entry.to_dict() for entry in entries]
    (GENERATED_DIR / "model-catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n",
        encoding="utf-8",
    )
    (UI_PUBLIC_DIR / "model-catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    entries = collect_model_doc_entries(ignore_import_errors=False)
    write_catalog(entries)
    write_legacy_snippets(entries)
    write_model_pages(entries)
    write_model_indexes(entries)


if __name__ == "__main__":
    main()
