from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Literal

Task = Literal["tts", "stt", "sts", "vad"]
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_BLOB_MAIN_URL = "https://github.com/Blaizzy/mlx-audio/blob/main/"
CATALOG_EXCLUDED_PACKAGES = {
    "mlx_audio.tts.models.llama",
    "mlx_audio.tts.models.qwen3",
    "mlx_audio.stt.models.qwen3_forced_aligner",
    "mlx_audio.stt.models.wav2vec",
}


@dataclass(frozen=True)
class ModelDocEntry:
    slug: str
    name: str
    task: Task
    description: str
    repo: str | None
    docs_path: str | None = None
    languages: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    license: str | None = None
    pipeline_tag: str | None = None
    streaming: bool | None = None
    voice_cloning: bool | None = None
    timestamps: bool | None = None
    diarization: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_model_doc_entry(package_name: str) -> ModelDocEntry | None:
    entries = get_model_doc_entries(package_name)
    return entries[0] if entries else None


def get_model_doc_entries(package_name: str) -> list[ModelDocEntry]:
    package_path = _package_name_to_path(package_name)
    metadata_file = _find_metadata_file(package_path)
    if metadata_file is None:
        return []

    return [
        _resolve_doc_path(package_path, entry)
        for entry in _parse_model_doc_entry(metadata_file)
    ]


def iter_model_packages(categories: Iterable[Task] | None = None) -> Iterable[str]:
    root = Path(__file__).resolve().parent
    categories = categories or ("tts", "stt", "sts", "vad")

    for category in categories:
        models_dir = root / category / "models"
        if not models_dir.exists():
            continue

        for item in sorted(models_dir.iterdir()):
            if not item.is_dir() or item.name.startswith("__"):
                continue
            package_name = f"mlx_audio.{category}.models.{item.name}"
            if package_name in CATALOG_EXCLUDED_PACKAGES:
                continue
            yield package_name


def _package_name_to_path(package_name: str) -> Path:
    parts = package_name.split(".")
    if parts[:1] != ["mlx_audio"]:
        raise ValueError(f"Unsupported package name: {package_name}")
    return REPO_ROOT.joinpath(*parts)


def _find_metadata_file(package_path: Path) -> Path | None:
    if not package_path.exists():
        return None

    for file_path in sorted(package_path.glob("*.py")):
        if _has_model_doc_entry(file_path):
            return file_path

    return None


def _has_model_doc_entry(file_path: Path) -> bool:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == "DOCS"
                ):
                    return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "ModelConfig"
                    and target.attr == "DOCS"
                ):
                    return True
    return False


def _parse_doc_entries(node: ast.AST) -> list[ModelDocEntry]:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelDocEntry"
    ):
        kwargs = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        return [ModelDocEntry(**kwargs)]

    if isinstance(node, (ast.Tuple, ast.List)):
        entries: list[ModelDocEntry] = []
        for item in node.elts:
            entries.extend(_parse_doc_entries(item))
        return entries

    return []


def _parse_model_doc_entry(file_path: Path) -> list[ModelDocEntry]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if not (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == "DOCS"
                ):
                    continue
                entries = _parse_doc_entries(statement.value)
                if entries:
                    return entries

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "ModelConfig"
                    and target.attr == "DOCS"
                ):
                    continue
                entries = _parse_doc_entries(node.value)
                if entries:
                    return entries

    return []


def _resolve_doc_path(package_path: Path, entry: ModelDocEntry) -> ModelDocEntry:
    if entry.docs_path:
        return entry

    markdown_files = sorted(package_path.glob("*.md")) + sorted(
        package_path.glob("*.MD")
    )
    if len(markdown_files) != 1:
        return entry

    relative_path = markdown_files[0].relative_to(REPO_ROOT).as_posix()
    return replace(entry, docs_path=f"{REPO_BLOB_MAIN_URL}{relative_path}")


def collect_model_doc_entries(
    categories: Iterable[Task] | None = None,
    packages: Iterable[str] | None = None,
    ignore_import_errors: bool = True,
) -> list[ModelDocEntry]:
    entries: list[ModelDocEntry] = []
    package_names = packages or iter_model_packages(categories)

    for package_name in package_names:
        try:
            model_entries = get_model_doc_entries(package_name)
        except Exception:
            if ignore_import_errors:
                continue
            raise

        entries.extend(model_entries)

    return sorted(entries, key=lambda entry: (entry.task, entry.name.lower()))


__all__ = [
    "CATALOG_EXCLUDED_PACKAGES",
    "ModelDocEntry",
    "REPO_BLOB_MAIN_URL",
    "Task",
    "collect_model_doc_entries",
    "get_model_doc_entries",
    "get_model_doc_entry",
    "iter_model_packages",
]
