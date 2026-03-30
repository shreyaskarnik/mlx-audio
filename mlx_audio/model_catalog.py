from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

Task = Literal["tts", "stt", "sts", "vad"]


@dataclass(frozen=True)
class ModelDocEntry:
    slug: str
    name: str
    task: Task
    description: str
    repo: str
    docs_path: str
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
    package_path = _package_name_to_path(package_name)
    metadata_file = _find_metadata_file(package_path)
    if metadata_file is None:
        return None

    return _parse_model_doc_entry(metadata_file)


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
            yield f"mlx_audio.{category}.models.{item.name}"


def _package_name_to_path(package_name: str) -> Path:
    root = Path(__file__).resolve().parent
    parts = package_name.split(".")
    if parts[:1] != ["mlx_audio"]:
        raise ValueError(f"Unsupported package name: {package_name}")
    return root.joinpath(*parts[1:])


def _find_metadata_file(package_path: Path) -> Path | None:
    if not package_path.exists():
        return None

    for file_path in sorted(package_path.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        if _has_model_doc_entry(file_path):
            return file_path

    return None


def _has_model_doc_entry(file_path: Path) -> bool:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "DOCS"
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "ModelDocEntry"
            ):
                return True
    return False


def _parse_model_doc_entry(file_path: Path) -> ModelDocEntry | None:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        for statement in node.body:
            if not (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "DOCS"
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "ModelDocEntry"
            ):
                continue

            kwargs = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in statement.value.keywords
                if keyword.arg is not None
            }
            return ModelDocEntry(**kwargs)

    return None


def collect_model_doc_entries(
    categories: Iterable[Task] | None = None,
    packages: Iterable[str] | None = None,
    ignore_import_errors: bool = True,
) -> list[ModelDocEntry]:
    entries: list[ModelDocEntry] = []
    package_names = packages or iter_model_packages(categories)

    for package_name in package_names:
        try:
            entry = get_model_doc_entry(package_name)
        except Exception:
            if ignore_import_errors:
                continue
            raise

        if entry is not None:
            entries.append(entry)

    return sorted(entries, key=lambda entry: (entry.task, entry.name.lower()))


__all__ = [
    "ModelDocEntry",
    "Task",
    "collect_model_doc_entries",
    "get_model_doc_entry",
    "iter_model_packages",
]
