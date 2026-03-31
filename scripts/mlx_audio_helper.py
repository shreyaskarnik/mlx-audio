from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    Console = None
    Panel = None
    Prompt = None
    Confirm = None
    Table = None
    HAS_RICH = False

try:
    import questionary

    HAS_QUESTIONARY = True
except ImportError:
    questionary = None
    HAS_QUESTIONARY = False

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_ROOT = REPO_ROOT / "output" / "helper-drafts"
MODEL_ROOT = REPO_ROOT / "mlx_audio"

TASK_CHOICES = ("tts", "stt", "sts", "vad")
CAPABILITY_CHOICES = ("streaming", "voice_cloning", "timestamps", "diarization")
TASK_DESCRIPTIONS = {
    "tts": "Text-to-speech model",
    "stt": "Speech-to-text model",
    "sts": "Speech-to-speech / enhancement / multimodal audio model",
    "vad": "Voice activity detection or diarization model",
}
CAPABILITY_DESCRIPTIONS = {
    "streaming": "Low-latency chunked inference or playback",
    "voice_cloning": "Reference-audio conditioning or cloned voices",
    "timestamps": "Word or segment timestamps",
    "diarization": "Speaker attribution / turn segmentation",
}
CONSOLE = Console() if HAS_RICH else None
STEP_TITLES = (
    ("1", "Model identity", "Name, task, and package naming"),
    ("2", "Source metadata", "Hub repo, upstream source, and languages"),
    ("3", "Capabilities", "User-facing behavior flags"),
    ("4", "Scaffold options", "README, conversion stub, and notes"),
    ("5", "Review", "Confirm the generated scaffold plan"),
)


@dataclass(frozen=True)
class ModelScaffoldAnswers:
    task: str
    slug: str
    package_name: str
    model_name: str
    hf_repo: str | None
    upstream_repo: str | None
    languages: tuple[str, ...]
    capabilities: tuple[str, ...]
    include_readme: bool
    add_convert_script: bool
    notes: str | None

    @property
    def package_dir(self) -> Path:
        return MODEL_ROOT / self.task / "models" / self.package_name

    @property
    def model_module(self) -> str:
        return self.package_name


WriteTarget = Literal["preview", "repo"]
OutputFormat = Literal["markdown", "json"]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_languages(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def normalize_capabilities(value: str) -> tuple[str, ...]:
    if not value.strip() or value.strip().lower() == "none":
        return ()
    requested = []
    for part in value.split(","):
        item = part.strip().lower()
        if item.isdigit():
            index = int(item) - 1
            if 0 <= index < len(CAPABILITY_CHOICES):
                item = CAPABILITY_CHOICES[index]
        if item and item not in requested:
            requested.append(item)
    invalid = [item for item in requested if item not in CAPABILITY_CHOICES]
    if invalid:
        raise ValueError(
            f"Unsupported capabilities: {', '.join(invalid)}. "
            f"Choose from: {', '.join(CAPABILITY_CHOICES)}"
        )
    return tuple(requested)


def prompt_text(prompt: str, default: str | None = None) -> str:
    if HAS_RICH:
        return Prompt.ask(prompt, default=default or "")
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def prompt_required_text(prompt: str) -> str:
    while True:
        if HAS_RICH:
            value = Prompt.ask(prompt, default="").strip()
        else:
            value = input(f"{prompt}: ").strip()
        if value:
            return value
        print_error("This field is required.")


def prompt_bool(prompt: str, default: bool) -> bool:
    if HAS_RICH:
        return Confirm.ask(prompt, default=default)
    default_label = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{default_label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def prompt_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    if HAS_QUESTIONARY:
        questionary_choices = [
            questionary.Choice(
                title=f"{choice}  ({TASK_DESCRIPTIONS.get(choice, '')})",
                value=choice,
            )
            for choice in choices
        ]
        selected = questionary.select(
            prompt,
            choices=questionary_choices,
            default=default,
            use_indicator=True,
        ).ask()
        if selected is None:
            return default
        return selected

    if HAS_RICH:
        table = Table(title=prompt, header_style="bold cyan")
        table.add_column("#", style="bold")
        table.add_column("Choice", style="bold green")
        table.add_column("Description")
        for index, choice in enumerate(choices, start=1):
            description = TASK_DESCRIPTIONS.get(choice, "")
            table.add_row(str(index), choice, description)
        CONSOLE.print(table)
    choice_labels = ", ".join(
        f"{index + 1}:{choice}" for index, choice in enumerate(choices)
    )
    while True:
        if HAS_RICH:
            value = Prompt.ask(
                f"{prompt} ({choice_labels})",
                default=default,
            ).strip().lower()
        else:
            value = input(f"{prompt} [{default}] ({choice_labels}): ").strip().lower()
        if not value:
            return default
        if value in choices:
            return value
        if value.isdigit():
            index = int(value) - 1
            if 0 <= index < len(choices):
                return choices[index]
        print_error(f"Choose one of: {', '.join(choices)}")


def prompt_capabilities(default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if HAS_QUESTIONARY:
        choices = [
            questionary.Choice(
                title=f"{capability}  ({CAPABILITY_DESCRIPTIONS.get(capability, '')})",
                value=capability,
                checked=capability in default,
            )
            for capability in CAPABILITY_CHOICES
        ]
        selected = questionary.checkbox(
            "Select capabilities",
            choices=choices,
            instruction="Use arrows to move, space to toggle, enter to confirm",
        ).ask()
        if selected is None:
            return default
        return tuple(selected)

    default_text = ", ".join(default) or "none"
    if HAS_RICH:
        table = Table(title="Capabilities", header_style="bold cyan")
        table.add_column("#", style="bold")
        table.add_column("Capability", style="bold green")
        table.add_column("Meaning")
        for index, capability in enumerate(CAPABILITY_CHOICES, start=1):
            table.add_row(
                str(index),
                capability,
                CAPABILITY_DESCRIPTIONS.get(capability, ""),
            )
        CONSOLE.print(table)
        CONSOLE.print(
            "[dim]Enter comma-separated numbers or names. Leave blank for none.[/dim]"
        )
    else:
        print("--capabilities")
        for index, capability in enumerate(CAPABILITY_CHOICES, start=1):
            print(f"  {index}. {capability}")
        print("  Enter comma-separated numbers or names, or leave blank for none.")
    while True:
        if HAS_RICH:
            raw = Prompt.ask("Selection", default=default_text).strip()
        else:
            raw = input(f"Selection [{default_text}]: ").strip()
        try:
            return normalize_capabilities(raw or default_text)
        except ValueError as exc:
            print_error(str(exc))


def print_error(message: str) -> None:
    if HAS_RICH:
        CONSOLE.print(f"[bold red]{message}[/bold red]")
    else:
        print(message)


def print_info(message: str) -> None:
    if HAS_RICH:
        CONSOLE.print(f"[bold cyan]{message}[/bold cyan]")
    else:
        print(message)


def render_step(step: str, title: str, subtitle: str) -> None:
    if HAS_RICH:
        CONSOLE.print(
            Panel.fit(
                f"[bold cyan]Step {step}[/bold cyan] [bold]{title}[/bold]\n"
                f"[dim]{subtitle}[/dim]",
                border_style="cyan",
            )
        )
    else:
        print(f"\nStep {step}: {title}")
        print(subtitle)


def render_wizard_intro() -> None:
    if not HAS_RICH:
        return
    CONSOLE.print(
        Panel.fit(
            "[bold]mlx-audio-helper[/bold]\n"
            "Draft wizard for scaffolding a new model package.\n\n"
            "[dim]This flow creates a package skeleton, README, and optional convert stub.[/dim]",
            border_style="cyan",
        )
    )


def render_summary(answers: ModelScaffoldAnswers) -> None:
    if not HAS_RICH:
        print("\nScaffold Summary")
        print(f"- Task: {answers.task}")
        print(f"- Model name: {answers.model_name}")
        print(f"- Slug: {answers.slug}")
        print(f"- Package: {answers.package_name}")
        print(f"- HF repo: {answers.hf_repo or '--'}")
        print(f"- Languages: {', '.join(answers.languages) or '--'}")
        print(f"- Capabilities: {', '.join(answers.capabilities) or 'none'}")
        print(f"- README: {'yes' if answers.include_readme else 'no'}")
        print(f"- convert.py: {'yes' if answers.add_convert_script else 'no'}")
        return
    table = Table(title="Scaffold Summary", header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Task", answers.task)
    table.add_row("Model name", answers.model_name)
    table.add_row("Slug", answers.slug)
    table.add_row("Package", answers.package_name)
    table.add_row("HF repo", answers.hf_repo or "--")
    table.add_row("Languages", ", ".join(answers.languages) or "--")
    table.add_row("Capabilities", ", ".join(answers.capabilities) or "none")
    table.add_row("README", "yes" if answers.include_readme else "no")
    table.add_row("convert.py", "yes" if answers.add_convert_script else "no")
    CONSOLE.print(table)
    files = Table(title="Files To Generate", header_style="bold cyan")
    files.add_column("Path")
    for relative_path in file_plan(answers):
        files.add_row(relative_path)
    CONSOLE.print(files)


def confirm_action(prompt: str, default: bool = True) -> bool:
    return prompt_bool(prompt, default=default)


def model_task_roots() -> list[Path]:
    return [MODEL_ROOT / task / "models" for task in TASK_CHOICES]


def existing_model_packages() -> list[Path]:
    packages: list[Path] = []
    for root in model_task_roots():
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir() and not path.name.startswith("__"):
                packages.append(path)
    return packages


def repo_collision_paths(hf_repo: str) -> list[Path]:
    if not hf_repo.strip():
        return []
    collisions: list[Path] = []
    quoted_repo = re.escape(hf_repo)
    repo_pattern = re.compile(quoted_repo)
    for package_dir in existing_model_packages():
        for file_path in package_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in {".py", ".md"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if repo_pattern.search(text):
                collisions.append(file_path.relative_to(REPO_ROOT))
    return collisions


def validate_answers(answers: ModelScaffoldAnswers) -> None:
    package_dir = answers.package_dir
    if package_dir.exists():
        raise SystemExit(
            f"Model package already exists: {package_dir.relative_to(REPO_ROOT)}"
        )

    same_name = [
        path.relative_to(REPO_ROOT)
        for path in existing_model_packages()
        if path.name == answers.package_name
    ]
    if same_name:
        raise SystemExit(
            "A model package with the same name already exists elsewhere: "
            + ", ".join(str(path) for path in same_name)
        )

    repo_collisions = repo_collision_paths(answers.hf_repo or "")
    if answers.hf_repo and repo_collisions:
        raise SystemExit(
            "This Hugging Face repo already appears in the codebase: "
            + ", ".join(str(path) for path in repo_collisions[:6])
        )


def ensure_preview_root(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Preview output already exists: {path}. Use --force to replace it."
        )


def collect_answers(args: argparse.Namespace) -> ModelScaffoldAnswers:
    interactive = args.interactive
    if interactive:
        render_wizard_intro()

    def need_text(current: str | None, prompt: str, default: str | None = None) -> str:
        if current:
            return current
        if default is not None:
            return default
        if not interactive:
            raise SystemExit(f"Missing required argument: {prompt}")
        return prompt_text(prompt, default=default)

    if args.task:
        task = args.task
    elif interactive:
        render_step(*STEP_TITLES[0])
        task = prompt_choice("--task", TASK_CHOICES, default="tts")
    else:
        task = need_text(args.task, "--task", default="tts")

    if task not in TASK_CHOICES:
        raise SystemExit(f"--task must be one of: {', '.join(TASK_CHOICES)}")

    if args.model_name:
        model_name = args.model_name
    elif interactive:
        if not args.task:
            pass
        model_name = prompt_required_text("--model-name")
    else:
        model_name = need_text(args.model_name, "--model-name")
    slug = need_text(args.slug, "--slug", default=slugify(model_name).replace("_", "-"))
    package_name = need_text(args.package_name, "--package-name", default=slugify(slug))
    hf_repo = args.hf_repo
    if interactive and hf_repo is None:
        render_step(*STEP_TITLES[1])
        hf_repo = prompt_text("--hf-repo", default="")
    hf_repo = hf_repo or None

    upstream_repo = args.upstream_repo
    if interactive and upstream_repo is None:
        upstream_repo = prompt_text("--upstream-repo", default="")
    upstream_repo = upstream_repo or None

    languages_raw = args.languages
    if interactive and languages_raw is None:
        languages_raw = prompt_text(
            "--languages (comma-separated language codes)", default="en"
        )
    languages = normalize_languages(languages_raw or "")

    capabilities_raw = args.capabilities
    if interactive and capabilities_raw is None:
        render_step(*STEP_TITLES[2])
        capabilities = prompt_capabilities()
    else:
        capabilities = normalize_capabilities(capabilities_raw or "")

    if interactive:
        render_step(*STEP_TITLES[3])
        include_readme = (
            args.include_readme
            if args.include_readme is not None
            else prompt_bool("Create co-located README.md?", default=True)
        )
        add_convert_script = (
            args.add_convert_script
            if args.add_convert_script is not None
            else prompt_bool("Include convert.py stub?", default=False)
        )
        notes = args.notes if args.notes is not None else prompt_text("Notes", default="")
    else:
        include_readme = True if args.include_readme is None else args.include_readme
        add_convert_script = (
            False if args.add_convert_script is None else args.add_convert_script
        )
        notes = args.notes

    return ModelScaffoldAnswers(
        task=task,
        slug=slug,
        package_name=package_name,
        model_name=model_name,
        hf_repo=hf_repo,
        upstream_repo=upstream_repo,
        languages=languages,
        capabilities=capabilities,
        include_readme=include_readme,
        add_convert_script=add_convert_script,
        notes=notes or None,
    )


def render_init_py(answers: ModelScaffoldAnswers) -> str:
    exports = '["Model", "ModelConfig"]'
    return f"""from .{answers.model_module} import Model, ModelConfig

__all__ = {exports}
"""


def render_model_py(answers: ModelScaffoldAnswers) -> str:
    return f'''"""Draft scaffold for {answers.model_name}."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_type: str = "{answers.package_name}"


class Model:
    def __init__(self, config: ModelConfig):
        self.config = config

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        raise NotImplementedError("TODO: load weights and config")

    def generate(self, *args, **kwargs):
        raise NotImplementedError("TODO: implement runtime")
'''


def render_readme(answers: ModelScaffoldAnswers) -> str:
    language_text = ", ".join(code.upper() for code in answers.languages) or "TODO"
    capabilities = ", ".join(
        capability.replace("_", " ") for capability in answers.capabilities
    ) or "TODO"
    hf_repo = f"`{answers.hf_repo}`" if answers.hf_repo else "`TODO`"
    upstream = (
        f"- Upstream repo: `{answers.upstream_repo}`\n" if answers.upstream_repo else ""
    )
    notes = f"\n## Notes\n\n{answers.notes}\n" if answers.notes else ""
    return f"""# {answers.model_name}

## Summary

- Task: `{answers.task}`
- Hugging Face repo: {hf_repo}
{upstream}- Languages: {language_text}
- Capabilities: {capabilities}

## Status

- [ ] Model loading wired up
- [ ] Generation / transcription path implemented
- [ ] README examples validated
- [ ] Tests added

## Usage

```python
# TODO: replace with a real example once the model is implemented
```
{notes}"""


def render_convert_py() -> str:
    return '''"""Draft conversion stub for a new model."""


def main() -> None:
    raise NotImplementedError("TODO: add conversion flow if this model needs one")


if __name__ == "__main__":
    main()
'''


def file_plan(answers: ModelScaffoldAnswers) -> dict[str, str]:
    relative_base = Path("mlx_audio") / answers.task / "models" / answers.package_name
    files = {
        str(relative_base / "__init__.py"): render_init_py(answers),
        str(relative_base / f"{answers.model_module}.py"): render_model_py(answers),
    }
    if answers.include_readme:
        files[str(relative_base / "README.md")] = render_readme(answers)
    if answers.add_convert_script:
        files[str(relative_base / "convert.py")] = render_convert_py()
    return files


def render_plan_markdown(answers: ModelScaffoldAnswers) -> str:
    capabilities = ", ".join(answers.capabilities) or "none"
    languages = ", ".join(answers.languages) or "none"
    hf_repo = answers.hf_repo or "--"
    lines = [
        f"# Draft scaffold for {answers.model_name}",
        "",
        "## Answers",
        "",
        f"- Task: `{answers.task}`",
        f"- Slug: `{answers.slug}`",
        f"- Package: `{answers.package_name}`",
        f"- Hugging Face repo: `{hf_repo}`",
        f"- Upstream repo: `{answers.upstream_repo or '--'}`",
        f"- Languages: `{languages}`",
        f"- Capabilities: `{capabilities}`",
        "",
        "## Proposed files",
        "",
    ]
    for relative_path in file_plan(answers):
        lines.append(f"- `{relative_path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a draft scaffold, not a full implementation.",
            "- It is intentionally limited to package scaffolding and contributor workflow.",
            "- A Codex skill could wrap this flow and answer the prompts from repo context instead of raw CLI flags.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_plan_json(answers: ModelScaffoldAnswers) -> str:
    payload = {
        "answers": asdict(answers),
        "files": sorted(file_plan(answers)),
    }
    return json.dumps(payload, indent=2) + "\n"


def write_preview(answers: ModelScaffoldAnswers, output_root: Path, force: bool) -> Path:
    preview_dir = output_root / answers.task / answers.package_name
    ensure_preview_root(preview_dir, force=force)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in file_plan(answers).items():
        path = preview_dir / Path(relative_path).name
        path.write_text(content, encoding="utf-8")
    (preview_dir / "plan.md").write_text(render_plan_markdown(answers), encoding="utf-8")
    (preview_dir / "answers.json").write_text(
        json.dumps(asdict(answers), indent=2) + "\n", encoding="utf-8"
    )
    return preview_dir


def write_repo_scaffold(answers: ModelScaffoldAnswers) -> list[Path]:
    written_paths: list[Path] = []
    for relative_path, content in file_plan(answers).items():
        path = REPO_ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written_paths.append(path)
    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft helper for scaffolding a new MLX Audio model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_model = subparsers.add_parser(
        "add-model",
        help="Collect scaffold inputs and emit a draft model package plan.",
    )
    add_model.add_argument("--task", choices=TASK_CHOICES)
    add_model.add_argument("--slug")
    add_model.add_argument("--package-name")
    add_model.add_argument("--model-name")
    add_model.add_argument("--hf-repo")
    add_model.add_argument("--upstream-repo")
    add_model.add_argument("--languages")
    add_model.add_argument("--capabilities")
    add_model.add_argument("--notes")
    add_model.add_argument(
        "--output-format",
        choices=("markdown", "json"),
        default="markdown",
        help="Render the scaffold plan as Markdown or JSON.",
    )
    add_model.add_argument(
        "--write-target",
        choices=("preview", "repo"),
        default=None,
        help="Write the scaffold to output/helper-drafts/ or directly into the repo.",
    )
    add_model.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing values interactively.",
    )
    add_model.add_argument(
        "--include-readme",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    add_model.add_argument(
        "--add-convert-script",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    add_model.add_argument(
        "--write-preview",
        action="store_true",
        help="Deprecated alias for --write-target preview.",
    )
    add_model.add_argument(
        "--output-root",
        default=str(PREVIEW_ROOT),
        help="Directory used when --write-target preview is selected.",
    )
    add_model.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing preview scaffold directory.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command != "add-model":
        raise SystemExit(f"Unsupported command: {args.command}")

    answers = collect_answers(args)
    validate_answers(answers)
    if args.interactive:
        render_step(*STEP_TITLES[4])
        render_summary(answers)
    if args.output_format == "json":
        print(render_plan_json(answers), end="")
    else:
        print(render_plan_markdown(answers))

    write_target: WriteTarget | None = args.write_target
    if args.write_preview and write_target is None:
        write_target = "preview"

    if write_target == "preview":
        if args.interactive and not confirm_action("Write preview scaffold?", default=True):
            print_info("Cancelled before writing preview scaffold.")
            return
        output_root = Path(args.output_root).resolve()
        preview_dir = write_preview(answers, output_root=output_root, force=args.force)
        print_info(f"Preview scaffold written to: {preview_dir}")
    elif write_target == "repo":
        if args.interactive and not confirm_action("Write scaffold into the repo?", default=False):
            print_info("Cancelled before writing scaffold into the repo.")
            return
        written_paths = write_repo_scaffold(answers)
        print_info("Scaffold written to repo:")
        for path in written_paths:
            print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
