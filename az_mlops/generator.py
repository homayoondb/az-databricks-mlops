"""Template rendering and file generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape


TEMPLATES_DIR = Path(__file__).parent / "templates"

# Maps template path (relative to templates/) to output path (relative to project root).
# Use None as output path to mean "same as template path minus .j2 suffix".
CORE_TEMPLATES: list[str] = [
    ".gitignore.j2",
    "databricks.yml.j2",
    "resources/training-job.yml.j2",
    "resources/inference-job.yml.j2",
    "mlops/__init__.py.j2",
    "mlops/config.py.j2",
    "mlops/validation.py.j2",
    "mlops/deploy.py.j2",
    "mlops/run_validation.py.j2",
    "mlops/run_deploy.py.j2",
    ".github/workflows/ci.yml.j2",
    ".github/workflows/cd.yml.j2",
]

DQX_TEMPLATES: list[str] = [
    "resources/dqx-job.yml.j2",
    "mlops/dqx_checks.py.j2",
]


@dataclass(frozen=True)
class ProjectConfig:
    """All values needed to render templates."""

    project_name: str
    staging_workspace_url: str
    prod_workspace_url: str
    with_dqx: bool = False


def _output_path(template_name: str) -> str:
    """Strip the .j2 suffix to get the output file path."""
    if template_name.endswith(".j2"):
        return template_name[: -len(".j2")]
    return template_name


def render_templates(config: ProjectConfig) -> dict[str, str]:
    """Render all templates and return {relative_path: content}."""
    env = Environment(
        loader=PackageLoader("az_mlops", "templates"),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )

    templates = list(CORE_TEMPLATES)
    if config.with_dqx:
        templates.extend(DQX_TEMPLATES)

    results: dict[str, str] = {}
    for tmpl_name in templates:
        template = env.get_template(tmpl_name)
        rendered = template.render(
            project_name=config.project_name,
            staging_workspace_url=config.staging_workspace_url,
            prod_workspace_url=config.prod_workspace_url,
            with_dqx=config.with_dqx,
        )
        results[_output_path(tmpl_name)] = rendered

    return results


def write_files(
    output_dir: Path,
    rendered: dict[str, str],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write rendered templates to disk. Returns list of created file paths."""
    created: list[Path] = []
    for rel_path, content in rendered.items():
        dest = output_dir / rel_path
        if dest.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {dest}. Use --overwrite to replace."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        created.append(dest)
    return created
