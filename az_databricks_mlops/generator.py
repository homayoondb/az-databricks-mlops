"""Template rendering and file generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape


CORE_TEMPLATES: list[str] = [
    ".gitignore.j2",
    "databricks.yml.j2",
    "resources/training-job.yml.j2",
    "mlops/__init__.py.j2",
    "mlops/config.py.j2",
    "mlops/run_training.py.j2",
    "mlops/validation.py.j2",
    "mlops/run_validation.py.j2",
    "mlops/deploy.py.j2",
    "mlops/run_deploy.py.j2",
    "notebooks/run_pipeline.py.j2",
]

INFERENCE_TEMPLATES: list[str] = [
    "resources/inference-job.yml.j2",
    "mlops/run_inference.py.j2",
]

DQX_TEMPLATES: list[str] = [
    "resources/dqx-job.yml.j2",
    "mlops/dqx_checks.py.j2",
]

LLMOPS_TEMPLATES: list[str] = [
    ".gitignore.j2",
    "databricks.yml.j2",
    "resources/agent-job.yml.j2",
    "llmops/__init__.py.j2",
    "llmops/config.py.j2",
    "llmops/run_agent_dev.py.j2",
    "llmops/scorers.py.j2",
    "llmops/run_agent_eval.py.j2",
    "llmops/deploy.py.j2",
    "llmops/run_agent_deploy.py.j2",
    "notebooks/run_agent_pipeline.py.j2",
]

LLMOPS_SERVE_TEMPLATES: list[str] = [
    "resources/agent-serve-job.yml.j2",
    "llmops/run_agent_serve.py.j2",
]


@dataclass(frozen=True)
class ProjectConfig:
    """All values needed to render templates."""

    project_name: str
    staging_workspace_url: str
    project_type: str = "classic_ml"
    catalog_name: str = ""
    schema_name: str = ""
    prod_workspace_url: str = ""
    # Classic ML fields
    training_notebook: str = "train.py"
    with_inference: bool = True
    with_dqx: bool = False
    # LLMOps fields (ignored when project_type == "classic_ml")
    agent_script: str = "agent.py"
    with_serving: bool = True


def _output_path(template_name: str) -> str:
    """Strip the .j2 suffix to get the output file path."""
    if template_name.endswith(".j2"):
        return template_name[: -len(".j2")]
    return template_name


def render_templates(config: ProjectConfig) -> dict[str, str]:
    """Render all templates and return {relative_path: content}."""
    env = Environment(
        loader=PackageLoader("az_databricks_mlops", "templates"),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )

    if config.project_type == "llmops":
        templates = list(LLMOPS_TEMPLATES)
        if config.with_serving:
            templates.extend(LLMOPS_SERVE_TEMPLATES)
    else:
        templates = list(CORE_TEMPLATES)
        if config.with_inference:
            templates.extend(INFERENCE_TEMPLATES)
        if config.with_dqx:
            templates.extend(DQX_TEMPLATES)

    context = {
        "project_name": config.project_name,
        "project_type": config.project_type,
        "staging_workspace_url": config.staging_workspace_url,
        "catalog_name": config.catalog_name,
        "schema_name": config.schema_name,
        "prod_workspace_url": config.prod_workspace_url,
        "training_notebook": config.training_notebook,
        "with_inference": config.with_inference,
        "with_dqx": config.with_dqx,
        "agent_script": config.agent_script,
        "with_serving": config.with_serving,
    }

    results: dict[str, str] = {}
    for tmpl_name in templates:
        template = env.get_template(tmpl_name)
        rendered = template.render(**context)
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


def find_notebooks(directory: Path) -> list[str]:
    """Find Python and Jupyter notebook files that look like training scripts."""
    patterns = ["**/*.py", "**/*.ipynb"]
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "mlops", "llmops"}
    # Generated adm file names that should not appear as training/agent script options
    skip_names = {
        "run_pipeline.py", "run_training.py", "run_validation.py", "run_deploy.py", "run_inference.py",
        "run_agent_pipeline.py", "run_agent_dev.py", "run_agent_eval.py", "run_agent_deploy.py", "run_agent_serve.py",
    }
    results: list[str] = []

    for pattern in patterns:
        for path in directory.glob(pattern):
            parts = path.relative_to(directory).parts
            if any(p in skip_dirs or p.startswith(".") for p in parts[:-1]):
                continue
            if path.name in skip_names:
                continue
            results.append(str(path.relative_to(directory)))

    return sorted(results)
