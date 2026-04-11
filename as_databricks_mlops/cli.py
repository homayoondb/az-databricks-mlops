"""CLI entry point for adm."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
import urllib.parse
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import click
import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied, ResourceAlreadyExists, ResourceDoesNotExist

from as_databricks_mlops import __version__
from as_databricks_mlops.generator import (
    CORE_TEMPLATES,
    DQX_TEMPLATES,
    INFERENCE_TEMPLATES,
    ProjectConfig,
    _output_path,
    find_notebooks,
    render_templates,
    write_files,
)
from as_databricks_mlops.review import review_repository


DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = (
    "adm.yml",
    "adm.yaml",
    ".adm.yml",
    ".adm.yaml",
)


@dataclass(frozen=True)
class CliDefaults:
    """Resolved YAML defaults for interactive CLI commands."""

    project_name: str | None = None
    staging_url: str | None = None
    prod_url: str | None = None
    catalog_name: str | None = None
    schema_name: str | None = None
    training_notebook: str | None = None
    ignored_inference_notebook: str | None = None
    with_inference: bool | None = None
    with_dqx: bool | None = None


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Lightweight MLOps scaffolding for Databricks projects."""


def _resolve_notebook_choice(choice: str, notebooks: list[str], directory: Path) -> str | None:
    """Resolve a notebook prompt response to a valid relative path."""
    try:
        idx = int(choice)
        if 1 <= idx <= len(notebooks):
            return notebooks[idx - 1]
    except ValueError:
        pass

    candidate = directory / choice
    if candidate.is_file():
        try:
            return str(candidate.relative_to(directory))
        except ValueError:
            return choice
    return None


def _prompt_notebook(label: str, directory: Path, default: str | None) -> str:
    """Prompt the user to select or type a notebook path, showing discovered files."""
    notebooks = find_notebooks(directory)
    if notebooks:
        click.echo(f"\n  Found notebooks/scripts in your project:")
        for i, nb in enumerate(notebooks, 1):
            click.echo(f"    {i}. {nb}")
        click.echo()
        while True:
            if default is not None:
                choice = click.prompt(
                    f"  {label} (number or path)",
                    default=default,
                )
            else:
                choice = click.prompt(f"  {label} (number or path)")
            resolved = _resolve_notebook_choice(choice, notebooks, directory)
            if resolved is not None:
                return resolved
            click.echo("  Enter a listed number or an existing notebook/script path.", err=True)
    if default is not None:
        return click.prompt(f"  {label}", default=default)
    return click.prompt(f"  {label}")


def _prompt_text(label: str, configured_default: str | None, fallback_default: str | None = None) -> str:
    """Prompt for a required text value, preferring YAML defaults over detected ones."""
    if configured_default is not None:
        return click.prompt(label, default=configured_default, show_default=True)
    if fallback_default is not None:
        return click.prompt(label, default=fallback_default, show_default=bool(fallback_default))
    return click.prompt(label)


def _prompt_optional_text(
    label: str,
    configured_default: str | None,
    fallback_default: str | None = None,
) -> str:
    """Prompt for an optional text value, preferring YAML defaults over detected ones."""
    if configured_default is not None:
        return click.prompt(label, default=configured_default, show_default=bool(configured_default))
    if fallback_default is not None:
        return click.prompt(label, default=fallback_default, show_default=bool(fallback_default))
    return click.prompt(label, default="", show_default=False)


def _prompt_bool(label: str, configured_default: bool | None, fallback_default: bool | None = None) -> bool:
    """Prompt for a boolean value, preferring YAML defaults over detected ones."""
    default_value = configured_default if configured_default is not None else fallback_default
    prompt_label = f"{label} [y/n]"
    default = None
    if default_value is True:
        prompt_label = f"{label} [Y/n]"
        default = "y"
    elif default_value is False:
        prompt_label = f"{label} [y/N]"
        default = "n"

    choice = click.prompt(
        prompt_label,
        default=default,
        show_default=False,
        show_choices=False,
        type=click.Choice(["y", "n"], case_sensitive=False),
    )
    return choice.lower() == "y"


def _warn_ignored_inference_notebook(source: str) -> None:
    """Explain that custom inference scripts are not wired into the generated scaffold."""
    click.echo(
        "Warning: custom inference notebooks/scripts are not supported yet; "
        f"ignoring {source}. The generated batch inference job always uses mlops/run_inference.py.",
        err=True,
    )


def _sanitize_url(value: str) -> str:
    """Strip control/escape characters; return empty string if not a valid URL."""
    # Remove ANSI escape sequences (e.g. arrow keys: \x1b[C) and other control chars
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value).strip()
    return clean if clean.startswith("http") else ""


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Validate that a parsed YAML field is a mapping."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise click.ClickException(
            f"Invalid config: '{field_name}' must be a mapping in adm.yml."
        )
    return value


def _as_optional_str(value: Any, field_name: str) -> str | None:
    """Validate an optional string value loaded from YAML."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise click.ClickException(
            f"Invalid config: '{field_name}' must be a string in adm.yml."
        )
    return value


def _as_optional_bool(value: Any, field_name: str) -> bool | None:
    """Validate an optional boolean value loaded from YAML."""
    if value is None:
        return None
    if not isinstance(value, bool):
        raise click.ClickException(
            f"Invalid config: '{field_name}' must be true or false in adm.yml."
        )
    return value


def _find_defaults_file(directory: Path, config_path: Path | None) -> Path | None:
    """Resolve an explicit config path or discover the nearest adm.yml."""
    if config_path is not None:
        resolved = config_path.expanduser()
        if not resolved.is_absolute():
            resolved = directory / resolved
        if not resolved.exists():
            raise click.ClickException(f"Config file not found: {resolved}")
        return resolved

    for current in [directory, *directory.parents]:
        for filename in DEFAULT_CONFIG_FILENAMES:
            candidate = current / filename
            if candidate.is_file():
                return candidate
    return None


def _display_path(path: Path, directory: Path) -> str:
    """Format a path for CLI output."""
    try:
        return str(path.relative_to(directory))
    except ValueError:
        return str(path)


def _load_cli_defaults(directory: Path, config_path: Path | None) -> tuple[CliDefaults, Path | None]:
    """Load YAML defaults for init/new commands."""
    resolved_path = _find_defaults_file(directory, config_path)
    if resolved_path is None:
        return CliDefaults(), None

    try:
        parsed = yaml.safe_load(resolved_path.read_text()) or {}
    except yaml.YAMLError as e:
        raise click.ClickException(f"Failed to parse {_display_path(resolved_path, directory)}: {e}") from e

    if not isinstance(parsed, dict):
        raise click.ClickException("Invalid config: adm.yml must contain a top-level mapping.")

    databricks_cfg = _as_mapping(parsed.get("databricks"), "databricks")
    training_cfg = _as_mapping(parsed.get("training"), "training")
    options_cfg = _as_mapping(parsed.get("options"), "options")

    with_inference = _as_optional_bool(
        training_cfg.get("with_inference", parsed.get("with_inference")),
        "training.with_inference",
    )
    skip_inference = _as_optional_bool(
        training_cfg.get("skip_inference", parsed.get("skip_inference")),
        "training.skip_inference",
    )
    if skip_inference is not None:
        with_inference = not skip_inference

    defaults = CliDefaults(
        project_name=_as_optional_str(parsed.get("project_name"), "project_name"),
        staging_url=_as_optional_str(
            databricks_cfg.get("staging_url", parsed.get("staging_url")),
            "databricks.staging_url",
        ),
        prod_url=_as_optional_str(
            databricks_cfg.get("prod_url", parsed.get("prod_url")),
            "databricks.prod_url",
        ),
        catalog_name=_as_optional_str(
            databricks_cfg.get("catalog_name", parsed.get("catalog_name")),
            "databricks.catalog_name",
        ),
        schema_name=_as_optional_str(
            databricks_cfg.get("schema_name", parsed.get("schema_name")),
            "databricks.schema_name",
        ),
        training_notebook=_as_optional_str(
            training_cfg.get("training_notebook", parsed.get("training_notebook")),
            "training.training_notebook",
        ),
        ignored_inference_notebook=_as_optional_str(
            training_cfg.get("inference_notebook", parsed.get("inference_notebook")),
            "training.inference_notebook",
        ),
        with_inference=with_inference,
        with_dqx=_as_optional_bool(
            options_cfg.get("with_dqx", parsed.get("with_dqx")),
            "options.with_dqx",
        ),
    )
    return defaults, resolved_path


def _detect_staging_url(directory: Path) -> str:
    """Try to detect staging workspace URL from existing databricks.yml."""
    databricks_yml = directory / "databricks.yml"
    if not databricks_yml.exists():
        return ""
    try:
        bundle = yaml.safe_load(databricks_yml.read_text())
        return bundle["targets"]["staging"]["workspace"]["host"]
    except (yaml.YAMLError, KeyError, TypeError):
        return ""


def _detect_prod_url(directory: Path) -> str:
    """Try to detect prod workspace URL from existing databricks.yml."""
    databricks_yml = directory / "databricks.yml"
    if not databricks_yml.exists():
        return ""
    try:
        bundle = yaml.safe_load(databricks_yml.read_text())
        return bundle["targets"]["prod"]["workspace"]["host"]
    except (yaml.YAMLError, KeyError, TypeError):
        return ""


def _detect_catalog_name(directory: Path) -> str:
    """Try to detect catalog name from existing databricks.yml."""
    databricks_yml = directory / "databricks.yml"
    if not databricks_yml.exists():
        return ""
    try:
        bundle = yaml.safe_load(databricks_yml.read_text())
        variable_default = (
            bundle.get("variables", {})
            .get("catalog_name", {})
            .get("default", "")
        )
        if variable_default:
            return variable_default
        for target in ("dev", "staging", "prod"):
            target_default = (
                bundle.get("targets", {})
                .get(target, {})
                .get("variables", {})
                .get("catalog_name", "")
            )
            if target_default:
                return target_default
        return ""
    except (yaml.YAMLError, KeyError, TypeError):
        return ""


def _detect_schema_name(directory: Path) -> str:
    """Try to detect schema name from existing databricks.yml."""
    databricks_yml = directory / "databricks.yml"
    if not databricks_yml.exists():
        return ""
    try:
        bundle = yaml.safe_load(databricks_yml.read_text())
        return bundle["variables"]["schema_name"]["default"]
    except (yaml.YAMLError, KeyError, TypeError):
        return ""


@cli.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to an adm YAML defaults file. Auto-discovers adm.yml if omitted.",
)
@click.option(
    "--project-name",
    default=None,
    help="Name for the ML project.",
)
@click.option(
    "--staging-url",
    default=None,
    help="Databricks staging workspace URL.",
)
@click.option(
    "--prod-url",
    default=None,
    help="Databricks production workspace URL. Optional.",
)
@click.option(
    "--catalog-name",
    default=None,
    help="Unity Catalog catalog name (e.g. us_comm_lakehouse_dev).",
)
@click.option(
    "--schema-name",
    default=None,
    help="Unity Catalog schema (database) where models will be registered.",
)
@click.option(
    "--training-notebook",
    default=None,
    help="Path to your training notebook/script (relative to project root).",
)
@click.option(
    "--inference-notebook",
    default=None,
    hidden=True,
)
@click.option(
    "--inference/--skip-inference",
    "with_inference",
    default=None,
    help="Generate the batch inference job scaffold.",
)
@click.option(
    "--with-dqx/--without-dqx",
    default=None,
    help="Include the optional DQX data quality scaffold.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
@click.option(
    "--no-validate",
    is_flag=True,
    default=False,
    help="Skip `databricks bundle validate` (useful in notebook environments).",
)
def init(
    config_path: Path | None,
    project_name: str | None,
    staging_url: str | None,
    prod_url: str | None,
    catalog_name: str | None,
    schema_name: str | None,
    training_notebook: str | None,
    inference_notebook: str | None,
    with_inference: bool | None,
    with_dqx: bool | None,
    overwrite: bool,
    no_validate: bool,
) -> None:
    """Add MLOps scaffolding to an existing project."""
    cwd = Path.cwd()
    defaults, defaults_path = _load_cli_defaults(cwd, config_path)
    if defaults_path is not None:
        click.echo(f"Using defaults from {_display_path(defaults_path, cwd)}")

    if project_name is None:
        project_name = _prompt_text("Project name", defaults.project_name, cwd.name)

    if staging_url is None:
        detected = _detect_staging_url(cwd) or None
        staging_url = _prompt_text("Staging workspace URL", defaults.staging_url, detected)

    if prod_url is None:
        detected = _detect_prod_url(cwd) or None
        prod_url = _prompt_optional_text("Prod workspace URL (enter to skip)", defaults.prod_url, detected)

    if catalog_name is None:
        detected = _detect_catalog_name(cwd) or None
        catalog_name = _prompt_text("Unity Catalog name", defaults.catalog_name, detected)

    if schema_name is None:
        detected = _detect_schema_name(cwd) or None
        schema_name = _prompt_text("Unity Catalog schema (database)", defaults.schema_name, detected)

    if training_notebook is None:
        training_notebook = _prompt_notebook(
            "Training notebook/script",
            cwd,
            default=defaults.training_notebook or "train.py",
        )

    if inference_notebook is not None:
        _warn_ignored_inference_notebook("--inference-notebook")
    elif defaults.ignored_inference_notebook is not None:
        _warn_ignored_inference_notebook("training.inference_notebook in adm.yml")

    if with_inference is None:
        with_inference = _prompt_bool("  Include batch inference job?", defaults.with_inference, True)

    if with_dqx is None:
        with_dqx = defaults.with_dqx if defaults.with_dqx is not None else False

    config = ProjectConfig(
        project_name=project_name,
        staging_workspace_url=_sanitize_url(staging_url),
        catalog_name=catalog_name,
        schema_name=schema_name,
        prod_workspace_url=_sanitize_url(prod_url) if prod_url else "",
        training_notebook=training_notebook,
        with_inference=with_inference,
        with_dqx=with_dqx,
    )
    rendered = render_templates(config)
    try:
        created = write_files(cwd, rendered, overwrite=overwrite)
    except FileExistsError as e:
        raise click.ClickException(f"{e} Run `adm clean` first, or pass `--overwrite`.") from e

    click.echo()
    for path in created:
        click.echo(f"  Created {path.relative_to(cwd)}")
    click.echo()
    if no_validate:
        click.echo("Skipping bundle validation (--no-validate).")
        valid = True
    else:
        valid = _run_bundle_validate(cwd)

    if valid:
        click.echo()
        try:
            if click.confirm("  Run the training job now?", default=False):
                ctx = click.get_current_context()
                ctx.invoke(run)
        except click.Abort:
            pass  # non-interactive — skip silently


@cli.command()
@click.argument("project_name", required=False)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to an adm YAML defaults file. Auto-discovers adm.yml if omitted.",
)
@click.option(
    "--staging-url",
    default=None,
    help="Databricks staging workspace URL.",
)
@click.option(
    "--prod-url",
    default=None,
    help="Databricks production workspace URL. Optional.",
)
@click.option(
    "--catalog-name",
    default=None,
    help="Unity Catalog catalog name (e.g. us_comm_lakehouse_dev).",
)
@click.option(
    "--schema-name",
    default=None,
    help="Unity Catalog schema (database) where models will be registered.",
)
@click.option(
    "--inference/--skip-inference",
    "with_inference",
    default=None,
    help="Generate the batch inference job scaffold.",
)
@click.option(
    "--with-dqx/--without-dqx",
    default=None,
    help="Include the optional DQX data quality scaffold.",
)
def new(
    project_name: str | None,
    config_path: Path | None,
    staging_url: str | None,
    prod_url: str | None,
    catalog_name: str | None,
    schema_name: str | None,
    with_inference: bool | None,
    with_dqx: bool | None,
) -> None:
    """Create a new ML project with MLOps scaffolding."""
    cwd = Path.cwd()
    defaults, defaults_path = _load_cli_defaults(cwd, config_path)
    if defaults_path is not None:
        click.echo(f"Using defaults from {_display_path(defaults_path, cwd)}")

    if project_name is None:
        project_name = _prompt_text("Project name", defaults.project_name)

    if staging_url is None:
        staging_url = _prompt_text("Staging workspace URL", defaults.staging_url)

    if prod_url is None:
        prod_url = _prompt_optional_text("Prod workspace URL (enter to skip)", defaults.prod_url)

    if catalog_name is None:
        catalog_name = _prompt_text("Unity Catalog name", defaults.catalog_name)

    if schema_name is None:
        schema_name = _prompt_text("Unity Catalog schema (database)", defaults.schema_name)

    if with_inference is None:
        with_inference = _prompt_bool("Include batch inference job?", defaults.with_inference, True)

    if with_dqx is None:
        with_dqx = defaults.with_dqx if defaults.with_dqx is not None else False

    project_dir = Path.cwd() / project_name
    if project_dir.exists():
        raise click.ClickException(f"Directory already exists: {project_dir}")

    project_dir.mkdir(parents=True)

    config = ProjectConfig(
        project_name=project_name,
        staging_workspace_url=_sanitize_url(staging_url),
        catalog_name=catalog_name,
        schema_name=schema_name,
        prod_workspace_url=_sanitize_url(prod_url) if prod_url else "",
        with_inference=with_inference,
        with_dqx=with_dqx,
    )
    rendered = render_templates(config)
    created = write_files(project_dir, rendered)

    for path in created:
        click.echo(f"  Created {path.relative_to(Path.cwd())}")
    click.echo()
    click.echo(f"Done! Run `cd {project_name} && databricks bundle validate` to verify.")


@cli.command()
def clean() -> None:
    """Remove all adm generated files from the current directory."""
    cwd = Path.cwd()

    all_templates = list(CORE_TEMPLATES) + list(INFERENCE_TEMPLATES) + list(DQX_TEMPLATES)
    generated_files = [_output_path(t) for t in all_templates]

    removed: list[str] = []
    for rel_path in generated_files:
        target = cwd / rel_path
        if target.exists():
            target.unlink()
            removed.append(rel_path)

    # Clean up empty directories left behind
    for dir_name in ["mlops", "resources", "notebooks"]:
        d = cwd / dir_name
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            removed.append(f"{dir_name}/")

    if removed:
        for r in removed:
            click.echo(f"  Removed {r}")
        click.echo()
        click.echo(f"Cleaned {len(removed)} files. Ready for a fresh `adm init`.")
    else:
        click.echo("Nothing to clean — no generated files found.")


@cli.command()
@click.option(
    "--source",
    default=None,
    help="Local repository path or public Git URL to review. Defaults to the current directory.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path for the generated Markdown review document.",
)
@click.option(
    "--endpoint",
    "preferred_endpoint",
    default=None,
    help="Specific Databricks Model Serving endpoint to use. Defaults to the best available preferred endpoint.",
)
@click.option(
    "--max-file-chars",
    default=120000,
    show_default=True,
    type=int,
    help="Maximum characters captured from any single file.",
)
@click.option(
    "--max-total-chars",
    default=2400000,
    show_default=True,
    type=int,
    help="Maximum total characters captured across the repository snapshot.",
)
def document(
    source: str | None,
    output_path: Path | None,
    preferred_endpoint: str | None,
    max_file_chars: int,
    max_total_chars: int,
) -> None:
    """Review a repository and generate a prioritized Markdown improvement document."""
    if max_file_chars <= 0:
        raise click.ClickException("--max-file-chars must be greater than 0.")
    if max_total_chars <= 0:
        raise click.ClickException("--max-total-chars must be greater than 0.")
    if max_file_chars > max_total_chars:
        raise click.ClickException("--max-file-chars cannot be greater than --max-total-chars.")

    try:
        artifact = review_repository(
            source=source,
            output_path=output_path,
            working_directory=Path.cwd(),
            preferred_endpoint=preferred_endpoint,
            max_file_chars=max_file_chars,
            max_total_chars=max_total_chars,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    omitted_count = len(artifact.snapshot.omitted_files)
    click.echo(f"Review document created: {artifact.output_path}")
    click.echo(f"Source reviewed: {artifact.source_label}")
    click.echo(f"Serving endpoint: {artifact.endpoint_name}")
    click.echo(
        f"Included {len(artifact.snapshot.files)} files totaling {artifact.snapshot.total_characters} characters"
    )
    click.echo(f"Omitted or truncated files: {omitted_count}")
    click.echo(f"Prompt snapshot saved to {artifact.prompt_path}")
    click.echo(f"Internal research note saved to {artifact.research_path}")


@cli.command()
@click.option(
    "--target",
    default="dev",
    show_default=True,
    help="Bundle target to deploy and run against (dev/staging/prod).",
)
def run(target: str) -> None:
    """Deploy the bundle and run the training job. Prints the MLflow experiment URL."""
    cwd = Path.cwd()

    databricks_yml = cwd / "databricks.yml"
    if not databricks_yml.exists():
        raise click.ClickException("No databricks.yml found. Run `adm init` first.")

    bundle = yaml.safe_load(databricks_yml.read_text())
    project_name = bundle.get("bundle", {}).get("name", cwd.name)

    # Workspace URL: try the requested target, fall back to dev/staging
    workspace_url = (
        _extract_yaml_host(databricks_yml.read_text(), target)
        or _extract_yaml_host(databricks_yml.read_text(), "dev")
        or _extract_yaml_host(databricks_yml.read_text(), "staging")
    )

    # 1. Deploy — stream output directly so user sees progress live
    click.echo(f"Deploying bundle to target '{target}'...")
    deploy = subprocess.run(
        ["databricks", "bundle", "deploy", "-t", target],
        cwd=cwd,
    )
    if deploy.returncode != 0:
        raise click.ClickException("Deploy failed (see output above).")

    experiment_url = ""
    try:
        experiment_name = _resolve_experiment_name(bundle, target)
        if experiment_name and workspace_url:
            profile = _resolve_profile_for_host(workspace_url)
            w = WorkspaceClient(host=workspace_url, profile=profile or None)
            _validate_registry_schema(workspace_url, profile, bundle, target)
            _ensure_workspace_parent_dir(w, experiment_name)
            experiment_id = _get_or_create_experiment_id(w, experiment_name)
            if experiment_id:
                experiment_url = f"{workspace_url}/ml/experiments/{experiment_id}"
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"  (Could not resolve experiment URL: {e})", err=True)

    # 2. Start training job — capture output to extract the run URL
    click.echo("Starting training job...")
    run_proc = subprocess.run(
        ["databricks", "bundle", "run", "model_training_job", "-t", target, "--no-wait"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if run_proc.returncode != 0:
        click.echo(run_proc.stderr or run_proc.stdout)
        raise click.ClickException("Failed to start training job (see output above).")

    # CLI outputs "Run URL: https://..." as plain text
    run_page_url = ""
    match = re.search(r"Run URL:\s*(https?://\S+)", run_proc.stdout + run_proc.stderr)
    if match:
        run_page_url = match.group(1)

    click.echo()
    click.echo("Training job started.")
    click.echo()
    if run_page_url:
        click.echo(f"  Job run:    {run_page_url}")
    if experiment_url:
        click.echo(f"  Experiment: {experiment_url}")
    click.echo()
    click.echo("The job runs Train → Validate → Deploy in sequence.")
    click.echo("Once complete, the model will be registered in Unity Catalog with the 'champion' alias.")
    click.echo()
    click.echo("  Tip: To retrigger from a notebook (no CLI needed):")
    click.echo("    from as_databricks_mlops import run_training_job")
    click.echo(f'    run_training_job("{target}-{project_name}-model-training-job")')
    click.echo()
    click.echo("  No local terminal? Use the Databricks Web Terminal (DBR 15.1+) in the browser UI.")


@cli.command()
@click.option(
    "--target",
    default="dev",
    show_default=True,
    help="Bundle target (dev/staging/prod) — used to construct the job name prefix.",
)
def trigger(target: str) -> None:
    """Trigger the deployed training job via the Databricks SDK (no CLI required)."""
    cwd = Path.cwd()

    databricks_yml = cwd / "databricks.yml"
    if not databricks_yml.exists():
        raise click.ClickException("No databricks.yml found. Run `adm init` first.")

    bundle = yaml.safe_load(databricks_yml.read_text())
    project_name = bundle.get("bundle", {}).get("name", cwd.name)
    job_name = f"{target}-{project_name}-model-training-job"

    click.echo(f"Triggering job: {job_name}")
    try:
        from as_databricks_mlops.trigger import run_training_job

        run_training_job(job_name)
    except ImportError:
        raise click.ClickException(
            "databricks-sdk is required for `adm trigger`. "
            "Install it: pip install 'as-databricks-mlops[sdk]'"
        )


@cli.group()
def add() -> None:
    """Add optional components to an existing adm project."""


@add.command("dqx")
@click.option("--overwrite", is_flag=True, help="Overwrite existing DQX files.")
def add_dqx(overwrite: bool) -> None:
    """Add DQX data quality checks to an existing project."""
    config_path = Path.cwd() / "mlops" / "config.py"
    if not config_path.exists():
        raise click.ClickException(
            "No mlops/config.py found. Run `adm init` first."
        )

    config_text = config_path.read_text()
    project_name = _extract_config_value(config_text, "MODEL_NAME")

    databricks_yml = Path.cwd() / "databricks.yml"
    if not databricks_yml.exists():
        raise click.ClickException(
            "No databricks.yml found. Run `adm init` first."
        )

    bundle_text = databricks_yml.read_text()
    staging_url = _extract_yaml_host(bundle_text, "staging")
    prod_url = _extract_yaml_host(bundle_text, "prod")

    config = ProjectConfig(
        project_name=project_name,
        staging_workspace_url=staging_url,
        prod_workspace_url=prod_url,
        with_dqx=True,
    )
    rendered = render_templates(config)

    dqx_files = {k: v for k, v in rendered.items() if "dqx" in k.lower()}
    created = write_files(Path.cwd(), dqx_files, overwrite=overwrite)

    for path in created:
        click.echo(f"  Created {path.relative_to(Path.cwd())}")
    click.echo()
    click.echo("Add this line to the `include` section of databricks.yml:")
    click.echo("  - ./resources/dqx-job.yml")


def _run_bundle_validate(directory: Path) -> bool:
    """Run `databricks bundle validate`. Returns True if valid."""
    click.echo("Running `databricks bundle validate`...")
    result = subprocess.run(
        ["databricks", "bundle", "validate"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        click.echo("  Bundle is valid.")
        return True
    else:
        click.echo("  Validation failed — fix the errors above before deploying.")
        click.echo(result.stdout or result.stderr)
        return False


def _extract_config_value(text: str, variable: str) -> str:
    """Extract a variable assignment value from config.py content."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(variable) and "=" in stripped:
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'")
    raise click.ClickException(
        f"Could not find {variable} in mlops/config.py"
    )


def _resolve_profile_for_host(host: str) -> str:
    """Return the ~/.databrickscfg profile name whose host matches the given URL, or ''."""
    try:
        import configparser
        cfg_path = Path.home() / ".databrickscfg"
        if not cfg_path.exists():
            return ""
        parser = configparser.ConfigParser()
        parser.read(cfg_path)
        host_norm = host.rstrip("/").lower()
        for section in parser.sections():
            cfg_host = parser.get(section, "host", fallback="").rstrip("/").lower()
            if cfg_host == host_norm:
                return section
    except Exception:
        pass
    return ""


def _extract_yaml_host(text: str, target: str) -> str:
    """Extract workspace host URL from databricks.yml for a given target."""
    try:
        bundle = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise click.ClickException(f"Failed to parse databricks.yml: {e}")

    try:
        return bundle["targets"][target]["workspace"]["host"]
    except (KeyError, TypeError):
        return ""


def _resolve_experiment_name(bundle: dict, target: str) -> str:
    """Resolve the configured experiment name for a target from databricks.yml."""
    experiment_name_template = (
        bundle.get("variables", {})
        .get("experiment_name", {})
        .get("default", "")
    )
    return experiment_name_template.replace("${bundle.target}", target)


def _resolve_bundle_variable(bundle: dict[str, Any], target: str, variable_name: str) -> str:
    """Resolve a bundle variable from target overrides first, then top-level defaults."""
    target_value = (
        bundle.get("targets", {})
        .get(target, {})
        .get("variables", {})
        .get(variable_name)
    )
    if isinstance(target_value, str) and target_value:
        return target_value

    default_value = (
        bundle.get("variables", {})
        .get(variable_name, {})
        .get("default", "")
    )
    return default_value if isinstance(default_value, str) else ""


def _ensure_workspace_parent_dir(workspace_client: WorkspaceClient, experiment_name: str) -> None:
    """Ensure the workspace parent directory exists for an MLflow experiment path."""
    parent_dir = str(PurePosixPath(experiment_name).parent)
    if not parent_dir or parent_dir == ".":
        return
    workspace_client.workspace.mkdirs(parent_dir)


def _get_or_create_experiment_id(workspace_client: WorkspaceClient, experiment_name: str) -> str | None:
    """Get or create an MLflow experiment and return its ID."""
    experiment_id = None

    try:
        resp = workspace_client.experiments.get_by_name(experiment_name=experiment_name)
        experiment_id = resp.experiment.experiment_id
    except ResourceDoesNotExist:
        try:
            resp = workspace_client.experiments.create_experiment(name=experiment_name)
            experiment_id = resp.experiment_id
        except ResourceAlreadyExists:
            resp = workspace_client.experiments.get_by_name(experiment_name=experiment_name)
            experiment_id = resp.experiment.experiment_id

    return experiment_id


def _validate_registry_schema(
    workspace_url: str,
    profile: str,
    bundle: dict[str, Any],
    target: str,
) -> None:
    """Fail early if the configured Unity Catalog schema does not exist or is inaccessible."""
    catalog_name = _resolve_bundle_variable(bundle, target, "catalog_name")
    schema_name = _resolve_bundle_variable(bundle, target, "schema_name")
    if not workspace_url or not catalog_name or not schema_name:
        return

    full_name = f"{catalog_name}.{schema_name}"
    workspace_client = WorkspaceClient(host=workspace_url, profile=profile or None)
    try:
        workspace_client.schemas.get(full_name=full_name)
    except ResourceDoesNotExist as exc:
        raise click.ClickException(
            f"Unity Catalog schema '{full_name}' does not exist in {workspace_url}. "
            "Use an existing schema in adm.yml or databricks.yml before running `adm run`."
        ) from exc
    except PermissionDenied as exc:
        raise click.ClickException(
            f"You do not have access to Unity Catalog schema '{full_name}' in {workspace_url}."
        ) from exc
