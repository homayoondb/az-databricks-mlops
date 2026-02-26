"""CLI entry point for adm."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
from pathlib import Path

import click
import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceAlreadyExists, ResourceDoesNotExist

from az_databricks_mlops.generator import (
    CORE_TEMPLATES,
    DQX_TEMPLATES,
    INFERENCE_TEMPLATES,
    ProjectConfig,
    _output_path,
    find_notebooks,
    render_templates,
    write_files,
)


@click.group()
@click.version_option()
def cli() -> None:
    """Lightweight MLOps scaffolding for Databricks projects."""


def _prompt_notebook(label: str, directory: Path, default: str) -> str:
    """Prompt the user to select or type a notebook path, showing discovered files."""
    notebooks = find_notebooks(directory)
    if notebooks:
        click.echo(f"\n  Found notebooks/scripts in your project:")
        for i, nb in enumerate(notebooks, 1):
            click.echo(f"    {i}. {nb}")
        click.echo()
        choice = click.prompt(
            f"  {label} (number or path)",
            default=default,
        )
        try:
            idx = int(choice)
            if 1 <= idx <= len(notebooks):
                return notebooks[idx - 1]
        except ValueError:
            pass
        return choice
    else:
        return click.prompt(f"  {label}", default=default)


def _sanitize_url(value: str) -> str:
    """Strip control/escape characters; return empty string if not a valid URL."""
    # Remove ANSI escape sequences (e.g. arrow keys: \x1b[C) and other control chars
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value).strip()
    return clean if clean.startswith("http") else ""


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
        return bundle["variables"]["catalog_name"]["default"]
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
    help="Path to your inference notebook/script. Omit to skip batch inference.",
)
@click.option(
    "--skip-inference",
    is_flag=True,
    help="Skip batch inference job generation.",
)
@click.option("--with-dqx", is_flag=True, help="Include DQX data quality checks.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
def init(
    project_name: str | None,
    staging_url: str | None,
    prod_url: str | None,
    catalog_name: str | None,
    schema_name: str | None,
    training_notebook: str | None,
    inference_notebook: str | None,
    skip_inference: bool,
    with_dqx: bool,
    overwrite: bool,
) -> None:
    """Add MLOps scaffolding to an existing project."""
    cwd = Path.cwd()

    # Smart defaults: detect from existing files or use folder name
    if project_name is None:
        project_name = click.prompt("Project name", default=cwd.name)

    if staging_url is None:
        detected = _detect_staging_url(cwd)
        staging_url = click.prompt("Staging workspace URL", default=detected or None)

    if prod_url is None:
        detected = _detect_prod_url(cwd)
        prod_url = click.prompt(
            "Prod workspace URL (enter to skip)",
            default=detected or "",
            show_default=bool(detected),
        )

    if catalog_name is None:
        detected = _detect_catalog_name(cwd)
        catalog_name = click.prompt("Unity Catalog name", default=detected or None)

    if schema_name is None:
        detected = _detect_schema_name(cwd)
        schema_name = click.prompt("Unity Catalog schema (database)", default=detected or None)

    if training_notebook is None:
        training_notebook = _prompt_notebook(
            "Training notebook/script",
            cwd,
            default="train.py",
        )

    with_inference = not skip_inference
    if with_inference and inference_notebook is None:
        include = click.confirm("  Include batch inference job?", default=True)
        if include:
            inference_notebook = _prompt_notebook(
                "Inference notebook/script",
                cwd,
                default="predict.py",
            )
        else:
            with_inference = False

    config = ProjectConfig(
        project_name=project_name,
        staging_workspace_url=_sanitize_url(staging_url),
        catalog_name=catalog_name,
        schema_name=schema_name,
        prod_workspace_url=_sanitize_url(prod_url) if prod_url else "",
        training_notebook=training_notebook,
        with_inference=with_inference,
        inference_notebook=inference_notebook or "",
        with_dqx=with_dqx,
    )
    rendered = render_templates(config)
    created = write_files(cwd, rendered, overwrite=overwrite)

    click.echo()
    for path in created:
        click.echo(f"  Created {path.relative_to(cwd)}")
    click.echo()
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
@click.argument("project_name")
@click.option(
    "--staging-url",
    prompt="Staging workspace URL",
    help="Databricks staging workspace URL.",
)
@click.option(
    "--prod-url",
    default="",
    prompt="Prod workspace URL (enter to skip)",
    prompt_required=False,
    help="Databricks production workspace URL. Optional.",
)
@click.option(
    "--catalog-name",
    prompt="Unity Catalog name",
    help="Unity Catalog catalog name (e.g. us_comm_lakehouse_dev).",
)
@click.option(
    "--schema-name",
    prompt="Unity Catalog schema (database)",
    help="Unity Catalog schema (database) where models will be registered.",
)
@click.option(
    "--skip-inference",
    is_flag=True,
    help="Skip batch inference job generation.",
)
@click.option("--with-dqx", is_flag=True, help="Include DQX data quality checks.")
def new(
    project_name: str,
    staging_url: str,
    prod_url: str,
    catalog_name: str,
    schema_name: str,
    skip_inference: bool,
    with_dqx: bool,
) -> None:
    """Create a new ML project with MLOps scaffolding."""
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
        with_inference=not skip_inference,
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

    # 3. Get or create the MLflow experiment to obtain its direct URL.
    #    Using the SDK (already a dep) instead of subprocess — avoids CLI output-format issues.
    experiment_url = ""
    try:
        experiment_name_template = (
            bundle.get("variables", {})
            .get("experiment_name", {})
            .get("default", "")
        )
        experiment_name = experiment_name_template.replace("${bundle.target}", target)

        if experiment_name and workspace_url:
            # Resolve the profile that the Databricks CLI would use for this host,
            # so the SDK authenticates the same way (PAT from ~/.databrickscfg).
            profile = _resolve_profile_for_host(workspace_url)
            w = WorkspaceClient(host=workspace_url, profile=profile or None)
            experiment_id = None

            try:
                resp = w.experiments.get_by_name(experiment_name=experiment_name)
                experiment_id = resp.experiment.experiment_id
            except ResourceDoesNotExist:
                try:
                    resp = w.experiments.create_experiment(name=experiment_name)
                    experiment_id = resp.experiment_id
                except ResourceAlreadyExists:
                    # Race: created between our get and create
                    resp = w.experiments.get_by_name(experiment_name=experiment_name)
                    experiment_id = resp.experiment.experiment_id

            if experiment_id:
                experiment_url = f"{workspace_url}/ml/experiments/{experiment_id}"
    except Exception as e:
        click.echo(f"  (Could not resolve experiment URL: {e})", err=True)

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
    click.echo("    from az_databricks_mlops import run_training_job")
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
        from az_databricks_mlops.trigger import run_training_job

        run_training_job(job_name)
    except ImportError:
        raise click.ClickException(
            "databricks-sdk is required for `adm trigger`. "
            "Install it: pip install 'az-databricks-mlops[sdk]'"
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
            "No databricks.yml found. Run `az-mlops init` first."
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
