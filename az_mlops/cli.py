"""CLI entry point for az-mlops."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click
import yaml

from az_mlops.generator import (
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
        staging_workspace_url=staging_url.rstrip("/"),
        prod_workspace_url=prod_url.strip().rstrip("/") if prod_url else "",
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
        if click.confirm("  Run the training job now?", default=False):
            ctx = click.get_current_context()
            ctx.invoke(run)


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
    "--skip-inference",
    is_flag=True,
    help="Skip batch inference job generation.",
)
@click.option("--with-dqx", is_flag=True, help="Include DQX data quality checks.")
def new(
    project_name: str,
    staging_url: str,
    prod_url: str,
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
        staging_workspace_url=staging_url.rstrip("/"),
        prod_workspace_url=prod_url.strip().rstrip("/") if prod_url else "",
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
    """Remove all az-mlops generated files from the current directory."""
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
    for dir_name in ["mlops", "resources"]:
        d = cwd / dir_name
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            removed.append(f"{dir_name}/")

    if removed:
        for r in removed:
            click.echo(f"  Removed {r}")
        click.echo()
        click.echo(f"Cleaned {len(removed)} files. Ready for a fresh `az-mlops init`.")
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
        raise click.ClickException("No databricks.yml found. Run `az-mlops init` first.")

    bundle = yaml.safe_load(databricks_yml.read_text())
    project_name = bundle.get("bundle", {}).get("name", cwd.name)

    # Workspace URL: try the requested target, fall back to dev/staging
    workspace_url = (
        _extract_yaml_host(databricks_yml.read_text(), target)
        or _extract_yaml_host(databricks_yml.read_text(), "dev")
        or _extract_yaml_host(databricks_yml.read_text(), "staging")
    )

    # 1. Deploy
    click.echo(f"Deploying bundle to target '{target}'...")
    deploy = subprocess.run(
        ["databricks", "bundle", "deploy", "-t", target],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if deploy.returncode != 0:
        raise click.ClickException(f"Deploy failed:\n{deploy.stderr or deploy.stdout}")
    click.echo("  Deployed.")

    # 2. Start training job (no-wait so we get the URL immediately)
    click.echo("Starting training job...")
    run_proc = subprocess.run(
        ["databricks", "bundle", "run", "model_training_job", "-t", target, "--no-wait", "--output", "json"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if run_proc.returncode != 0:
        raise click.ClickException(f"Run failed:\n{run_proc.stderr or run_proc.stdout}")

    run_page_url = ""
    try:
        run_data = json.loads(run_proc.stdout)
        run_page_url = run_data.get("run_page_url", "")
    except (json.JSONDecodeError, AttributeError):
        pass

    # 3. Get current user to construct the experiment URL
    experiment_url = ""
    try:
        me = subprocess.run(
            ["databricks", "current-user", "me", "--output", "json"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        user_name = json.loads(me.stdout).get("userName", "")
        if user_name and workspace_url:
            experiment_name = f"{target}-{project_name}-experiment"
            experiment_url = f"{workspace_url}/#mlflow/experiments?searchFilter=name+like+%27%25{experiment_name}%25%27"
    except Exception:
        pass

    click.echo()
    click.echo("Training job started.")
    click.echo()
    if run_page_url:
        click.echo(f"  Job run:    {run_page_url}")
    if experiment_url:
        click.echo(f"  Experiment: {experiment_url}")
    elif workspace_url:
        click.echo(f"  Experiment: {workspace_url}/#mlflow/experiments")
        click.echo(f"              (search for '{target}-{project_name}-experiment')")
    click.echo()
    click.echo("The job runs Train → Validate → Deploy in sequence.")
    click.echo("Once complete, the model will be registered in Unity Catalog with the 'champion' alias.")


@cli.group()
def add() -> None:
    """Add optional components to an existing az-mlops project."""


@add.command("dqx")
@click.option("--overwrite", is_flag=True, help="Overwrite existing DQX files.")
def add_dqx(overwrite: bool) -> None:
    """Add DQX data quality checks to an existing project."""
    config_path = Path.cwd() / "mlops" / "config.py"
    if not config_path.exists():
        raise click.ClickException(
            "No mlops/config.py found. Run `az-mlops init` first."
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
