"""CLI entry point for az-mlops."""

from __future__ import annotations

from pathlib import Path

import click

from az_mlops.generator import (
    ProjectConfig,
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
        # If they typed a number, resolve it
        try:
            idx = int(choice)
            if 1 <= idx <= len(notebooks):
                return notebooks[idx - 1]
        except ValueError:
            pass
        return choice
    else:
        return click.prompt(f"  {label}", default=default)


@cli.command()
@click.option(
    "--project-name",
    prompt="Project name",
    default=lambda: Path.cwd().name,
    show_default="current directory name",
    help="Name for the ML project.",
)
@click.option(
    "--staging-url",
    prompt="Staging workspace URL",
    help="Databricks staging workspace URL.",
)
@click.option(
    "--prod-url",
    prompt="Prod workspace URL",
    help="Databricks production workspace URL.",
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
    project_name: str,
    staging_url: str,
    prod_url: str,
    training_notebook: str | None,
    inference_notebook: str | None,
    skip_inference: bool,
    with_dqx: bool,
    overwrite: bool,
) -> None:
    """Add MLOps scaffolding to an existing project."""
    cwd = Path.cwd()

    # Prompt for training notebook if not provided
    if training_notebook is None:
        training_notebook = _prompt_notebook(
            "Training notebook/script",
            cwd,
            default="training/notebooks/Train.py",
        )

    # Prompt for inference notebook if not skipped and not provided
    with_inference = not skip_inference
    if with_inference and inference_notebook is None:
        include = click.confirm("  Include batch inference job?", default=True)
        if include:
            inference_notebook = _prompt_notebook(
                "Inference notebook/script",
                cwd,
                default="deployment/batch_inference/notebooks/BatchInference.py",
            )
        else:
            with_inference = False

    config = ProjectConfig(
        project_name=project_name,
        staging_workspace_url=staging_url.rstrip("/"),
        prod_workspace_url=prod_url.rstrip("/"),
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
    click.echo("Done! Next steps are in GETTING_STARTED.md")


@cli.command()
@click.argument("project_name")
@click.option(
    "--staging-url",
    prompt="Staging workspace URL",
    help="Databricks staging workspace URL.",
)
@click.option(
    "--prod-url",
    prompt="Prod workspace URL",
    help="Databricks production workspace URL.",
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
        prod_workspace_url=prod_url.rstrip("/"),
        with_inference=not skip_inference,
        with_dqx=with_dqx,
    )
    rendered = render_templates(config)
    created = write_files(project_dir, rendered)

    for path in created:
        click.echo(f"  Created {path.relative_to(Path.cwd())}")
    click.echo()
    click.echo(f"Done! See {project_name}/GETTING_STARTED.md for next steps.")


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


def _extract_config_value(text: str, variable: str) -> str:
    """Extract a variable assignment value from config.py content."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{variable}") and "=" in stripped:
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'")
    return "my_project"


def _extract_yaml_host(text: str, target: str) -> str:
    """Extract workspace host URL from databricks.yml for a given target."""
    in_target = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == f"{target}:":
            in_target = True
            continue
        if in_target and stripped.startswith("host:"):
            return stripped.split(":", 1)[1].strip()
        if in_target and not stripped.startswith("#") and ":" in stripped and not stripped.startswith("host") and not stripped.startswith("catalog") and not stripped.startswith("variable"):
            if line[0] != " " or (len(line) - len(line.lstrip())) <= 4:
                in_target = False
    return ""
