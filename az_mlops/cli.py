"""CLI entry point for az-mlops."""

from __future__ import annotations

from pathlib import Path

import click

from az_mlops.generator import ProjectConfig, render_templates, write_files


@click.group()
@click.version_option()
def cli() -> None:
    """Lightweight MLOps scaffolding for Databricks projects."""


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
@click.option("--with-dqx", is_flag=True, help="Include DQX data quality checks.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
def init(
    project_name: str,
    staging_url: str,
    prod_url: str,
    with_dqx: bool,
    overwrite: bool,
) -> None:
    """Add MLOps scaffolding to an existing project."""
    config = ProjectConfig(
        project_name=project_name,
        staging_workspace_url=staging_url.rstrip("/"),
        prod_workspace_url=prod_url.rstrip("/"),
        with_dqx=with_dqx,
    )
    rendered = render_templates(config)
    created = write_files(Path.cwd(), rendered, overwrite=overwrite)

    for path in created:
        click.echo(f"  Created {path.relative_to(Path.cwd())}")
    click.echo()
    click.echo("Done! Run `databricks bundle validate` to verify.")


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
@click.option("--with-dqx", is_flag=True, help="Include DQX data quality checks.")
def new(
    project_name: str,
    staging_url: str,
    prod_url: str,
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
        with_dqx=with_dqx,
    )
    rendered = render_templates(config)
    created = write_files(project_dir, rendered)

    for path in created:
        click.echo(f"  Created {path.relative_to(Path.cwd())}")
    click.echo()
    click.echo(f"Done! cd {project_name} && databricks bundle validate")


@cli.group()
def add() -> None:
    """Add optional components to an existing az-mlops project."""


@add.command("dqx")
@click.option("--overwrite", is_flag=True, help="Overwrite existing DQX files.")
def add_dqx(overwrite: bool) -> None:
    """Add DQX data quality checks to an existing project."""
    # Read existing config to get project settings
    config_path = Path.cwd() / "mlops" / "config.py"
    if not config_path.exists():
        raise click.ClickException(
            "No mlops/config.py found. Run `az-mlops init` first."
        )

    # Parse project name from existing config
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

    # Only write DQX-specific files
    dqx_files = {
        k: v for k, v in rendered.items() if "dqx" in k.lower()
    }
    created = write_files(Path.cwd(), dqx_files, overwrite=overwrite)

    # Remind user to add the DQX resource include to databricks.yml
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
        # Stop if we hit another top-level target
        if in_target and not stripped.startswith("#") and ":" in stripped and not stripped.startswith("host") and not stripped.startswith("catalog") and not stripped.startswith("variable"):
            if line[0] != " " or (len(line) - len(line.lstrip())) <= 4:
                in_target = False
    return ""
