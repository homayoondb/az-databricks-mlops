"""Tests for CLI commands."""

import os

import click
import pytest
from click.testing import CliRunner
from databricks.sdk.errors import ResourceDoesNotExist

from az_mlops.cli import cli as legacy_cli
from as_databricks_mlops.cli import (
    _detect_catalog_name,
    _resolve_profile_for_host,
    _validate_registry_schema,
    cli,
)


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_legacy_az_mlops_version(runner):
    result = runner.invoke(legacy_cli, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_new_creates_project(runner, tmp_path):
    os.chdir(tmp_path)
    result = runner.invoke(
        cli,
        [
            "new",
            "my_test_project",
            "--staging-url",
            "https://staging.cloud.databricks.com",
            "--prod-url",
            "https://prod.cloud.databricks.com",
            "--catalog-name",
            "my_catalog",
            "--schema-name",
            "my_schema",
        ],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "my_test_project"
    assert (project_dir / ".gitignore").exists()
    assert (project_dir / "databricks.yml").exists()
    assert (project_dir / "resources" / "training-job.yml").exists()
    assert (project_dir / "resources" / "inference-job.yml").exists()
    assert (project_dir / "mlops" / "__init__.py").exists()
    assert (project_dir / "mlops" / "config.py").exists()
    assert (project_dir / "mlops" / "validation.py").exists()
    assert (project_dir / "mlops" / "deploy.py").exists()
    assert (project_dir / "mlops" / "run_training.py").exists()
    assert (project_dir / "mlops" / "run_validation.py").exists()
    assert (project_dir / "mlops" / "run_deploy.py").exists()
    assert (project_dir / "mlops" / "run_inference.py").exists()

    # DQX not included by default
    assert not (project_dir / "resources" / "dqx-job.yml").exists()
    assert not (project_dir / "mlops" / "dqx_checks.py").exists()


def test_new_with_dqx(runner, tmp_path):
    os.chdir(tmp_path)
    result = runner.invoke(
        cli,
        [
            "new",
            "dqx_project",
            "--staging-url",
            "https://staging.cloud.databricks.com",
            "--prod-url",
            "https://prod.cloud.databricks.com",
            "--catalog-name",
            "my_catalog",
            "--schema-name",
            "my_schema",
            "--with-dqx",
        ],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "dqx_project"
    assert (project_dir / "resources" / "dqx-job.yml").exists()
    assert (project_dir / "mlops" / "dqx_checks.py").exists()


def test_new_skip_inference(runner, tmp_path):
    os.chdir(tmp_path)
    result = runner.invoke(
        cli,
        [
            "new",
            "no_inf_project",
            "--staging-url",
            "https://staging.cloud.databricks.com",
            "--prod-url",
            "https://prod.cloud.databricks.com",
            "--catalog-name",
            "my_catalog",
            "--schema-name",
            "my_schema",
            "--skip-inference",
        ],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "no_inf_project"
    assert (project_dir / "resources" / "training-job.yml").exists()
    assert not (project_dir / "resources" / "inference-job.yml").exists()


def test_new_without_prod_url(runner, tmp_path):
    os.chdir(tmp_path)
    result = runner.invoke(
        cli,
        [
            "new",
            "staging_only",
            "--staging-url",
            "https://staging.cloud.databricks.com",
            "--catalog-name",
            "my_catalog",
            "--schema-name",
            "my_schema",
        ],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "staging_only"
    assert (project_dir / "databricks.yml").exists()

    content = (project_dir / "databricks.yml").read_text()
    assert "staging" in content
    assert "prod" not in content


def test_new_fails_if_dir_exists(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "existing_project").mkdir()
    result = runner.invoke(
        cli,
        [
            "new",
            "existing_project",
            "--staging-url",
            "https://staging.cloud.databricks.com",
            "--prod-url",
            "https://prod.cloud.databricks.com",
            "--catalog-name",
            "my_catalog",
            "--schema-name",
            "my_schema",
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_init_creates_files_in_cwd(runner, tmp_path):
    os.chdir(tmp_path)
    result = runner.invoke(
        cli,
        [
            "init",
            "--project-name",
            "init_project",
            "--staging-url",
            "https://staging.cloud.databricks.com",
            "--prod-url",
            "https://prod.cloud.databricks.com",
            "--catalog-name",
            "my_catalog",
            "--schema-name",
            "my_schema",
            "--training-notebook",
            "train.py",
            "--skip-inference",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "databricks.yml").exists()
    assert (tmp_path / "mlops" / "config.py").exists()
    assert (tmp_path / "mlops" / "run_training.py").exists()


def test_init_refuses_overwrite_by_default(runner, tmp_path):
    os.chdir(tmp_path)
    args = [
        "init",
        "--project-name", "p",
        "--staging-url", "https://s.com",
        "--prod-url", "https://p.com",
        "--catalog-name", "my_catalog",
        "--schema-name", "my_schema",
        "--training-notebook", "train.py",
        "--skip-inference",
    ]
    runner.invoke(cli, args)
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    assert "adm clean" in result.output
    assert "--overwrite" in result.output


def test_init_overwrite_flag(runner, tmp_path):
    os.chdir(tmp_path)
    args = [
        "init",
        "--project-name", "p",
        "--staging-url", "https://s.com",
        "--prod-url", "https://p.com",
        "--catalog-name", "my_catalog",
        "--schema-name", "my_schema",
        "--training-notebook", "train.py",
        "--skip-inference",
    ]
    runner.invoke(cli, args)
    result = runner.invoke(cli, args + ["--overwrite"])
    assert result.exit_code == 0, result.output


def test_init_uses_adm_yml_defaults(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "adm.yml").write_text(
        """
project_name: yaml_project
databricks:
  staging_url: https://staging.from.yaml
  prod_url: ""
  catalog_name: us_comm_lakehouse_dev
  schema_name: az_databricks_mlops
training:
  training_notebook: notebooks/train.py
  skip_inference: true
options:
  with_dqx: false
""".strip()
    )
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "train.py").write_text("# training")

    result = runner.invoke(cli, ["init"], input="\n\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Using defaults from adm.yml" in result.output

    databricks_yml = (tmp_path / "databricks.yml").read_text()
    assert "https://staging.from.yaml" in databricks_yml
    assert "us_comm_lakehouse_dev" in databricks_yml
    assert "az_databricks_mlops" in databricks_yml
    assert "inference-job" not in databricks_yml


def test_init_warns_when_adm_yml_sets_ignored_inference_notebook(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "adm.yml").write_text(
        """
databricks:
  staging_url: https://staging.from.yaml
  catalog_name: us_comm_lakehouse_dev
  schema_name: az_databricks_mlops
training:
  training_notebook: train.py
  inference_notebook: predict.py
  skip_inference: false
""".strip()
    )
    (tmp_path / "train.py").write_text("# training")

    result = runner.invoke(cli, ["init"], input="\n\n\n\n\ny\nn\n")
    assert result.exit_code == 0, result.output
    assert "ignoring training.inference_notebook in adm.yml" in result.output
    assert "Inference notebook/script" not in result.output


def test_init_prefers_cli_over_adm_yml_defaults(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "adm.yml").write_text(
        """
databricks:
  staging_url: https://staging.from.yaml
  catalog_name: yaml_catalog
  schema_name: yaml_schema
training:
  training_notebook: train.py
  skip_inference: true
""".strip()
    )
    (tmp_path / "train.py").write_text("# training")

    result = runner.invoke(
        cli,
        [
            "init",
            "--staging-url",
            "https://flag.example.com",
            "--catalog-name",
            "flag_catalog",
            "--schema-name",
            "flag_schema",
        ],
    )
    assert result.exit_code == 0, result.output
    databricks_yml = (tmp_path / "databricks.yml").read_text()
    assert "https://flag.example.com" in databricks_yml
    assert "flag_catalog" in databricks_yml
    assert "flag_schema" in databricks_yml


def test_init_uses_built_in_prompt_defaults_without_adm_yml(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "train.py").write_text("# training")

    result = runner.invoke(
        cli,
        ["init"],
        input=(
            "plain_project\n"
            "https://staging.example.com\n"
            "\n"
            "plain_catalog\n"
            "plain_schema\n"
            "\n"
            "\n"
            "n\n"
        ),
    )
    assert result.exit_code == 0, result.output
    assert f"Project name [{tmp_path.name}]" in result.output
    assert "Training notebook/script (number or path) [train.py]" in result.output
    assert "Include batch inference job? [Y/n]" in result.output
    assert "Inference notebook/script" not in result.output


def test_init_adm_yml_overrides_detected_defaults(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "adm.yml").write_text(
        """
databricks:
  staging_url: https://staging.from.yaml
  catalog_name: yaml_catalog
  schema_name: yaml_schema
training:
  training_notebook: notebooks/from_yaml.py
  skip_inference: false
""".strip()
    )
    (tmp_path / "databricks.yml").write_text(
        """
targets:
  staging:
    workspace:
      host: https://staging.from.detected
  dev:
    variables:
      catalog_name: detected_catalog
variables:
  schema_name:
    default: detected_schema
""".strip()
    )
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "from_yaml.py").write_text("# training")

    result = runner.invoke(
        cli,
        ["init", "--overwrite"],
        input="\n\n\n\n\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "Staging workspace URL [https://staging.from.yaml]" in result.output
    assert "Unity Catalog name [yaml_catalog]" in result.output
    assert "Unity Catalog schema (database) [yaml_schema]" in result.output
    assert "Training notebook/script (number or path) [notebooks/from_yaml.py]" in result.output
    assert "Inference notebook/script" not in result.output


def test_init_ignores_legacy_inference_notebook_flag(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "train.py").write_text("# training")

    result = runner.invoke(
        cli,
        [
            "init",
            "--project-name",
            "reprompt_project",
            "--staging-url",
            "https://staging.example.com",
            "--prod-url",
            "",
            "--catalog-name",
            "my_catalog",
            "--schema-name",
            "my_schema",
            "--training-notebook",
            "train.py",
            "--inference-notebook",
            "predict.py",
            "--overwrite",
        ],
        input="y\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "ignoring --inference-notebook" in result.output
    assert "Inference notebook/script" not in result.output


def test_new_uses_adm_yml_defaults(runner, tmp_path):
    os.chdir(tmp_path)
    (tmp_path / "adm.yml").write_text(
        """
project_name: yaml_new_project
databricks:
  staging_url: https://staging.from.yaml
  prod_url: ""
  catalog_name: us_comm_lakehouse_dev
  schema_name: az_databricks_mlops
training:
  skip_inference: true
""".strip()
    )
    result = runner.invoke(cli, ["new"], input="\n\n\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "yaml_new_project" / "databricks.yml").exists()
    assert not (tmp_path / "yaml_new_project" / "resources" / "inference-job.yml").exists()


def test_init_discovers_parent_adm_yml(runner, tmp_path):
    (tmp_path / "adm.yml").write_text(
        """
databricks:
  staging_url: https://staging.from.parent
  catalog_name: us_comm_lakehouse_dev
  schema_name: az_databricks_mlops
training:
  training_notebook: train.py
  skip_inference: true
""".strip()
    )
    child = tmp_path / "nested" / "project"
    child.mkdir(parents=True)
    (child / "train.py").write_text("# training")
    os.chdir(child)

    result = runner.invoke(cli, ["init", "--project-name", "nested_project"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Using defaults from" in result.output
    assert "staging.from.parent" in (child / "databricks.yml").read_text()


def test_add_dqx_to_existing_project(runner, tmp_path):
    os.chdir(tmp_path)
    runner.invoke(
        cli,
        [
            "init",
            "--project-name", "dqx_test",
            "--staging-url", "https://staging.example.com",
            "--prod-url", "https://prod.example.com",
            "--catalog-name", "my_catalog",
            "--schema-name", "my_schema",
            "--training-notebook", "train.py",
            "--skip-inference",
        ],
    )
    result = runner.invoke(cli, ["add", "dqx"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "resources" / "dqx-job.yml").exists()
    assert (tmp_path / "mlops" / "dqx_checks.py").exists()


def test_add_dqx_fails_without_init(runner, tmp_path):
    os.chdir(tmp_path)
    result = runner.invoke(cli, ["add", "dqx"])
    assert result.exit_code != 0
    assert "config.py" in result.output


def test_clean_removes_generated_files(runner, tmp_path):
    os.chdir(tmp_path)
    # First generate files
    result = runner.invoke(
        cli,
        [
            "init",
            "--project-name", "clean_test",
            "--staging-url", "https://staging.example.com",
            "--prod-url", "",
            "--catalog-name", "my_catalog",
            "--schema-name", "my_schema",
            "--training-notebook", "train.py",
            "--skip-inference",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "databricks.yml").exists()
    assert (tmp_path / "mlops" / "config.py").exists()

    # Now clean
    result = runner.invoke(cli, ["clean"])
    assert result.exit_code == 0, result.output
    assert "Cleaned" in result.output

    assert not (tmp_path / "databricks.yml").exists()
    assert not (tmp_path / "mlops" / "config.py").exists()
    assert not (tmp_path / "resources").exists()


def test_clean_nothing_to_clean(runner, tmp_path):
    os.chdir(tmp_path)
    result = runner.invoke(cli, ["clean"])
    assert result.exit_code == 0
    assert "Nothing to clean" in result.output


def test_clean_preserves_user_files(runner, tmp_path):
    os.chdir(tmp_path)
    # Create user files
    (tmp_path / "train.py").write_text("# my training code")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input.csv").write_text("a,b\n1,2")

    # Generate mlops files
    runner.invoke(
        cli,
        [
            "init",
            "--project-name", "preserve_test",
            "--staging-url", "https://staging.example.com",
            "--prod-url", "",
            "--catalog-name", "my_catalog",
            "--schema-name", "my_schema",
            "--training-notebook", "train.py",
            "--skip-inference",
        ],
    )

    # Clean
    runner.invoke(cli, ["clean"])

    # User files untouched
    assert (tmp_path / "train.py").exists()
    assert (tmp_path / "data" / "input.csv").exists()
    # Generated files gone
    assert not (tmp_path / "databricks.yml").exists()


def test_resolve_profile_for_host_matches(tmp_path, monkeypatch):
    cfg = tmp_path / ".databrickscfg"
    cfg.write_text(
        "[as-dev]\nhost = https://adb-123.azuredatabricks.net\ntoken = dapi123\n\n"
        "[prod]\nhost = https://adb-456.azuredatabricks.net\ntoken = dapi456\n"
    )
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert _resolve_profile_for_host("https://adb-123.azuredatabricks.net") == "as-dev"
    assert _resolve_profile_for_host("https://adb-456.azuredatabricks.net") == "prod"


def test_resolve_profile_for_host_trailing_slash(tmp_path, monkeypatch):
    cfg = tmp_path / ".databrickscfg"
    cfg.write_text("[myprofile]\nhost = https://adb-123.azuredatabricks.net/\ntoken = dapi123\n")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert _resolve_profile_for_host("https://adb-123.azuredatabricks.net") == "myprofile"


def test_resolve_profile_for_host_no_match(tmp_path, monkeypatch):
    cfg = tmp_path / ".databrickscfg"
    cfg.write_text("[other]\nhost = https://adb-999.azuredatabricks.net\ntoken = dapi999\n")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert _resolve_profile_for_host("https://adb-123.azuredatabricks.net") == ""


def test_resolve_profile_for_host_missing_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert _resolve_profile_for_host("https://adb-123.azuredatabricks.net") == ""


def test_detect_catalog_name_uses_target_variables(tmp_path):
    (tmp_path / "databricks.yml").write_text(
        """
targets:
  dev:
    variables:
      catalog_name: us_comm_lakehouse_dev
""".strip()
    )
    assert _detect_catalog_name(tmp_path) == "us_comm_lakehouse_dev"


def test_validate_registry_schema_fails_early(monkeypatch):
    class FakeSchemas:
        def get(self, full_name: str):
            raise ResourceDoesNotExist("missing")

    class FakeWorkspaceClient:
        def __init__(self, host: str, profile: str | None = None):
            self.schemas = FakeSchemas()

    monkeypatch.setattr("as_databricks_mlops.cli.WorkspaceClient", FakeWorkspaceClient)

    bundle = {
        "variables": {
            "schema_name": {"default": "missing_schema"},
        },
        "targets": {
            "dev": {
                "variables": {
                    "catalog_name": "us_comm_lakehouse_dev",
                }
            }
        },
    }

    with pytest.raises(click.ClickException) as excinfo:
        _validate_registry_schema(
            "https://workspace.example.com",
            "az-dev",
            bundle,
            "dev",
        )

    assert "missing_schema" in str(excinfo.value)
