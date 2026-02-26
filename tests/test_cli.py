"""Tests for CLI commands."""

import os

import pytest
from click.testing import CliRunner

from az_databricks_mlops.cli import _resolve_profile_for_host, cli


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
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
        "[az-dev]\nhost = https://adb-123.azuredatabricks.net\ntoken = dapi123\n\n"
        "[prod]\nhost = https://adb-456.azuredatabricks.net\ntoken = dapi456\n"
    )
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert _resolve_profile_for_host("https://adb-123.azuredatabricks.net") == "az-dev"
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
