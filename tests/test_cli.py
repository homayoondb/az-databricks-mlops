"""Tests for CLI commands."""

import os

import pytest
from click.testing import CliRunner

from az_mlops.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


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
    assert (project_dir / "mlops" / "run_validation.py").exists()
    assert (project_dir / "mlops" / "run_deploy.py").exists()
    assert (project_dir / ".github" / "workflows" / "ci.yml").exists()
    assert (project_dir / ".github" / "workflows" / "cd.yml").exists()
    assert (project_dir / "GETTING_STARTED.md").exists()

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
            "--skip-inference",
        ],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "no_inf_project"
    assert (project_dir / "resources" / "training-job.yml").exists()
    assert not (project_dir / "resources" / "inference-job.yml").exists()


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
            "--training-notebook",
            "train.py",
            "--skip-inference",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "databricks.yml").exists()
    assert (tmp_path / "mlops" / "config.py").exists()
    assert (tmp_path / "GETTING_STARTED.md").exists()


def test_init_refuses_overwrite_by_default(runner, tmp_path):
    os.chdir(tmp_path)
    args = [
        "init",
        "--project-name", "p",
        "--staging-url", "https://s.com",
        "--prod-url", "https://p.com",
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
