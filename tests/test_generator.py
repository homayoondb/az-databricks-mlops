"""Tests for template rendering."""

import yaml
import pytest

from az_mlops.generator import ProjectConfig, render_templates


@pytest.fixture
def config():
    return ProjectConfig(
        project_name="test_project",
        staging_workspace_url="https://staging.cloud.databricks.com",
        prod_workspace_url="https://prod.cloud.databricks.com",
    )


@pytest.fixture
def config_with_dqx():
    return ProjectConfig(
        project_name="test_project",
        staging_workspace_url="https://staging.cloud.databricks.com",
        prod_workspace_url="https://prod.cloud.databricks.com",
        with_dqx=True,
    )


def test_core_templates_rendered(config):
    rendered = render_templates(config)

    expected_files = [
        ".gitignore",
        "databricks.yml",
        "resources/training-job.yml",
        "resources/inference-job.yml",
        "mlops/__init__.py",
        "mlops/config.py",
        "mlops/validation.py",
        "mlops/deploy.py",
        "mlops/run_validation.py",
        "mlops/run_deploy.py",
        ".github/workflows/ci.yml",
        ".github/workflows/cd.yml",
    ]
    for f in expected_files:
        assert f in rendered, f"Missing: {f}"


def test_dqx_templates_included_when_enabled(config_with_dqx):
    rendered = render_templates(config_with_dqx)

    assert "resources/dqx-job.yml" in rendered
    assert "mlops/dqx_checks.py" in rendered


def test_dqx_templates_excluded_by_default(config):
    rendered = render_templates(config)

    assert "resources/dqx-job.yml" not in rendered
    assert "mlops/dqx_checks.py" not in rendered


def test_databricks_yml_has_correct_values(config):
    rendered = render_templates(config)
    content = rendered["databricks.yml"]
    bundle = yaml.safe_load(content)

    assert bundle["bundle"]["name"] == "test_project"
    assert bundle["targets"]["staging"]["workspace"]["host"] == "https://staging.cloud.databricks.com"
    assert bundle["targets"]["prod"]["workspace"]["host"] == "https://prod.cloud.databricks.com"
    assert bundle["targets"]["dev"]["mode"] == "development"
    assert bundle["targets"]["dev"]["default"] is True


def test_databricks_yml_includes_dqx_when_enabled(config_with_dqx):
    rendered = render_templates(config_with_dqx)
    content = rendered["databricks.yml"]

    assert "./resources/dqx-job.yml" in content


def test_databricks_yml_excludes_dqx_by_default(config):
    rendered = render_templates(config)
    content = rendered["databricks.yml"]

    assert "dqx" not in content


def test_training_job_has_project_name(config):
    rendered = render_templates(config)
    content = rendered["resources/training-job.yml"]

    assert "test_project" in content
    assert "model-training-job" in content


def test_training_job_has_dqx_task_when_enabled(config_with_dqx):
    rendered = render_templates(config_with_dqx)
    content = rendered["resources/training-job.yml"]

    assert "DataQuality" in content
    assert "depends_on" in content


def test_training_job_no_dqx_task_by_default(config):
    rendered = render_templates(config)
    content = rendered["resources/training-job.yml"]

    assert "DataQuality" not in content


def test_inference_job_has_project_name(config):
    rendered = render_templates(config)
    content = rendered["resources/inference-job.yml"]

    assert "test_project" in content
    assert "batch-inference-job" in content


def test_config_py_has_model_name(config):
    rendered = render_templates(config)
    content = rendered["mlops/config.py"]

    assert 'MODEL_NAME = "test_project"' in content


def test_validation_py_has_thresholds(config):
    rendered = render_templates(config)
    content = rendered["mlops/validation.py"]

    assert "MetricThreshold" in content
    assert "max_error" in content


def test_ci_workflow_references_staging(config):
    rendered = render_templates(config)
    content = rendered[".github/workflows/ci.yml"]

    assert "staging.cloud.databricks.com" in content
    assert "bundle validate" in content


def test_cd_workflow_has_both_targets(config):
    rendered = render_templates(config)
    content = rendered[".github/workflows/cd.yml"]

    assert "deploy-staging" in content
    assert "deploy-prod" in content
    assert "STAGING_WORKSPACE_TOKEN" in content
    assert "PROD_WORKSPACE_TOKEN" in content


def test_deploy_py_has_promotion_logic(config):
    rendered = render_templates(config)
    content = rendered["mlops/deploy.py"]

    assert "champion" in content
    assert "challenger" in content
    assert "promote_model" in content


def test_gitignore_generated(config):
    rendered = render_templates(config)

    assert ".gitignore" in rendered
    assert "__pycache__" in rendered[".gitignore"]
