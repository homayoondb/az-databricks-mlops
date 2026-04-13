"""Tests for template rendering."""

import yaml
import pytest

from az_databricks_mlops.generator import ProjectConfig, find_notebooks, render_templates


@pytest.fixture
def config():
    return ProjectConfig(
        project_name="test_project",
        staging_workspace_url="https://staging.cloud.databricks.com",
        prod_workspace_url="https://prod.cloud.databricks.com",
        training_notebook="notebooks/train.py",
    )


@pytest.fixture
def config_with_dqx():
    return ProjectConfig(
        project_name="test_project",
        staging_workspace_url="https://staging.cloud.databricks.com",
        prod_workspace_url="https://prod.cloud.databricks.com",
        training_notebook="notebooks/train.py",
        with_dqx=True,
    )


@pytest.fixture
def config_no_inference():
    return ProjectConfig(
        project_name="test_project",
        staging_workspace_url="https://staging.cloud.databricks.com",
        prod_workspace_url="https://prod.cloud.databricks.com",
        training_notebook="notebooks/train.py",
        with_inference=False,
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
        "mlops/run_training.py",
        "mlops/validation.py",
        "mlops/run_validation.py",
        "mlops/deploy.py",
        "mlops/run_deploy.py",
        "mlops/run_inference.py",
        "notebooks/run_pipeline.py",
    ]
    for f in expected_files:
        assert f in rendered, f"Missing: {f}"

    # GitHub Actions and GETTING_STARTED should NOT be generated
    assert ".github/workflows/ci.yml" not in rendered
    assert ".github/workflows/cd.yml" not in rendered
    assert "GETTING_STARTED.md" not in rendered


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
    assert bundle["variables"]["experiment_name"]["default"] == "/Shared/az-databricks-mlops/${bundle.target}-test_project-experiment"
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


def test_training_job_points_to_wrapper(config):
    rendered = render_templates(config)
    content = rendered["resources/training-job.yml"]

    assert "../mlops/run_training.py" in content
    assert "model-training-job" in content
    assert "databricks-sdk>=0.20" in content
    # Should NOT reference user's notebook directly in job YAML
    assert "notebooks/train.py" not in content


def test_run_training_references_user_notebook(config):
    rendered = render_templates(config)
    content = rendered["mlops/run_training.py"]

    assert "notebooks/train.py" in content
    assert "WorkspaceClient" in content
    assert "workspace.mkdirs" in content
    assert "mlflow.autolog" in content
    assert "register_model" in content
    assert "challenger" in content


def test_training_job_has_dqx_task_when_enabled(config_with_dqx):
    rendered = render_templates(config_with_dqx)
    content = rendered["resources/training-job.yml"]

    assert "DataQuality" in content
    assert "depends_on" in content


def test_training_job_no_dqx_task_by_default(config):
    rendered = render_templates(config)
    content = rendered["resources/training-job.yml"]

    assert "DataQuality" not in content


def test_inference_job_points_to_wrapper(config):
    rendered = render_templates(config)
    content = rendered["resources/inference-job.yml"]

    assert "../mlops/run_inference.py" in content


def test_run_inference_loads_champion(config):
    rendered = render_templates(config)
    content = rendered["mlops/run_inference.py"]

    assert "@champion" in content
    assert "predict" in content


def test_inference_job_excluded_when_skipped(config_no_inference):
    rendered = render_templates(config_no_inference)

    assert "resources/inference-job.yml" not in rendered
    assert "mlops/run_inference.py" not in rendered
    assert "inference" not in rendered["databricks.yml"]


def test_inference_job_included_by_default(config):
    rendered = render_templates(config)

    assert "resources/inference-job.yml" in rendered
    assert "mlops/run_inference.py" in rendered


def test_config_py_has_model_name(config):
    rendered = render_templates(config)
    content = rendered["mlops/config.py"]

    assert 'MODEL_NAME = "test_project"' in content
    assert 'SCHEMA_NAME = ""' in content


def test_config_py_uses_configured_schema_name():
    config = ProjectConfig(
        project_name="test_project",
        staging_workspace_url="https://staging.cloud.databricks.com",
        training_notebook="train.py",
        schema_name="az_brand_assistant",
    )
    rendered = render_templates(config)
    content = rendered["mlops/config.py"]

    assert 'SCHEMA_NAME = "az_brand_assistant"' in content


def test_validation_py_has_thresholds(config):
    rendered = render_templates(config)
    content = rendered["mlops/validation.py"]

    assert "MetricThreshold" in content
    assert "max_error" in content


def test_no_prod_target_when_url_empty():
    config = ProjectConfig(
        project_name="test_project",
        staging_workspace_url="https://staging.cloud.databricks.com",
        training_notebook="train.py",
    )
    rendered = render_templates(config)

    bundle = yaml.safe_load(rendered["databricks.yml"])
    assert "staging" in bundle["targets"]
    assert "prod" not in bundle["targets"]


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


def test_find_notebooks(tmp_path):
    (tmp_path / "train.py").write_text("# training")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "explore.ipynb").write_text("{}")
    (tmp_path / "notebooks" / "run_pipeline.py").write_text("# generated")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hooks.py").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_text("")

    results = find_notebooks(tmp_path)

    assert "train.py" in results
    # notebooks/ content is included (except generated adm filenames)
    assert "notebooks/explore.ipynb" in results
    # Generated adm filenames are excluded even inside notebooks/
    assert "notebooks/run_pipeline.py" not in results
    # Should not include hidden dirs or pycache
    assert not any(".git" in r for r in results)
    assert not any("__pycache__" in r for r in results)


def test_run_pipeline_notebook_generated(config):
    rendered = render_templates(config)

    assert "notebooks/run_pipeline.py" in rendered
    content = rendered["notebooks/run_pipeline.py"]
    assert "dbutils.notebook.run" in content
    assert "run_training" in content
    assert "run_validation" in content
    assert "run_deploy" in content
    assert "dataset_table" in content


def test_run_pipeline_contains_project_values(config):
    rendered = render_templates(config)
    content = rendered["notebooks/run_pipeline.py"]

    assert "test_project" in content
    assert "/Shared/az-databricks-mlops/" in content


def test_run_training_has_dataset_table_widget(config):
    rendered = render_templates(config)
    content = rendered["mlops/run_training.py"]

    assert 'widgets.text("dataset_table"' in content
    assert "mlflow.log_input" in content
    assert "load_delta" in content
    assert "dataset_row_count" in content


def test_training_job_has_dataset_table_parameter(config):
    rendered = render_templates(config)
    content = rendered["resources/training-job.yml"]

    assert "dataset_table" in content
