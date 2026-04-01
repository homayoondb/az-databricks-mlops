# Databricks notebook source
"""Run the full MLOps pipeline interactively (no deployed job needed).

Uses dbutils.notebook.run() to call the three MLOps notebooks in sequence:
  Train → Validate → Deploy

Useful for:
  - Interactive development and debugging
  - Running the pipeline without `adm run` or a scheduled job
  - Stepping through individual stages by running cells selectively

Prerequisite: Run `adm init` to generate the mlops/ notebooks first.
"""

# COMMAND ----------

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("dataset_table", "")  # optional: e.g. catalog.schema.features

env = dbutils.widgets.get("env")
dataset_table = dbutils.widgets.get("dataset_table")

project_name = "messy-ml-project"
catalog_name = "us_comm_lakehouse_dev"
schema_name = "az_brand_assistant"
model_name_slug = project_name.replace("-", "_")

experiment_name = f"/Shared/az-databricks-mlops/{env}-{project_name}-experiment"
model_name = f"{catalog_name}.{schema_name}.{model_name_slug}"

print(f"env:             {env}")
print(f"experiment_name: {experiment_name}")
print(f"model_name:      {model_name}")
if dataset_table:
    print(f"dataset_table:   {dataset_table}")

# COMMAND ----------

# Stage 1: Train
print("=" * 60)
print("Stage 1/3: Training")
print("=" * 60)
dbutils.notebook.run(
    "../mlops/run_training",
    3600,
    {
        "env": env,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "dataset_table": dataset_table,
    },
)
print("Training complete.")

# COMMAND ----------

# Stage 2: Validate
print("=" * 60)
print("Stage 2/3: Validation")
print("=" * 60)
dbutils.notebook.run(
    "../mlops/run_validation",
    3600,
    {
        "env": env,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "run_mode": "dry_run",
        "model_type": "regressor",
    },
)
print("Validation complete.")

# COMMAND ----------

# Stage 3: Deploy
print("=" * 60)
print("Stage 3/3: Deployment")
print("=" * 60)
dbutils.notebook.run(
    "../mlops/run_deploy",
    3600,
    {
        "env": env,
        "model_name": model_name,
    },
)
print("Deployment complete.")
print(f"Model '{model_name}' is now registered with the 'champion' alias.")
