# Databricks notebook source
"""MLOps training wrapper.

Wraps your existing training script (notebooks/train_model_v3_FINAL.py) with
automatic MLflow tracking and Unity Catalog model registration.
Your training code does NOT need any modifications.

What this does automatically:
  1. Sets up the MLflow experiment
  2. Enables mlflow.autolog() — captures model, metrics, params
  3. Runs your training script
  4. Registers the logged model in Unity Catalog
  5. Sets the "challenger" alias for validation
"""

import importlib.util
import os
import sys
from pathlib import Path
from pathlib import PurePosixPath

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

# COMMAND ----------

dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("dataset_table", "")  # optional: UC table used for training (e.g. catalog.schema.features)

experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
env = dbutils.widgets.get("env")
dataset_table = dbutils.widgets.get("dataset_table")

# COMMAND ----------

# Ensure the workspace directory exists before MLflow tries to create the experiment.
parent_dir = str(PurePosixPath(experiment_name).parent)
if parent_dir and parent_dir != ".":
    WorkspaceClient().workspace.mkdirs(parent_dir)

# COMMAND ----------

# 1. Configure MLflow
mlflow.set_experiment(experiment_name)
mlflow.autolog(log_models=True)

# COMMAND ----------

# 2. Resolve the project root from this notebook's workspace path.
#    In Databricks, notebookPath() returns the workspace path without extension,
#    e.g. /Users/.../.bundle/project/dev/files/mlops/run_training
#    We prefix /Workspace to get the real filesystem path, then go two levels
#    up (mlops/ → project root) to find the user's training script.
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_workspace_path = context.notebookPath().get()
project_root = Path("/Workspace" + notebook_workspace_path).parent.parent
script_path = project_root / "notebooks/train_model_v3_FINAL.py"
print(f"Loading training script from: {script_path}")

# Change working directory to project root so relative paths in the
# training script (e.g. "data/houses.csv") resolve correctly.
os.chdir(project_root)

# COMMAND ----------

# 3. Run user's training script inside an MLflow run.
#    autolog captures the model, metrics, and parameters automatically.
with mlflow.start_run() as run:
    # 3a. Log dataset lineage for Unity Catalog (requires MLflow 2.11+, DBR 15.3+).
    #     If dataset_table is set, UC will draw a lineage graph: table → run → model.
    if dataset_table:
        try:
            dataset = mlflow.data.load_delta(table_name=dataset_table)
            mlflow.log_input(dataset, context="training")
            mlflow.log_metric("dataset_row_count", spark.table(dataset_table).count())
            print(f"Logged dataset lineage: {dataset_table}")
        except Exception as e:
            print(f"Warning: Could not log dataset lineage: {e}")

    spec = importlib.util.spec_from_file_location("user_training", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_training"] = module
    spec.loader.exec_module(module)

    run_id = run.info.run_id
    print(f"Training complete. MLflow run: {run_id}")

# COMMAND ----------

# 4. Register the model in Unity Catalog
client = MlflowClient()

# Find the model artifact logged by autolog
artifacts = [a.path for a in client.list_artifacts(run_id) if a.is_dir]
model_artifact = "model"
for a in artifacts:
    if "model" in a.lower():
        model_artifact = a
        break

model_uri = f"runs:/{run_id}/{model_artifact}"
mv = mlflow.register_model(model_uri, model_name)
print(f"Registered model: {model_name} version {mv.version}")

# COMMAND ----------

# 5. Set "challenger" alias so validation can evaluate it
client.set_registered_model_alias(model_name, "challenger", mv.version)
print(f"Set 'challenger' alias on version {mv.version}")
