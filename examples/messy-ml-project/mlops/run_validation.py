# Databricks notebook source
"""Run model validation using mlflow.evaluate().

This notebook is called by the ModelValidation task in the training workflow.
It loads thresholds from validation.py and evaluates the latest model version.

Parameters (passed via job config):
  - experiment_name: MLflow experiment path
  - model_name: Full UC model name (catalog.schema.model)
  - run_mode: disabled | dry_run | enabled
  - model_type: regressor | classifier
"""

import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

# Add project root to sys.path so `mlops` package imports work
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
project_root = Path("/Workspace" + context.notebookPath().get()).parent.parent
sys.path.insert(0, str(project_root))

from mlops.validation import custom_metrics, evaluator_config, validation_thresholds

# COMMAND ----------

dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("run_mode", "dry_run")
dbutils.widgets.text("model_type", "regressor")

experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
run_mode = dbutils.widgets.get("run_mode")
model_type = dbutils.widgets.get("model_type")

# COMMAND ----------

if run_mode == "disabled":
    print("Validation disabled — skipping.")
    dbutils.notebook.exit(0)

# COMMAND ----------

# Get the latest model version from the Train task
client = MlflowClient()
latest_version = client.get_model_version_by_alias(model_name, "challenger")
model_uri = f"models:/{model_name}@challenger"
print(f"Validating model: {model_name} version {latest_version.version}")

# COMMAND ----------

# Load validation data (same as training data by default — customize as needed)
mlflow.set_experiment(experiment_name)
run = client.get_run(latest_version.run_id)
training_data_path = run.data.params.get("training_data_path", "")

if training_data_path:
    validation_data = spark.read.format("delta").load(training_data_path)
else:
    print("WARNING: No training_data_path found in run params. Using empty validation.")
    dbutils.notebook.exit(0)

# COMMAND ----------

# Run evaluation
thresholds = validation_thresholds()
metrics = custom_metrics()
eval_config = evaluator_config()

try:
    result = mlflow.evaluate(
        model=model_uri,
        data=validation_data.toPandas(),
        model_type=model_type,
        custom_metrics=metrics,
        validation_thresholds=thresholds,
        evaluator_config=eval_config,
    )
    print(f"Validation passed. Metrics: {result.metrics}")
except Exception as e:
    if run_mode == "dry_run":
        print(f"Validation FAILED (dry_run — not blocking): {e}")
    else:
        raise
