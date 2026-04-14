# Databricks notebook source
"""Run agent evaluation using mlflow.genai.evaluate().

This notebook is called by the AgentEval task in the agent workflow.
It loads scorers from scorers.py and evaluates the latest agent version.

Parameters (passed via job config):
  - experiment_name: MLflow experiment path
  - model_name: Full UC model name (catalog.schema.model)
  - run_mode: disabled | dry_run | enabled
  - eval_dataset: Optional UC table with evaluation data
"""

import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
project_root = Path("/Workspace" + context.notebookPath().get()).parent.parent
sys.path.insert(0, str(project_root))

from llmops.scorers import get_scorers

# COMMAND ----------

dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("run_mode", "dry_run")
dbutils.widgets.text("eval_dataset", "")

experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
run_mode = dbutils.widgets.get("run_mode")
eval_dataset = dbutils.widgets.get("eval_dataset")

# COMMAND ----------

if run_mode == "disabled":
    print("Evaluation disabled — skipping.")
    dbutils.notebook.exit(0)

# COMMAND ----------

client = MlflowClient()
latest_version = client.get_model_version_by_alias(model_name, "challenger")
model_uri = f"models:/{model_name}@challenger"
print(f"Evaluating agent: {model_name} version {latest_version.version}")

# COMMAND ----------

mlflow.set_experiment(experiment_name)

# Build evaluation data.
if eval_dataset:
    eval_data = spark.table(eval_dataset).toPandas()
    print(f"Loaded {len(eval_data)} evaluation examples from {eval_dataset}")
else:
    import pandas as pd
    eval_data = pd.DataFrame({
        "inputs": [
            "What can you help me with?",
            "Tell me about the project.",
            "Summarize the key findings.",
        ],
        "expected_response": [
            "",
            "",
            "",
        ],
    })
    print(f"Using {len(eval_data)} built-in evaluation examples (customize eval_dataset parameter)")

# COMMAND ----------

scorers = get_scorers()
print(f"Running evaluation with {len(scorers)} scorers...")

try:
    result = mlflow.genai.evaluate(
        model=model_uri,
        data=eval_data,
        scorers=scorers,
    )
    print(f"Evaluation complete. Metrics: {result.metrics}")
    print(f"See detailed results in the MLflow experiment: {experiment_name}")
except Exception as e:
    if run_mode == "dry_run":
        print(f"Evaluation FAILED (dry_run — not blocking): {e}")
    else:
        raise
