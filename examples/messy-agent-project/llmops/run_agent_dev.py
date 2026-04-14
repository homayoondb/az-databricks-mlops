# Databricks notebook source
"""LLMOps agent development wrapper.

Wraps your existing agent script (agent.py) with
MLflow 3 tracing and Unity Catalog model registration.
Your agent code does NOT need any modifications beyond exposing a callable.

What this does automatically:
  1. Sets up the MLflow experiment
  2. Enables MLflow tracing (captures all LLM calls, tool invocations, retrieval)
  3. Loads your agent script, which must define an `agent` callable or a `predict` function
  4. Logs the agent as a PyFunc model in MLflow
  5. Registers the model in Unity Catalog
  6. Sets the "challenger" alias for evaluation

Agent script contract:
  Your script (agent.py) must expose one of:
    - agent = some_callable   # any callable that takes a string and returns a string
    - def predict(model_input): ...
  Any framework works (LangChain, LangGraph, custom code, etc.).
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

experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
env = dbutils.widgets.get("env")

# COMMAND ----------

# Ensure the workspace directory exists before MLflow tries to create the experiment.
parent_dir = str(PurePosixPath(experiment_name).parent)
if parent_dir and parent_dir != ".":
    WorkspaceClient().workspace.mkdirs(parent_dir)

# COMMAND ----------

# 1. Configure MLflow
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# 2. Resolve the project root from this notebook's workspace path.
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_workspace_path = context.notebookPath().get()
project_root = Path("/Workspace" + notebook_workspace_path).parent.parent
script_path = project_root / "agent.py"
print(f"Loading agent script from: {script_path}")

os.chdir(project_root)

# COMMAND ----------

# 3. Load the user's agent module.
spec = importlib.util.spec_from_file_location("user_agent", script_path)
module = importlib.util.module_from_spec(spec)
sys.modules["user_agent"] = module
spec.loader.exec_module(module)

# Discover the agent callable.
agent_fn = getattr(module, "agent", None) or getattr(module, "predict", None)
if agent_fn is None:
    raise AttributeError(
        f"Agent script {script_path} must define either an `agent` variable "
        f"or a `predict()` function."
    )

# COMMAND ----------

# 4. Log the agent as an MLflow PyFunc model with tracing.
with mlflow.start_run() as run:
    # Enable tracing for this run — captures all LLM calls automatically.
    mlflow.tracing.enable()

    # Run a sample invocation to verify the agent works and capture a trace.
    sample_input = "Hello, can you help me?"
    try:
        sample_output = agent_fn(sample_input)
        print(f"Sample invocation succeeded: {str(sample_output)[:200]}")
    except Exception as e:
        print(f"Warning: sample invocation failed: {e}")

    # Log the agent as a PyFunc model.
    # Unity Catalog requires a model signature; infer it from the sample I/O.
    from mlflow.models.signature import infer_signature
    signature = infer_signature(sample_input, sample_output)

    mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=agent_fn,
        input_example=sample_input,
        signature=signature,
    )

    run_id = run.info.run_id
    print(f"Agent logged. MLflow run: {run_id}")

# COMMAND ----------

# 5. Register the model in Unity Catalog and set "challenger" alias.
client = MlflowClient()
model_uri = f"runs:/{run_id}/agent"
mv = mlflow.register_model(model_uri, model_name)
print(f"Registered model: {model_name} version {mv.version}")

client.set_registered_model_alias(model_name, "challenger", mv.version)
print(f"Set 'challenger' alias on version {mv.version}")
