# Databricks notebook source
"""Promote a validated agent via Unity Catalog aliases.

Parameters (passed via job config):
  - env: Deployment target (dev, staging, prod)
  - model_name: Full UC model name (catalog.schema.model)
"""

import sys
from pathlib import Path

# Add project root to sys.path so `llmops` package imports work
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
project_root = Path("/Workspace" + context.notebookPath().get()).parent.parent
sys.path.insert(0, str(project_root))

from llmops.deploy import promote_model

# COMMAND ----------

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("model_name", "")

env = dbutils.widgets.get("env")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

promote_model(model_name, env)
