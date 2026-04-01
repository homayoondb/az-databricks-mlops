"""Single source of truth for project MLOps configuration."""

# -- Model Registry (Unity Catalog) --
MODEL_NAME = "messy-ml-project"
SCHEMA_NAME = "az_brand_assistant"

# -- MLflow Experiment --
# In production, the experiment path is set via bundle variables.
# This default is for local / interactive development.
EXPERIMENT_BASE_NAME = "messy-ml-project-experiment"

# -- Inference --
INPUT_TABLE = "inference_input"
OUTPUT_TABLE = "predictions"
