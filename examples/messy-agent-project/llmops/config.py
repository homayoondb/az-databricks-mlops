"""Single source of truth for project LLMOps configuration."""

# -- Model Registry (Unity Catalog) --
MODEL_NAME = "us_comm_lakehouse_dev.az_brand_assistant.messy_agent_project"
SCHEMA_NAME = "az_brand_assistant"

# -- MLflow Experiment --
EXPERIMENT_BASE_NAME = "messy-agent-project-experiment"

# -- Agent Evaluation --
# Name of the evaluation dataset table in Unity Catalog.
# Create this table with columns: inputs (STRING), expected_response (STRING).
EVAL_DATASET_TABLE = ""  # e.g. us_comm_lakehouse_dev.az_brand_assistant.agent_eval_dataset

# -- Agent Serving --
INPUT_TABLE = "us_comm_lakehouse_dev.az_brand_assistant.agent_input"
OUTPUT_TABLE = "us_comm_lakehouse_dev.az_brand_assistant.agent_output"
