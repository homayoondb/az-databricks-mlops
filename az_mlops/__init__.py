"""Backward-compatible imports for older az-mlops installs."""

from as_databricks_mlops import __version__, run_training_job

__all__ = ["__version__", "run_training_job"]
