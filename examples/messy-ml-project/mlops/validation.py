"""Model validation thresholds and custom metrics.

This module is called by the ModelValidation task in the training workflow.
It uses mlflow.evaluate() to check that a newly trained model meets quality
thresholds before it is promoted.

Customize the thresholds and metrics below for your use case.
"""

import numpy as np
from mlflow.models import MetricThreshold, make_metric


def custom_metrics():
    """Return a list of custom MLflow metrics for model evaluation.

    Each metric is created with `make_metric()`. The evaluation function
    receives a DataFrame with 'prediction' and 'target' columns.
    """

    def squared_diff_plus_one(eval_df, _builtin_metrics):
        return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)

    return [
        make_metric(
            eval_fn=squared_diff_plus_one,
            greater_is_better=False,
            name="squared_diff_plus_one",
        ),
    ]


def validation_thresholds():
    """Return metric thresholds that the model must pass.

    Keys must match either built-in mlflow.evaluate() metric names
    or names from custom_metrics() above.
    """
    return {
        "max_error": MetricThreshold(threshold=500, greater_is_better=False),
        "mean_squared_error": MetricThreshold(threshold=500, greater_is_better=False),
    }


def evaluator_config():
    """Return additional evaluator configuration for mlflow.evaluate()."""
    return {}
