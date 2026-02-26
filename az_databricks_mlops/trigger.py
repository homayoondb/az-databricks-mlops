"""SDK-based job trigger for notebook-native execution."""

from __future__ import annotations


def run_training_job(job_name: str, params: dict[str, str] | None = None) -> None:
    """Find and trigger a deployed Databricks job by name.

    Authenticates automatically when called from inside a Databricks notebook
    (uses the cluster's ambient credentials via the SDK).

    Args:
        job_name: Exact name of the job (e.g. "dev-my-project-model-training-job").
        params: Optional notebook parameter overrides for this run.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    matching = [j for j in w.jobs.list() if j.settings and j.settings.name == job_name]
    if not matching:
        raise ValueError(f"No job found with name: {job_name!r}")
    if len(matching) > 1:
        raise ValueError(f"Multiple jobs found with name: {job_name!r}. Provide a unique name.")

    job_id = matching[0].job_id
    run_kwargs: dict = {}
    if params:
        run_kwargs["notebook_params"] = params

    response = w.jobs.run_now(job_id=job_id, **run_kwargs)
    print(f"Job '{job_name}' triggered. Run ID: {response.run_id}")
    print(f"Monitor at: {w.config.host}#job/{job_id}/run/{response.run_id}")
