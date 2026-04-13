# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Tests for `az-databricks-mlops` in Notebook Context
# MAGIC
# MAGIC This notebook validates that the package works when pip-installed inside a
# MAGIC Databricks workspace notebook. It tests:
# MAGIC 1. Package installation and import
# MAGIC 2. `review_repository()` on a cloned repo
# MAGIC 3. `run_training_job()` triggering a deployed job
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - A Databricks Model Serving endpoint must be READY (e.g. `databricks-gpt-5-4-mini`)
# MAGIC - For the trigger test: a training job must already be deployed via `adm run`

# COMMAND ----------

# MAGIC %pip install git+https://github.com/homayoon-moradi_data/az-databricks-mlops.git
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Package import and version

# COMMAND ----------

import az_databricks_mlops

print(f"Package version: {az_databricks_mlops.__version__}")
assert hasattr(az_databricks_mlops, "review_repository"), "review_repository not exported"
assert hasattr(az_databricks_mlops, "run_training_job"), "run_training_job not exported"
print("PASS: Package imports and exports are correct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Snapshot collection (no LLM call)

# COMMAND ----------

import subprocess
import tempfile
from pathlib import Path

from az_databricks_mlops.review import collect_repository_snapshot

with tempfile.TemporaryDirectory(prefix="adm-test-") as tmp:
    tmp_path = Path(tmp)
    # Create a minimal project
    (tmp_path / "train.py").write_text("import sklearn\nmodel = sklearn.linear_model.LinearRegression()\n")
    (tmp_path / "README.md").write_text("# Test project\n")
    (tmp_path / "model.pkl").write_bytes(b"\x80\x04\x95\x00")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")

    snapshot = collect_repository_snapshot(tmp_path, source_label="test-project")

    included = [f.path for f in snapshot.files]
    print(f"Included files: {included}")
    assert "train.py" in included, f"train.py not included: {included}"
    assert "README.md" in included, f"README.md not included: {included}"
    assert "model.pkl" not in included, "model.pkl should be excluded (binary)"

    omitted = {f.path: f.reason for f in snapshot.omitted_files}
    print(f"Omitted files: {omitted}")
    assert "model.pkl" in omitted, "model.pkl should appear in omitted"

print("PASS: Snapshot collection works correctly in notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Clone a remote repo and collect snapshot

# COMMAND ----------

import subprocess
import tempfile
from pathlib import Path

from az_databricks_mlops.review import collect_repository_snapshot

# Clone a small repo to test git-based file iteration
with tempfile.TemporaryDirectory(prefix="adm-clone-") as tmp:
    clone_target = Path(tmp) / "repo"
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/homayoon-moradi_data/az-databricks-mlops.git", str(clone_target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Clone failed (may be private): {result.stderr}")
        print("Falling back to local temp project for snapshot test")
        clone_target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=clone_target, check=True, capture_output=True)
        (clone_target / "fallback.py").write_text("x = 1\n")
        (clone_target / ".gitignore").write_text("*.log\n")
        (clone_target / "debug.log").write_text("noise\n")
        subprocess.run(["git", "add", "."], cwd=clone_target, check=True, capture_output=True)

    snapshot = collect_repository_snapshot(clone_target, source_label="remote-test")
    print(f"Collected {len(snapshot.files)} files, {snapshot.total_characters} chars")
    assert len(snapshot.files) > 0, "No files collected from cloned repo"

    # If it's a real git repo, gitignore should be respected
    included_paths = {f.path for f in snapshot.files}
    assert "debug.log" not in included_paths, "debug.log should be gitignored"

print("PASS: Remote clone + snapshot works in notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 4: Full `review_repository()` with LLM endpoint

# COMMAND ----------

import tempfile
from pathlib import Path

from az_databricks_mlops import review_repository

with tempfile.TemporaryDirectory(prefix="adm-review-") as tmp:
    tmp_path = Path(tmp)
    # Create a minimal ML project to review
    src = tmp_path / "src"
    src.mkdir()
    (src / "train.py").write_text(
        "import pandas as pd\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "df = pd.read_csv('data.csv')\n"
        "model = RandomForestClassifier()\n"
        "model.fit(df.drop('target', axis=1), df['target'])\n"
    )
    (src / "config.py").write_text("MODEL_NAME = 'my_model'\nCATALOG = 'main'\n")

    output = tmp_path / "review.md"
    artifact = review_repository(
        source=str(src),
        output_path=output,
        working_directory=tmp_path,
        max_file_chars=50_000,
        max_total_chars=100_000,
    )

    assert output.exists(), f"Review output not created at {output}"
    content = output.read_text()
    assert "<!-- Generated by adm document -->" in content, "Missing header in review"
    assert len(content) > 200, f"Review suspiciously short: {len(content)} chars"
    print(f"Review generated: {len(content)} chars")
    print(f"Endpoint used: {artifact.endpoint_name}")
    print(f"Files reviewed: {len(artifact.snapshot.files)}")

print("PASS: review_repository() works end-to-end in notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 5: `run_training_job()` (requires a deployed job)
# MAGIC
# MAGIC Set the widget below to the job name, or leave empty to skip.
# MAGIC Use the short name (e.g. `dev-myproject-model-training-job`) — the trigger
# MAGIC handles DAB development mode's `[dev username]` prefix automatically via suffix matching.

# COMMAND ----------

dbutils.widgets.text("job_name", "", "Training job name (leave empty to skip)")

# COMMAND ----------

job_name = dbutils.widgets.get("job_name").strip()

if not job_name:
    print("SKIP: No job_name provided — skipping trigger test")
else:
    from az_databricks_mlops import run_training_job

    try:
        run_training_job(job_name)
        print(f"PASS: run_training_job('{job_name}') triggered successfully")
    except ValueError as e:
        print(f"FAIL: {e}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("All notebook E2E tests completed.")
print("=" * 60)
