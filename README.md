# as-databricks-mlops

Lightweight MLOps scaffolding for Databricks projects. Adds production-grade MLOps to any ML project in ~10 files — no complex templates, no 20-question wizards, no code changes to your existing scripts.

The published package name is `as-databricks-mlops`. The CLI command is `adm`.

Built for AWS Databricks with Unity Catalog.

## Why?

Most ML teams have working models but no production pipeline. Setting up MLOps from scratch means figuring out Databricks Asset Bundles, model registration, validation, and deployment promotion — all from blank files.

`as-databricks-mlops` does it in one command. You point it at your training script, answer a few prompts, and get a production-ready pipeline. **Your existing training code stays untouched** — the tool wraps it with automatic MLflow tracking, model registration, and validation.

## Installation

```bash
# Install directly from GitHub (recommended)
pip install git+https://github.com/<org>/as-databricks-mlops.git

# Or clone and install locally
git clone https://github.com/<org>/as-databricks-mlops.git
pip install ./as-databricks-mlops
```

## The complete flow

Two commands take you from messy ML project to tracked experiment in Databricks:

```
adm init    →    adm run    →    open the experiment URL
 (prompts)     (deploy + train)    (MLflow tracking in Databricks)
```

### Step 1 — `adm init`

Add MLOps scaffolding to your existing project:

```
$ cd my_ml_project
$ adm init

Project name [my_ml_project]:
Staging workspace URL: https://xxx.cloud.databricks.com
Prod workspace URL (enter to skip):

  Found notebooks/scripts in your project:
    1. notebooks/train_model.py
    2. notebooks/exploration.ipynb
    3. src/preprocess.py

  Training notebook/script (number or path) [train.py]: 1
  Include batch inference job? [Y/n]: n

  Created databricks.yml
  Created resources/training-job.yml
  Created mlops/run_training.py
  Created mlops/config.py
  Created mlops/validation.py
  ...

Running `databricks bundle validate`...
  Bundle is valid.

Next: run `adm run` to deploy and start your first training job.
```

`adm init` now supports three layers of defaults:

1. CLI flags win if you pass them explicitly
2. `adm.yml` values become the prompt defaults if present
3. If `adm.yml` leaves a field blank, `adm` falls back to helpful built-in defaults like the current folder name, notebook discovery, and any existing `databricks.yml` values

You still see the interactive prompts in all cases, so you can accept the default with Enter or type a different value on the spot.

### Optional `adm.yml`

If you scaffold multiple projects into the same workspace, keep an `adm.yml` in the repo root and let `adm init` or `adm new` auto-discover it:

```yaml
# Example defaults for this repo.
project_name:

databricks:
  staging_url: https://az-commercial-us-lakehouse-dev.cloud.databricks.com
  prod_url: ""
  catalog_name: us_comm_lakehouse_dev
  schema_name: az_brand_assistant

training:
  training_notebook:
  skip_inference: false

options:
  with_dqx: false
```

`adm.yml` is optional. If you want a different filename or location, pass it explicitly:

```bash
adm init --config path/to/adm.yml
adm new --config path/to/adm.yml
```

### Step 2 — run the training job

At the end of `adm init`, you'll be asked:

```
  Run the training job now? [y/N]:
```

Press `y` to deploy and start immediately, or press Enter to skip and run later.

To run later, either use the convenience command:

```bash
adm run
```

Or run the standard Databricks commands directly (same thing under the hood):

```bash
databricks bundle deploy -t dev
databricks bundle run model_training_job -t dev
```

Both print the same output:

```
  Job run:    https://xxx.cloud.databricks.com/jobs/123456/runs/789
  Experiment: https://xxx.cloud.databricks.com/#mlflow/experiments?searchFilter=...
```

Open the **Job run** URL to watch the pipeline execute. Open the **Experiment** URL to see MLflow tracking — metrics, parameters, and the registered model — in real time.

### What the training pipeline does

The job runs three tasks in sequence:

```
Train  →  ModelValidation  →  ModelDeployment
```

1. **Train** — wraps your training script with `mlflow.autolog()`, logs metrics/params/model automatically, registers the model in Unity Catalog as `challenger`
2. **ModelValidation** — evaluates the challenger against configurable thresholds (edit `mlops/validation.py` to set your metrics)
3. **ModelDeployment** — if validation passes, promotes the model to `champion` in Unity Catalog

If you added DQX, a **DataQuality** task runs first and blocks training if data checks fail.

---

## What gets generated

```
my_project/
├── .gitignore
├── databricks.yml              # Bundle config: dev/staging/prod targets
├── resources/
│   ├── training-job.yml        # Train → Validate → Deploy workflow
│   └── inference-job.yml       # Scheduled batch inference (optional)
├── mlops/
│   ├── __init__.py
│   ├── config.py               # Project config (model name, catalog, schema)
│   ├── run_training.py         # Wraps YOUR script with MLflow + UC registration
│   ├── validation.py           # Model quality thresholds (customizable)
│   ├── run_validation.py       # Runs mlflow.evaluate() with thresholds
│   ├── deploy.py               # UC alias promotion logic
│   ├── run_deploy.py           # Runs promotion (challenger → champion)
│   └── run_inference.py        # Loads champion model, scores input table
└── [existing ML code untouched]
```

**Your existing code is never modified** — only new files are added alongside it.

## How it works (zero code changes)

The key insight: `mlops/run_training.py` wraps your existing training script automatically:

1. Sets up the MLflow experiment
2. Enables `mlflow.autolog()` — captures model, metrics, and parameters
3. Executes your training script (unchanged)
4. Registers the logged model in Unity Catalog
5. Sets the "challenger" alias for validation

Your training script doesn't need `import mlflow` or any modifications. The wrapper handles everything.

Similarly, `mlops/run_inference.py` loads the champion model from Unity Catalog and scores your input table — no inference code changes needed.
If you enable batch inference, `adm` generates that generic wrapper directly; it does not execute an existing `predict.py` from your project.

## Try it on a real (messy) project

The `examples/messy-ml-project/` directory is a deliberately sloppy ML project — scattered notebooks, dead code, pickle files. Run the demo to see `adm` add MLOps to it without touching a single existing file:

```bash
bash examples/demo.sh
```

Or manually:

```bash
cd examples/messy-ml-project
python notebooks/train_model_v3_FINAL.py   # verify it works first
adm init                                    # add MLOps scaffolding
adm run                                     # deploy + run, get experiment URL
```

## Commands

### `adm init`

Add MLOps scaffolding to the current directory.

```bash
# Interactive (discovers your notebooks, prompts for choices)
adm init

# Non-interactive
adm init \
  --project-name my_project \
  --staging-url https://staging.cloud.databricks.com \
  --training-notebook notebooks/train.py \
  --skip-inference
```

Options:
- `--config` — path to an `adm.yml` defaults file (auto-discovered if omitted)
- `--training-notebook` — path to your training script (prompted interactively if omitted)
- `--skip-inference` — skip batch inference job entirely
- `--with-dqx` — include DQX data quality checks
- `--overwrite` — replace existing generated files

### `adm run`

Deploy the bundle and start the training job. Prints direct URLs to the job run and MLflow experiment.

```bash
adm run              # deploys to dev (default)
adm run --target staging
```

### `adm clean`

Remove all generated files from the current directory while keeping your own project files.

```bash
adm clean    # removes generated files, keeps your code
adm init     # start fresh
```

### `adm new <name>`

Create a new project directory with MLOps scaffolding.

```bash
adm new my_project --staging-url https://staging.cloud.databricks.com
```

### `adm add dqx`

Add DQX data quality checks to an existing project scaffolded by `adm`.

```bash
adm add dqx
```

Generates `mlops/dqx_checks.py` and `resources/dqx-job.yml`. When added at init time with `--with-dqx`, the training job automatically depends on the data quality check passing.

## Customization

- **Validation thresholds** — edit `mlops/validation.py` (metric names, threshold values)
- **Cluster config** — edit node type, workers, Spark version in resource YAMLs
- **Schedule** — edit the `schedule` block in any resource YAML (cron expression)
- **Model promotion logic** — edit `mlops/deploy.py`
- **DQX rules** — edit `mlops/dqx_checks.py` (add/remove column-level checks)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
